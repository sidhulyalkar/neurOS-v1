"""
FastAPI server for neurOS.

This module defines a FastAPI application that exposes endpoints for
training models, running the pipeline and streaming classification
results via WebSocket.  It allows neurOS to be integrated into
cloud‑native environments.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Header
from typing import Optional
from pydantic import BaseModel

from datetime import datetime, timezone
import os

from ..drivers.mock_driver import MockDriver
from ..pipeline import Pipeline
from ..agents.orchestrator_agent import Orchestrator
from ..cloud import CloudStorage, LocalStorage
from ..db.database import Database
from ..security import require_role, get_token_info, load_token_map

# Import classification model and numerical utilities for the inference endpoint.
# SimpleClassifier is used as the primary model when a model has been trained
# via the /train endpoint.  NumPy provides array operations for feature
# handling, and time/perf_counter is used to measure request latency.
from ..models.simple_classifier import SimpleClassifier  # type: ignore
import numpy as np  # type: ignore
import time
from prometheus_client import Counter as _Counter, Histogram as _Histogram, make_asgi_app


app = FastAPI(title="neurOS API", version="2.0")


class TrainRequest(BaseModel):
    features: List[List[float]]
    labels: List[int]


# global model storage (in-memory); in a real deployment this would be persisted
_global_model: Optional[SimpleClassifier] = None
_global_fs: float = 250.0
_global_channels: int = 8

# ---------------------------------------------------------------------------
# Inference schema, dummy model and metrics
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Schema for prediction requests.

    The ``features`` field should contain a flat list of floats
    representing a single feature vector.  An optional ``metadata``
    dictionary may include additional contextual information (ignored
    by the current models).  This schema mirrors the simple inference
    API defined in ``neuros/serve/api.py`` for backward compatibility.
    """

    features: List[float]
    metadata: Optional[Dict[str, str]] = None


class PredictResponse(BaseModel):
    """Schema for prediction responses.

    The response comprises a string label and a floating‑point
    confidence value between 0 and 1.  When using the default dummy
    model the labels correspond to nominal brain states; when using a
    trained classifier the labels are numeric class identifiers.
    """

    label: str
    confidence: float


class DummyBrainStateModel:
    """Fallback model that returns a random brain state.

    This class is used when no global model has been trained via the
    ``/train`` endpoint.  It samples uniformly from a fixed set of
    nominal brain states and assigns a random confidence between 0.5
    and 1.0.  Replace this with a real model for production use.
    """

    labels = ["awake", "drowsy", "focused", "stressed"]

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        label = str(np.random.choice(self.labels))
        confidence = float(np.random.uniform(0.5, 1.0))
        return label, confidence


# Prometheus metrics for the inference endpoint.  These counters and
# histograms mirror those used in the standalone inference API and
# allow monitoring request volume and latency.  Using separate
# label dimensions enables per‑endpoint aggregation.
INFERENCE_REQUEST_COUNT = _Counter(
    "neuros_inference_requests_total",
    "Total number of inference requests",
    ["endpoint"],
)
INFERENCE_LATENCY = _Histogram(
    "neuros_inference_request_latency_seconds",
    "Inference request latency in seconds",
    ["endpoint"],
)


# Initialise a fallback inference model.  When the global
# classifier is trained, it will replace this dummy model.  The type
# annotation is Union[DummyBrainStateModel, SimpleClassifier], but
# Optional is avoided here to simplify isinstance checks.
_inference_model: DummyBrainStateModel | SimpleClassifier = DummyBrainStateModel()

# configure global storage backend based on environment
_storage: CloudStorage
_storage_type = os.getenv("NEUROS_STORAGE", "local").lower()
if _storage_type == "s3":  # pragma: no cover - requires boto3
    from ..cloud import S3Storage  # type: ignore

    bucket = os.getenv("NEUROS_S3_BUCKET")
    if not bucket:
        raise RuntimeError("NEUROS_S3_BUCKET environment variable must be set for S3 storage")
    prefix = os.getenv("NEUROS_S3_PREFIX", "")
    _storage = S3Storage(bucket=bucket, prefix=prefix)  # type: ignore
else:
    base_dir = os.getenv("NEUROS_LOCAL_DIR", os.path.expanduser("~/.neuros_runs"))
    _storage = LocalStorage(base_dir=base_dir)

# database instance for persistent metrics/results
_db = Database(os.getenv("NEUROS_DB_PATH", "neuros.db"))

# ---------------------------------------------------------------------------
# Security configuration
# ---------------------------------------------------------------------------
# Define role dependencies.  The default role map is loaded from
# NEUROS_API_KEYS_JSON or NEUROS_API_TOKEN/HASH environment variables.  When
# no key map is configured, all tokens are treated as admin.

# Preload token map on startup to avoid repeated parsing
_token_map = load_token_map()

def _info_from_header(authorization: Optional[str]) -> dict:
    """Helper to extract token info from Authorization header.

    Returns a dictionary with ``role`` and ``tenant`` fields.  Raises
    HTTPException on failure.
    """
    if authorization is None or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing Authorization header")
    token = authorization[len("Bearer "):].strip()
    try:
        return get_token_info(token)
    except PermissionError:
        raise HTTPException(status_code=401, detail="Invalid token")


@app.post("/train")
async def train_model(req: TrainRequest, info: dict = Depends(require_role(("admin", "trainer")))) -> Dict[str, Any]:
    """Train the global classifier with provided feature vectors and labels."""
    global _global_model
    X = req.features
    y = req.labels
    if len(X) != len(y):
        raise HTTPException(status_code=400, detail="Number of samples and labels must match")
    model = SimpleClassifier(max_iter=200)
    import numpy as np

    X_arr = np.array(X)
    y_arr = np.array(y)
    model.train(X_arr, y_arr)
    _global_model = model
    return {"status": "trained", "samples": len(X)}


class RunRequest(BaseModel):
    duration: float = 5.0


@app.post("/start")
async def start_pipeline(req: RunRequest, info: dict = Depends(require_role(("admin", "runner")))) -> Dict[str, Any]:
    """Run the pipeline for a fixed duration and return performance metrics."""
    if _global_model is None:
        raise HTTPException(status_code=400, detail="Model has not been trained")
    driver = MockDriver(sampling_rate=_global_fs, channels=_global_channels)
    pipeline = Pipeline(driver=driver, model=_global_model, fs=_global_fs)
    # run pipeline
    metrics = await pipeline.run(duration=req.duration)
    # generate a run identifier and persist metrics with tenant context
    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tenant_id = info.get("tenant", "default")
    try:
        _storage.upload_metrics(metrics, run_id)
    except Exception:
        pass
    try:
        _db.insert_run_metrics(
            run_id,
            metrics,
            tenant_id=tenant_id,
            driver=driver.__class__.__name__,
            model=_global_model.__class__.__name__ if _global_model else None,
            task="manual",
        )
    except Exception:
        pass
    try:
        # type: ignore[attr-defined]
        _storage.upload_database(_db.path)  # pyright: ignore[reportGeneralTypeIssues]
    except Exception:
        pass
    return {**metrics, "run_id": run_id}


@app.websocket("/stream")
async def stream(websocket: WebSocket, duration: float = 10.0) -> None:
    """Stream classification results in real time over WebSocket.

    Clients must connect via WebSocket.  Optionally a ``duration`` query
    parameter (seconds) limits how long the stream should run; a value of
    0 indicates an indefinite stream.  Results are sent as JSON objects with
    timestamp, label, confidence and latency fields.
    """
    # extract token and tenant for RBAC
    try:
        info = _info_from_header(websocket.headers.get("authorization"))
    except HTTPException as exc:
        await websocket.close(code=1008)
        return
    role = info.get("role")
    tenant_id = info.get("tenant", "default")
    # viewers and admins can stream; other roles are forbidden
    if role not in ("admin", "viewer"):
        await websocket.close(code=1008)
        return
    await websocket.accept()
    if _global_model is None:
        await websocket.send_json({"error": "Model not trained"})
        await websocket.close(code=1000)
        return
    driver = MockDriver(sampling_rate=_global_fs, channels=_global_channels)
    orchestrator = Orchestrator(
        driver=driver,
        model=_global_model,
        fs=_global_fs,
        duration=None,
        adaptation=True,
    )
    await orchestrator._start_agents()
    start_time = asyncio.get_event_loop().time()
    result_buffer: list[tuple[float, int, float, float]] = []
    try:
        async for ts, label, conf, latency in orchestrator.stream_results():
            msg = {
                "timestamp": ts,
                "label": label,
                "confidence": conf,
                "latency": latency,
            }
            await websocket.send_text(json.dumps(msg))
            result_buffer.append((ts, label, conf, latency))
            if duration and duration > 0 and (asyncio.get_event_loop().time() - start_time) >= duration:
                break
    except WebSocketDisconnect:
        pass
    finally:
        await orchestrator.stop()
        await websocket.close()
        # persist results if streaming completed normally
        if duration and duration > 0 and result_buffer:
            run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
            try:
                _storage.stream_results(result_buffer, run_id)
            except Exception:
                pass
            try:
                _db.insert_stream_results(run_id, result_buffer, tenant_id=tenant_id)
            except Exception:
                pass
            try:
                _storage.upload_database(_db.path)  # type: ignore[attr-defined]
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Additional endpoints for run management and pipeline auto‑configuration
# ---------------------------------------------------------------------------

class AutoConfigRequest(BaseModel):
    task_description: str
    model_name: Optional[str] = None
    use_brainflow: Optional[bool] = False
    duration: Optional[float] = 5.0


@app.post("/autoconfig")
async def autoconfig(req: AutoConfigRequest, info: dict = Depends(require_role(("admin", "runner")))) -> Dict[str, Any]:
    """Generate a pipeline for a task description and run it briefly.

    This endpoint automatically selects an appropriate model and frequency
    bands based on the provided task description, instantiates a pipeline,
    trains it on synthetic data and runs it for the specified duration.
    The resulting metrics and run ID are returned.
    """
    from ..autoconfig import generate_pipeline_for_task
    # create pipeline using heuristics
    pipeline = generate_pipeline_for_task(
        req.task_description,
        use_brainflow=req.use_brainflow or False,
        model_name=req.model_name,
    )
    # synthesize random training data for demonstration
    import numpy as np
    n_samples = 200
    # features depend on number of channels and heuristics: we approximate by 5×channels
    n_features = 5 * pipeline.driver.channels
    X_train = np.random.randn(n_samples, n_features)
    y_train = np.random.randint(0, 2, size=n_samples)
    pipeline.train(X_train, y_train)
    # run pipeline
    metrics = await pipeline.run(duration=req.duration or 5.0)
    run_id = datetime.now(timezone.utc).isoformat(timespec="seconds")
    tenant_id = info.get("tenant", "default")
    # persist metrics
    try:
        _storage.upload_metrics(metrics, run_id)
    except Exception:
        pass
    try:
        _db.insert_run_metrics(
            run_id,
            metrics,
            tenant_id=tenant_id,
            driver=pipeline.driver.__class__.__name__,
            model=pipeline.model.__class__.__name__ if pipeline.model else None,
            task=req.task_description,
        )
    except Exception:
        pass
    try:
        _storage.upload_database(_db.path)  # type: ignore[attr-defined]
    except Exception:
        pass
    return {**metrics, "run_id": run_id}


@app.get("/runs")
async def list_runs(info: dict = Depends(require_role(("admin", "viewer", "runner", "trainer")))) -> Dict[str, Any]:
    """Return a list of run identifiers with basic metrics for the caller's tenant."""
    tenant_id = info.get("tenant", "default")
    run_ids = _db.list_runs(tenant_id=tenant_id)
    runs: list[dict] = []
    for rid in run_ids:
        m = _db.get_run_metrics(rid, tenant_id=tenant_id)
        if m:
            runs.append(m)
    return {"runs": runs}


@app.get("/runs/{run_id}")
async def get_run(run_id: str, info: dict = Depends(require_role(("admin", "viewer", "runner", "trainer")))) -> Dict[str, Any]:
    """Return metrics and streaming results for a specific run (tenant‑scoped)."""
    tenant_id = info.get("tenant", "default")
    metrics = _db.get_run_metrics(run_id, tenant_id=tenant_id)
    if metrics is None:
        raise HTTPException(status_code=404, detail="Run not found")
    results = _db.get_stream_results(run_id, tenant_id=tenant_id)
    return {"metrics": metrics, "results": results}

# ---------------------------------------------------------------------------
# Run search endpoint for cross‑modal analysis
# ---------------------------------------------------------------------------

@app.get("/runs/search")
async def search_runs(
    driver: Optional[str] = None,
    model: Optional[str] = None,
    task: Optional[str] = None,
    info: dict = Depends(require_role(("admin", "viewer", "runner", "trainer"))),
) -> Dict[str, Any]:
    """Search runs by driver, model or task.

    Clients can specify one or more query parameters to filter runs
    belonging to their tenant.  All parameters are optional; if no
    filter is provided, all runs for the tenant are returned.

    Parameters
    ----------
    driver : str, optional
        Class name of the driver (e.g. "ECoGDriver").
    model : str, optional
        Class name of the model (e.g. "EEGNetModel").
    task : str, optional
        Task description substring used when configuring the pipeline.
    info : dict
        Injected token info containing role and tenant.

    Returns
    -------
    dict
        A dictionary with a single key "runs" mapping to a list of
        run metrics objects matching the filters.
    """
    tenant_id = info.get("tenant", "default")
    run_ids = _db.list_runs(tenant_id=tenant_id)
    filtered: list[dict] = []
    for rid in run_ids:
        m = _db.get_run_metrics(rid, tenant_id=tenant_id)
        if not m:
            continue
        # apply filters: skip runs not matching the provided criteria
        if driver and m.get("driver") != driver:
            continue
        if model and m.get("model") != model:
            continue
        if task and m.get("task") and task.lower() not in m.get("task", "").lower():
            continue
        filtered.append(m)
    return {"runs": filtered}


# ---------------------------------------------------------------------------
# Predict endpoint and Prometheus metrics
# ---------------------------------------------------------------------------

@app.post("/predict", response_model=PredictResponse)
async def predict(request: PredictRequest) -> PredictResponse:
    """Predict the brain state for a given feature vector.

    This endpoint mirrors the lightweight real‑time inference API
    originally defined in ``neuros/serve/api.py``.  When a model has
    been trained via the ``/train`` endpoint, that model is used to
    compute the class probabilities.  Otherwise a dummy model returns
    a random brain state.  Latency metrics are recorded with
    Prometheus.
    """
    if not request.features:
        raise HTTPException(status_code=400, detail="No features provided")
    start = time.perf_counter()
    x = np.array(request.features, dtype=np.float32).reshape(1, -1)
    # choose model: use trained global model if available; else fallback
    model = _global_model if _global_model is not None else _inference_model
    # compute prediction and confidence
    if isinstance(model, SimpleClassifier):
        # logistic regression supports predict_proba; fallback to predict if not available
        try:
            prob = model._model.predict_proba(x)[0]
            pred_idx = int(np.argmax(prob))
            label = str(pred_idx)
            confidence = float(np.max(prob))
        except Exception:
            pred_idx = int(model._model.predict(x)[0])
            label = str(pred_idx)
            confidence = 1.0
    else:
        label, confidence = model.predict(x)
    duration = time.perf_counter() - start
    INFERENCE_REQUEST_COUNT.labels(endpoint="/predict").inc()
    INFERENCE_LATENCY.labels(endpoint="/predict").observe(duration)
    return PredictResponse(label=label, confidence=confidence)

# Mount the Prometheus metrics endpoint at /metrics.  Exposing
# metrics allows scraping of both training/pipeline metrics and
# inference metrics with a single server.  ``make_asgi_app`` builds a
# Starlette application that exports metrics in the standard format.
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)