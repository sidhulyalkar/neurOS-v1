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

from datetime import datetime
import os

from ..drivers.mock_driver import MockDriver
from ..pipeline import Pipeline
from ..agents.orchestrator_agent import Orchestrator
from ..cloud import CloudStorage, LocalStorage
from ..db.database import Database
from ..security import require_role, get_token_info, load_token_map


app = FastAPI(title="neurOS API", version="2.0")


class TrainRequest(BaseModel):
    features: List[List[float]]
    labels: List[int]


# global model storage (in-memory); in a real deployment this would be persisted
_global_model: Optional[SimpleClassifier] = None
_global_fs: float = 250.0
_global_channels: int = 8

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
    run_id = datetime.utcnow().isoformat(timespec="seconds")
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
            run_id = datetime.utcnow().isoformat(timespec="seconds")
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
    run_id = datetime.utcnow().isoformat(timespec="seconds")
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