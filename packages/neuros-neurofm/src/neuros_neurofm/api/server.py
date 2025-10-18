"""
FastAPI server for NeuroFM-X inference.

Provides REST API for real-time neural decoding.
"""

import os
import time
from typing import Dict, List, Optional, Union
from pathlib import Path

try:
    from fastapi import FastAPI, HTTPException, BackgroundTasks
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    JSONResponse = None
    BaseModel = object
    Field = lambda *args, **kwargs: None

import numpy as np
import torch

from neuros_neurofm.inference.realtime_pipeline import RealtimeInferencePipeline


# Request/Response models
class PredictRequest(BaseModel):
    """Inference request schema."""
    data: List[List[float]] = Field(..., description="Input neural data (batch x features x time)")
    request_id: Optional[str] = Field(None, description="Optional request ID")
    timeout: float = Field(1.0, description="Timeout in seconds", ge=0.1, le=30.0)


class PredictResponse(BaseModel):
    """Inference response schema."""
    request_id: str
    predictions: List[float]
    latency_ms: float
    batch_size: int
    timestamp: float


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model_loaded: bool
    device: str
    uptime_seconds: float


class StatsResponse(BaseModel):
    """Statistics response."""
    latency: Dict[str, float]
    batch_latency: Dict[str, float]
    end_to_end: Dict[str, float]
    queue_size: int


# Global state
app = None
pipeline = None
start_time = None


def create_app(
    model_path: Optional[str] = None,
    device: str = 'cpu',
    max_batch_size: int = 32,
    max_wait_ms: float = 10.0,
) -> FastAPI:
    """Create FastAPI application.

    Parameters
    ----------
    model_path : str, optional
        Path to model checkpoint.
    device : str, optional
        Device to run on.
    max_batch_size : int, optional
        Maximum batch size.
    max_wait_ms : float, optional
        Maximum batching wait time.

    Returns
    -------
    FastAPI
        FastAPI application.
    """
    if not FASTAPI_AVAILABLE:
        raise ImportError(
            "fastapi is required for API server. "
            "Install with: pip install fastapi uvicorn"
        )

    global app, pipeline, start_time

    app = FastAPI(
        title="NeuroFM-X API",
        description="Real-time neural decoding with NeuroFM-X foundation model",
        version="1.0.0",
    )

    start_time = time.time()

    # Load model
    if model_path is not None:
        print(f"Loading model from {model_path}...")
        model = torch.jit.load(model_path)
    else:
        # Use dummy model for testing
        print("No model path provided, using dummy model")
        from neuros_neurofm.models.simple_neurofmx import SimpleNeuroFMX
        model = SimpleNeuroFMX(d_model=256, n_blocks=4)

    # Create pipeline
    pipeline = RealtimeInferencePipeline(
        model=model,
        device=device,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )

    # Start pipeline
    example_input = torch.randn(1, 96, 100)  # Example shape
    pipeline.start(example_input)

    @app.get("/", response_model=Dict[str, str])
    async def root():
        """Root endpoint."""
        return {
            "message": "NeuroFM-X API",
            "version": "1.0.0",
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(
            status="healthy" if pipeline.running else "unhealthy",
            model_loaded=pipeline is not None,
            device=device,
            uptime_seconds=time.time() - start_time,
        )

    @app.post("/predict", response_model=PredictResponse)
    async def predict(request: PredictRequest):
        """Inference endpoint.

        Parameters
        ----------
        request : PredictRequest
            Inference request with neural data.

        Returns
        -------
        PredictResponse
            Predictions and metadata.
        """
        if not pipeline.running:
            raise HTTPException(status_code=503, detail="Pipeline not running")

        try:
            # Convert data to tensor
            data = torch.tensor(request.data, dtype=torch.float32)

            # Ensure correct shape (batch=1)
            if data.ndim == 2:
                data = data.unsqueeze(0)

            # Run inference
            result = pipeline.predict(
                data=data,
                request_id=request.request_id,
                timeout=request.timeout,
            )

            if result is None:
                raise HTTPException(status_code=408, detail="Request timeout")

            # Convert predictions to list
            if isinstance(result.predictions, dict):
                # Multi-task output
                predictions = result.predictions['decoder'].cpu().numpy().tolist()
            else:
                predictions = result.predictions.cpu().numpy().tolist()

            return PredictResponse(
                request_id=result.request_id,
                predictions=predictions,
                latency_ms=result.latency_ms,
                batch_size=result.batch_size,
                timestamp=time.time(),
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/stats", response_model=StatsResponse)
    async def stats():
        """Get pipeline statistics."""
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        stats = pipeline.get_stats()
        return StatsResponse(**stats)

    @app.post("/reset-stats")
    async def reset_stats():
        """Reset statistics."""
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")

        pipeline.profiler.reset()
        return {"message": "Statistics reset"}

    @app.on_event("shutdown")
    async def shutdown():
        """Shutdown handler."""
        if pipeline is not None:
            pipeline.stop()

    return app


def main():
    """Run API server."""
    import argparse

    parser = argparse.ArgumentParser(description="NeuroFM-X API Server")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--max-batch-size", type=int, default=32, help="Max batch size")
    parser.add_argument("--max-wait-ms", type=float, default=10.0, help="Max wait time (ms)")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")

    args = parser.parse_args()

    # Read from environment variables
    model_path = args.model_path or os.getenv("NEUROFM_MODEL_PATH")
    device = args.device or os.getenv("NEUROFM_DEVICE", "cpu")
    max_batch_size = int(os.getenv("NEUROFM_BATCH_SIZE", args.max_batch_size))
    max_wait_ms = float(os.getenv("NEUROFM_MAX_WAIT_MS", args.max_wait_ms))

    # Create app
    app = create_app(
        model_path=model_path,
        device=device,
        max_batch_size=max_batch_size,
        max_wait_ms=max_wait_ms,
    )

    # Run server
    try:
        import uvicorn
    except ImportError:
        raise ImportError(
            "uvicorn is required to run the server. "
            "Install with: pip install uvicorn"
        )

    print(f"Starting server on {args.host}:{args.port}")
    print(f"Device: {device}")
    print(f"Max batch size: {max_batch_size}")
    print(f"Max wait time: {max_wait_ms}ms")

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
