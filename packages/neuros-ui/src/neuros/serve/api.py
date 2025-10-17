"""
Alias to the unified neurOS API server.

This module re‑exports the FastAPI application defined in
``neuros.api.server`` so that existing references to
``neuros.serve.api:app`` continue to work.  All HTTP endpoints,
WebSocket handlers, role‑based access control and metrics are
implemented in the ``neuros.api.server`` module.  Additional schema
classes (``PredictRequest`` and ``PredictResponse``) are also
re‑exported for compatibility.
"""

from ..api.server import app, PredictRequest, PredictResponse

__all__ = ["app", "PredictRequest", "PredictResponse"]