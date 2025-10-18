"""
API module for NeuroFM-X.

Provides REST API for real-time neural decoding.
"""

try:
    from neuros_neurofm.api.server import create_app, main
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    create_app = None
    main = None

__all__ = ['create_app', 'main', 'FASTAPI_AVAILABLE']
