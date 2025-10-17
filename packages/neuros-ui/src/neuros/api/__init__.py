"""
API package for neurOS.

This package exposes a FastAPI application that can be used to control
pipelines remotely and stream classification results in real time.  It is
designed to run behind `uvicorn` or another ASGI server and requires
`fastapi` to be installed.
"""

from neuros.api.server import app  # noqa: F401
