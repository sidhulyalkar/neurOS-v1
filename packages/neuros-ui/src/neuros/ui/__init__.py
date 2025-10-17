"""
neurOS UI Package
=================

User interfaces including dashboard, API server, and visualizations.
"""

# Conditional imports for optional UI components
try:
    from neuros.dashboard import run_dashboard
except ImportError:
    run_dashboard = None

try:
    from neuros.api.server import create_app
except ImportError:
    create_app = None

__all__ = [
    "run_dashboard",
    "create_app",
]

__version__ = "2.0.0"
