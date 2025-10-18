"""
Integration modules for NeuroFM-X.

Provides adapters for using NeuroFM-X with external frameworks.
"""

from neuros_neurofm.integration.neuros_adapter import (
    NeuroFMXNeurOSAdapter,
    create_neuros_model,
)

__all__ = [
    "NeuroFMXNeurOSAdapter",
    "create_neuros_model",
]
