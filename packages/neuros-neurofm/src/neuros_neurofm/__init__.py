"""NeuroFM-X: Neural Foundation Model for Population Dynamics."""

__version__ = "0.1.0"
__author__ = "neurOS Team"
__license__ = "MIT"

from neuros_neurofm.models.neurofmx_complete import NeuroFMXComplete
from neuros_neurofm.models.neurofmx_multitask import NeuroFMXMultiTask

__all__ = [
    "NeuroFMXComplete",
    "NeuroFMXMultiTask",
    "__version__",
]
