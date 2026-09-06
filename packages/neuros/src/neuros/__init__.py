"""Public neurOS SDK namespace.

The implementation is split across installable distributions that share the
``neuros`` namespace. This module keeps the familiar top-level API while the
kernel remains independently installable.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from neuros.contracts import DecoderOutput, SignalFrame, StreamDescriptor  # noqa: E402,F401
from neuros.dataset import (  # noqa: E402,F401
    AlignedWindow,
    AlignmentPlan,
    DataWindow,
    Dataset,
    NativeRuntimeUnavailable,
    native_runtime_available,
)
from neuros.pipeline import MultiModalPipeline, Pipeline  # noqa: E402,F401
from neuros.plugins import load_plugin  # noqa: E402,F401
from neuros.runtime import OverflowPolicy, RuntimeState  # noqa: E402,F401
from neuros.drivers.base_driver import BaseDriver  # noqa: E402,F401
from neuros.drivers.mock_driver import MockDriver  # noqa: E402,F401
from neuros.models.base_model import BaseModel  # noqa: E402,F401
from neuros.models.simple_classifier import SimpleClassifier  # noqa: E402,F401
from neuros.processing.feature_extraction import BandPowerExtractor  # noqa: E402,F401
from neuros.processing.filters import BandpassFilter, SmoothingFilter  # noqa: E402,F401

__all__ = [
    "AlignedWindow",
    "AlignmentPlan",
    "BandPowerExtractor",
    "BandpassFilter",
    "BaseDriver",
    "BaseModel",
    "DataWindow",
    "Dataset",
    "DecoderOutput",
    "MockDriver",
    "MultiModalPipeline",
    "NativeRuntimeUnavailable",
    "OverflowPolicy",
    "Pipeline",
    "RuntimeState",
    "SignalFrame",
    "SimpleClassifier",
    "SmoothingFilter",
    "StreamDescriptor",
    "load_plugin",
    "native_runtime_available",
]
