"""Public neurOS SDK namespace.

The implementation is split across installable distributions that share the
``neuros`` namespace. This module keeps the familiar top-level API while the
kernel remains independently installable.

Driver and model exports are resolved lazily so dependency-light subpackages
such as ``neuros.evidence`` can be imported without eagerly importing every SDK
distribution. Normal ``neuros`` installations still provide these dependencies;
the lazy boundary prevents unrelated evidence imports from coupling to them.
"""

from importlib import import_module
from pkgutil import extend_path
from typing import Any

__path__ = extend_path(__path__, __name__)

from neuros.authority import (  # noqa: E402,F401
    alignment_authority_provenance,
    runtime_dataset_binding,
    to_research_dataset_authority,
)
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
from neuros.processing.feature_extraction import BandPowerExtractor  # noqa: E402,F401
from neuros.processing.filters import BandpassFilter, SmoothingFilter  # noqa: E402,F401
from neuros.runtime import OverflowPolicy, RuntimeState  # noqa: E402,F401

_LAZY_EXPORTS = {
    "BaseDriver": ("neuros.drivers.base_driver", "BaseDriver"),
    "MockDriver": ("neuros.drivers.mock_driver", "MockDriver"),
    "BaseModel": ("neuros.models.base_model", "BaseModel"),
    "SimpleClassifier": ("neuros.models.simple_classifier", "SimpleClassifier"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    try:
        value = getattr(import_module(module_name), attribute_name)
    except ModuleNotFoundError as exc:
        missing_name = exc.name or ""
        target_is_missing = missing_name == module_name or module_name.startswith(
            f"{missing_name}."
        )
        if not target_is_missing:
            raise
        raise ImportError(
            f"{name} requires the corresponding neurOS driver/model distribution; "
            "install the neuros SDK with its declared dependencies"
        ) from exc
    globals()[name] = value
    return value


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
    "alignment_authority_provenance",
    "load_plugin",
    "native_runtime_available",
    "runtime_dataset_binding",
    "to_research_dataset_authority",
]
