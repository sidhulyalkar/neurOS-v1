"""Model and decoder contracts for neurOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from .signal import _freeze_metadata_mapping


_SUPPORTED_OUTPUT_ARRAY_KINDS = frozenset("biufc")


def _freeze_output_array(value: Any, *, field_name: str) -> NDArray[np.generic]:
    """Detach one canonical numeric output array from caller-owned storage."""

    array = np.array(value, copy=True, subok=False)
    if array.dtype.kind not in _SUPPORTED_OUTPUT_ARRAY_KINDS:
        raise TypeError(
            f"{field_name} must use a boolean or numeric dtype; received {array.dtype}"
        )
    array.setflags(write=False)
    return array


def _freeze_prediction(value: Any, *, path: str = "prediction") -> Any:
    """Freeze deterministic prediction values without retaining mutable aliases.

    ``prediction`` remains intentionally more flexible than the typed numeric
    score arrays because class labels may be strings. Numeric/bool ndarrays stay
    arrays; string/object arrays are canonicalized through nested immutable
    sequences when every contained value belongs to the deterministic prediction
    language supported by the shared-memory codec.
    """

    if isinstance(value, np.generic):
        return _freeze_prediction(value.item(), path=path)
    if isinstance(value, np.ndarray):
        if value.dtype.kind in _SUPPORTED_OUTPUT_ARRAY_KINDS:
            return _freeze_output_array(value, field_name=path)
        if value.dtype.kind in "USO":
            return _freeze_prediction(value.tolist(), path=path)
        raise TypeError(f"{path} uses unsupported ndarray dtype {value.dtype}")
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = _freeze_prediction(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_prediction(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{path} cannot contain unordered set values")
    raise TypeError(
        f"{path} contains unsupported value type {type(value).__module__}."
        f"{type(value).__qualname__}; use deterministic prediction primitives"
    )


def _optional_output_array(value: Any, *, field_name: str) -> NDArray[np.generic] | None:
    if value is None:
        return None
    return _freeze_output_array(value, field_name=field_name)


@dataclass(frozen=True, slots=True)
class DecoderCapabilities:
    """Capabilities a decoder can expose to the runtime."""

    probabilities: bool = False
    uncertainty: bool = False
    online_fit: bool = False
    streaming_state: bool = False
    embeddings: bool = False


@dataclass(frozen=True, slots=True)
class DecoderOutput:
    """Structured decoder output with detached immutable result/provenance state.

    Array-bearing fields are copied at construction and marked read-only, so a
    caller cannot mutate a canonical output through a retained input buffer.
    Deterministic prediction containers and metadata are recursively frozen for
    the same reason. This is representation immutability; it does not invent
    confidence, calibration, or scientific validity that a decoder did not
    provide.
    """

    prediction: Any
    confidence: float | None = None
    uncertainty: float | None = None
    probabilities: NDArray[np.floating] | None = None
    logits: NDArray[np.floating] | None = None
    embedding: NDArray[np.floating] | None = None
    model_id: str | None = None
    model_version: str | None = None
    inference_time_ns: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if self.uncertainty is not None and self.uncertainty < 0:
            raise ValueError("uncertainty must be non-negative")

        object.__setattr__(self, "prediction", _freeze_prediction(self.prediction))
        object.__setattr__(
            self,
            "probabilities",
            _optional_output_array(self.probabilities, field_name="probabilities"),
        )
        object.__setattr__(
            self,
            "logits",
            _optional_output_array(self.logits, field_name="logits"),
        )
        object.__setattr__(
            self,
            "embedding",
            _optional_output_array(self.embedding, field_name="embedding"),
        )
        object.__setattr__(self, "metadata", _freeze_metadata_mapping(self.metadata))


@runtime_checkable
class Decoder(Protocol):
    @property
    def capabilities(self) -> DecoderCapabilities:
        ...

    def infer(self, X: NDArray[np.generic]) -> DecoderOutput:
        ...


@runtime_checkable
class TrainableDecoder(Decoder, Protocol):
    @property
    def is_trained(self) -> bool:
        ...

    def train(self, X: NDArray[np.generic], y: NDArray[np.generic]) -> None:
        ...
