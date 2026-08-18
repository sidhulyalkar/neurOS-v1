"""Model and decoder contracts for neurOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


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
    """Structured decoder output without fabricated certainty."""

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
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


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
