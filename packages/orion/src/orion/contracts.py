"""Stable interfaces connecting neurOS streams to ORION intelligence."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neuros.contracts import DecoderOutput, SignalFrame


@dataclass(frozen=True, slots=True)
class TokenizerManifest:
    tokenizer_id: str
    version: str
    schema_version: str = "1"
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tokenizer_id or not self.version:
            raise ValueError("tokenizer_id and version must be non-empty")
        object.__setattr__(self, "parameters", MappingProxyType(dict(self.parameters)))


@dataclass(frozen=True, slots=True)
class NeuroTokenBatch:
    """Machine-native token representation of a neural time interval."""

    token_ids: NDArray[np.integer]
    timestamps_ns: NDArray[np.integer]
    mask: NDArray[np.bool_] | None = None
    side_features: Mapping[str, NDArray[np.generic]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        token_ids = np.asarray(self.token_ids)
        timestamps = np.asarray(self.timestamps_ns)
        if token_ids.ndim != 1 or timestamps.ndim != 1:
            raise ValueError("token_ids and timestamps_ns must be 1-D")
        if len(token_ids) != len(timestamps):
            raise ValueError("token_ids and timestamps_ns must align")
        if self.mask is not None and len(np.asarray(self.mask)) != len(token_ids):
            raise ValueError("mask length must match token_ids")
        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "timestamps_ns", timestamps)
        if self.mask is not None:
            object.__setattr__(self, "mask", np.asarray(self.mask, dtype=bool))
        object.__setattr__(
            self,
            "side_features",
            MappingProxyType({key: np.asarray(value) for key, value in self.side_features.items()}),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RepresentationBatch:
    """Continuous neural representation aligned to time."""

    values: NDArray[np.floating]
    timestamps_ns: NDArray[np.integer]
    mask: NDArray[np.bool_] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        timestamps = np.asarray(self.timestamps_ns)
        if values.ndim < 2:
            raise ValueError("values must be at least [time, features]")
        if values.shape[0] != len(timestamps):
            raise ValueError("representation time axis must match timestamps")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class AdaptationProposal:
    """Auditable request to change an online decoder or representation."""

    reason: str
    changes: Mapping[str, Any]
    evidence: Mapping[str, float] = field(default_factory=dict)
    requires_approval: bool = False

    def __post_init__(self) -> None:
        if not self.reason:
            raise ValueError("reason must be non-empty")
        object.__setattr__(self, "changes", MappingProxyType(dict(self.changes)))
        object.__setattr__(self, "evidence", MappingProxyType(dict(self.evidence)))


@runtime_checkable
class NeuroTokenizer(Protocol):
    @property
    def manifest(self) -> TokenizerManifest:
        ...

    def encode(self, frames: list[SignalFrame]) -> NeuroTokenBatch:
        ...


@runtime_checkable
class NeuralEncoder(Protocol):
    def encode(self, tokens: NeuroTokenBatch) -> RepresentationBatch:
        ...


@runtime_checkable
class AdaptiveDecoder(Protocol):
    def infer(self, representation: RepresentationBatch) -> DecoderOutput:
        ...

    def propose_adaptation(
        self,
        representation: RepresentationBatch,
        output: DecoderOutput,
    ) -> AdaptationProposal | None:
        ...
