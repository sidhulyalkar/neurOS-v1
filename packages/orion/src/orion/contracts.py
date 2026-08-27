"""Stable interfaces connecting neurOS streams to ORION intelligence.

The contracts in this module are deliberately stricter than ordinary container
classes. They sit on scientific/evidence boundaries, so a frozen dataclass must
not still expose mutable NumPy buffers and time-aligned arrays must not rely on
implicit dtype coercion or undocumented ordering assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from neuros.contracts import DecoderOutput, SignalFrame


def _readonly_copy(values: Any) -> np.ndarray:
    array = np.asarray(values).copy()
    array.setflags(write=False)
    return array


def _timestamps(values: Any, *, expected_length: int) -> np.ndarray:
    timestamps = np.asarray(values)
    if timestamps.ndim != 1:
        raise ValueError("timestamps_ns must be 1-D")
    if timestamps.dtype == np.bool_ or not np.issubdtype(timestamps.dtype, np.integer):
        raise ValueError("timestamps_ns must contain integer nanosecond timestamps")
    if len(timestamps) != expected_length:
        raise ValueError("timestamps_ns must align with the time axis")
    if len(timestamps) > 1 and np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError("timestamps_ns must be nondecreasing")
    return _readonly_copy(timestamps)


def _mask(values: Any, *, expected_length: int) -> np.ndarray:
    mask = np.asarray(values)
    if mask.ndim != 1:
        raise ValueError("mask must be 1-D")
    if mask.dtype != np.bool_:
        raise ValueError("mask must contain boolean values without lossy coercion")
    if len(mask) != expected_length:
        raise ValueError("mask length must match the time axis")
    return _readonly_copy(mask)


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
    """Machine-native token representation of a neural time interval.

    Token IDs and timestamps are immutable after construction. Nanosecond
    timestamps are integer and nondecreasing; equal timestamps are permitted so
    simultaneous events can remain simultaneous rather than receiving invented
    ordering jitter. Every side feature is aligned on the leading token axis.
    """

    token_ids: NDArray[np.integer]
    timestamps_ns: NDArray[np.integer]
    mask: NDArray[np.bool_] | None = None
    side_features: Mapping[str, NDArray[np.generic]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        token_ids = np.asarray(self.token_ids)
        if token_ids.ndim != 1:
            raise ValueError("token_ids must be 1-D")
        if token_ids.dtype == np.bool_ or not np.issubdtype(token_ids.dtype, np.integer):
            raise ValueError("token_ids must contain integer token identities")
        token_ids = _readonly_copy(token_ids)
        timestamps = _timestamps(self.timestamps_ns, expected_length=len(token_ids))

        mask = None if self.mask is None else _mask(self.mask, expected_length=len(token_ids))

        side_features: dict[str, np.ndarray] = {}
        for key, value in self.side_features.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("side feature names must be non-empty strings")
            array = np.asarray(value)
            if array.ndim < 1:
                raise ValueError(f"side feature {key!r} must have a token-aligned leading axis")
            if array.shape[0] != len(token_ids):
                raise ValueError(
                    f"side feature {key!r} leading axis must match token_ids; "
                    f"expected {len(token_ids)}, got {array.shape[0]}"
                )
            side_features[key] = _readonly_copy(array)

        object.__setattr__(self, "token_ids", token_ids)
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "side_features", MappingProxyType(side_features))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RepresentationBatch:
    """Continuous neural representation aligned to time.

    ``values`` must be a finite floating representation whose leading axis is
    time. Padding/missingness belongs in the explicit boolean ``mask`` rather
    than being encoded as NaN/Inf. Arrays are copied and marked read-only so
    evidence fingerprints cannot be invalidated through aliases retained by a
    caller after construction.
    """

    values: NDArray[np.floating]
    timestamps_ns: NDArray[np.integer]
    mask: NDArray[np.bool_] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim < 2:
            raise ValueError("values must be at least [time, features]")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("representation values must use a real floating dtype")
        if not np.all(np.isfinite(values)):
            raise ValueError(
                "representation values must be finite; encode padding/missingness with mask"
            )
        values = _readonly_copy(values)
        timestamps = _timestamps(self.timestamps_ns, expected_length=values.shape[0])
        mask = None if self.mask is None else _mask(self.mask, expected_length=values.shape[0])

        object.__setattr__(self, "values", values)
        object.__setattr__(self, "timestamps_ns", timestamps)
        object.__setattr__(self, "mask", mask)
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
