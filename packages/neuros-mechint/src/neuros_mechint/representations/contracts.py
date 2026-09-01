"""Immutable representation-benchmark contracts."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import numpy as np


class FitRegime(str, Enum):
    TRAIN_ONLY_INDUCTIVE = "train_only_inductive"
    TRANSDUCTIVE_TARGET_OBSERVED = "transductive_target_observed"
    EXTERNAL_PRETRAINED = "external_pretrained"


class MethodStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class RepresentationError(RuntimeError):
    """Base error for representation methods."""


class RepresentationUnavailableError(RepresentationError):
    """Optional external representation capability is unavailable."""


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metadata keys must be nonblank strings")
            frozen[key] = _freeze_value(item)
        return MappingProxyType(frozen)
    if isinstance(value, tuple):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, list):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze_value(item) for item in value)
    if isinstance(value, np.ndarray):
        array = np.array(value, copy=True, subok=False)
        array.setflags(write=False)
        return array
    return value


def _freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metadata keys must be nonblank strings")
        frozen[key] = _freeze_value(value)
    return MappingProxyType(frozen)


def _validated_array(value: Any, *, name: str, min_rows: int) -> np.ndarray:
    array = np.array(value, copy=True, subok=False)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2-D [time, features] array")
    if array.shape[0] < min_rows:
        raise ValueError(f"{name} must contain at least {min_rows} timepoints")
    if array.shape[1] < 1:
        raise ValueError(f"{name} must contain at least one feature")
    if array.dtype.kind not in "iuf":
        raise TypeError(f"{name} must contain real numeric, non-boolean values")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class SequenceBatch:
    """Independent, ordered trajectories with a common feature geometry."""

    sequences: tuple[np.ndarray, ...]
    sequence_ids: tuple[str, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        sequences = tuple(self.sequences)
        sequence_ids = tuple(self.sequence_ids)
        if not sequences:
            raise ValueError("SequenceBatch requires at least one sequence")
        if len(sequences) != len(sequence_ids):
            raise ValueError("sequence_ids must match the number of sequences")
        normalized_ids: list[str] = []
        for sequence_id in sequence_ids:
            if not isinstance(sequence_id, str) or not sequence_id.strip():
                raise ValueError("sequence IDs must be explicit nonblank strings")
            normalized_ids.append(sequence_id)
        if len(set(normalized_ids)) != len(normalized_ids):
            raise ValueError("sequence IDs must be unique")
        validated = tuple(
            _validated_array(sequence, name=f"sequence {sequence_id!r}", min_rows=3)
            for sequence_id, sequence in zip(normalized_ids, sequences, strict=True)
        )
        if len({array.shape[1] for array in validated}) != 1:
            raise ValueError("all sequences must share the same feature dimension")
        object.__setattr__(self, "sequences", validated)
        object.__setattr__(self, "sequence_ids", tuple(normalized_ids))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def feature_count(self) -> int:
        return int(self.sequences[0].shape[1])

    @property
    def sample_count(self) -> int:
        return int(sum(sequence.shape[0] for sequence in self.sequences))

    def concatenate(self) -> np.ndarray:
        """Return detached samples for order-independent train fitting only."""
        return np.concatenate([np.asarray(sequence) for sequence in self.sequences], axis=0)


@dataclass(frozen=True, slots=True)
class RepresentationEmbedding:
    method_id: str
    sequences: tuple[np.ndarray, ...]
    sequence_ids: tuple[str, ...]
    fit_regime: FitRegime
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        sequences = tuple(self.sequences)
        ids = tuple(self.sequence_ids)
        if not sequences or len(sequences) != len(ids):
            raise ValueError("embedding sequences and IDs must be nonempty and aligned")
        if len(set(ids)) != len(ids):
            raise ValueError("embedding sequence IDs must be unique")
        validated: list[np.ndarray] = []
        latent_dims: set[int] = set()
        for sequence_id, sequence in zip(ids, sequences, strict=True):
            if not isinstance(sequence_id, str) or not sequence_id.strip():
                raise ValueError("embedding sequence IDs must be nonblank strings")
            array = _validated_array(sequence, name=f"embedding {sequence_id!r}", min_rows=3)
            validated.append(array)
            latent_dims.add(int(array.shape[1]))
        if len(latent_dims) != 1:
            raise ValueError("all embedding sequences must share one latent dimension")
        object.__setattr__(self, "sequences", tuple(validated))
        object.__setattr__(self, "sequence_ids", ids)
        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def n_components(self) -> int:
        return int(self.sequences[0].shape[1])


@dataclass(frozen=True, slots=True)
class MethodOutcome:
    method_id: str
    fit_regime: FitRegime
    status: MethodStatus
    embedding: RepresentationEmbedding | None = None
    metrics: Mapping[str, float | None] | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        regime = FitRegime(self.fit_regime)
        status = MethodStatus(self.status)
        if status is MethodStatus.OK:
            if self.embedding is None:
                raise ValueError("successful outcomes require an embedding")
            if self.embedding.method_id != self.method_id:
                raise ValueError("outcome and embedding method IDs must match")
            if self.embedding.fit_regime is not regime:
                raise ValueError("outcome and embedding fit regimes must match")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful outcomes cannot carry an error")
        else:
            if self.embedding is not None:
                raise ValueError("failed/unavailable outcomes cannot carry an embedding")
            if not self.error_type or not self.error_message:
                raise ValueError("failed/unavailable outcomes require explicit error evidence")
        metric_values: dict[str, float | None] = {}
        if self.metrics is not None:
            if not isinstance(self.metrics, Mapping):
                raise TypeError("metrics must be a mapping")
            for key, value in self.metrics.items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("metric IDs must be nonblank strings")
                if value is None:
                    metric_values[key] = None
                else:
                    numeric = float(value)
                    if not np.isfinite(numeric):
                        raise ValueError("metric values must be finite or None")
                    metric_values[key] = numeric
        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class RepresentationBenchmarkResult:
    """Complete method result set with no scalar winner/ranking field."""

    train_sequence_ids: tuple[str, ...]
    evaluation_sequence_ids: tuple[str, ...]
    outcomes: tuple[MethodOutcome, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        outcomes = tuple(self.outcomes)
        if not outcomes:
            raise ValueError("benchmark result requires at least one method outcome")
        method_ids = [outcome.method_id for outcome in outcomes]
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("benchmark outcomes must have unique method IDs")
        object.__setattr__(self, "train_sequence_ids", tuple(self.train_sequence_ids))
        object.__setattr__(self, "evaluation_sequence_ids", tuple(self.evaluation_sequence_ids))
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def by_method(self) -> dict[str, MethodOutcome]:
        return {outcome.method_id: outcome for outcome in self.outcomes}


@runtime_checkable
class RepresentationMethod(Protocol):
    method_id: str
    fit_regime: FitRegime

    def embed(self, train: SequenceBatch, evaluation: SequenceBatch) -> RepresentationEmbedding:
        ...
