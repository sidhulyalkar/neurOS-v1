"""Sequence-level failure-preserving representation benchmark authority."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType
from typing import Any

import numpy as np

from .contracts import (
    FitRegime,
    MethodStatus,
    RepresentationEmbedding,
    RepresentationMethod,
    RepresentationUnavailableError,
    SequenceBatch,
    _freeze_metadata,
)
from .metrics import aggregate_geometry_metrics, aggregate_reference_metrics
from .pca import _positive_int


@dataclass(frozen=True, slots=True)
class SequenceMethodOutcome:
    """One declared method × evaluation-sequence outcome."""

    method_id: str
    sequence_id: str
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
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a nonblank string")
        regime = FitRegime(self.fit_regime)
        status = MethodStatus(self.status)
        metrics: dict[str, float | None] = {}
        for key, value in dict(self.metrics or {}).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metric IDs must be nonblank strings")
            if value is None:
                metrics[key] = None
            else:
                numeric = float(value)
                if not np.isfinite(numeric):
                    raise ValueError("metric values must be finite or None")
                metrics[key] = numeric

        if status is MethodStatus.OK:
            if self.embedding is None:
                raise ValueError("successful sequence outcomes require an embedding")
            if self.embedding.method_id != self.method_id:
                raise ValueError("outcome and embedding method IDs must match")
            if self.embedding.fit_regime is not regime:
                raise ValueError("outcome and embedding fit regimes must match")
            if self.embedding.sequence_ids != (self.sequence_id,):
                raise ValueError("sequence outcome embedding must contain exactly its sequence")
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful sequence outcomes cannot carry an error")
        else:
            if self.embedding is not None:
                raise ValueError("failed/unavailable sequence outcomes cannot carry an embedding")
            if metrics:
                raise ValueError(
                    "failed/unavailable sequence outcomes cannot carry scientific metrics"
                )
            if not self.error_type or not self.error_message:
                raise ValueError(
                    "failed/unavailable sequence outcomes require explicit error evidence"
                )

        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SequenceRepresentationBenchmarkResult:
    """Cartesian-complete method × sequence evidence with no ranking field."""

    method_ids: tuple[str, ...]
    train_sequence_ids: tuple[str, ...]
    evaluation_sequence_ids: tuple[str, ...]
    outcomes: tuple[SequenceMethodOutcome, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        method_ids = tuple(self.method_ids)
        train_ids = tuple(self.train_sequence_ids)
        evaluation_ids = tuple(self.evaluation_sequence_ids)
        outcomes = tuple(self.outcomes)
        if not method_ids or not evaluation_ids or not outcomes:
            raise ValueError("sequence benchmark requires methods, sequences, and outcomes")
        if any(not isinstance(value, str) or not value.strip() for value in method_ids):
            raise ValueError("method IDs must be nonblank strings")
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("method IDs must be unique")
        if len(set(evaluation_ids)) != len(evaluation_ids):
            raise ValueError("evaluation sequence IDs must be unique")

        actual = [(row.method_id, row.sequence_id) for row in outcomes]
        if len(set(actual)) != len(actual):
            raise ValueError("method × sequence outcome identities must be unique")
        expected = {
            (method_id, sequence_id)
            for method_id in method_ids
            for sequence_id in evaluation_ids
        }
        if set(actual) != expected:
            missing = sorted(expected - set(actual))
            extra = sorted(set(actual) - expected)
            raise ValueError(
                "method × sequence evidence must be Cartesian complete: "
                f"missing={missing}, extra={extra}"
            )

        regimes: dict[str, FitRegime] = {}
        for outcome in outcomes:
            prior = regimes.setdefault(outcome.method_id, outcome.fit_regime)
            if prior is not outcome.fit_regime:
                raise ValueError("a method cannot change fit regime across sequences")

        object.__setattr__(self, "method_ids", method_ids)
        object.__setattr__(self, "train_sequence_ids", train_ids)
        object.__setattr__(self, "evaluation_sequence_ids", evaluation_ids)
        object.__setattr__(self, "outcomes", outcomes)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def by_pair(self) -> dict[tuple[str, str], SequenceMethodOutcome]:
        return {(row.method_id, row.sequence_id): row for row in self.outcomes}


def _single_sequence(batch: SequenceBatch, index: int) -> SequenceBatch:
    return SequenceBatch(
        sequences=(batch.sequences[index],),
        sequence_ids=(batch.sequence_ids[index],),
        metadata=dict(batch.metadata),
    )


def _score_sequence(
    source: np.ndarray,
    latent: np.ndarray,
    reference: np.ndarray | None,
    *,
    neighborhood_k: int,
) -> dict[str, float | None]:
    metrics = aggregate_geometry_metrics((source,), (latent,), k=neighborhood_k)
    if reference is not None:
        metrics.update(
            aggregate_reference_metrics((reference,), (latent,), k=neighborhood_k)
        )
    return metrics


def _single_embedding(
    embedding: RepresentationEmbedding,
    index: int,
) -> RepresentationEmbedding:
    return RepresentationEmbedding(
        method_id=embedding.method_id,
        sequences=(embedding.sequences[index],),
        sequence_ids=(embedding.sequence_ids[index],),
        fit_regime=embedding.fit_regime,
        metadata=dict(embedding.metadata),
    )


def _failure_outcomes(
    method: RepresentationMethod,
    evaluation: SequenceBatch,
    *,
    status: MethodStatus,
    error: Exception,
    runtime_seconds: float,
) -> list[SequenceMethodOutcome]:
    return [
        SequenceMethodOutcome(
            method_id=method.method_id,
            sequence_id=sequence_id,
            fit_regime=method.fit_regime,
            status=status,
            error_type=type(error).__name__,
            error_message=str(error),
            metadata={
                "execution_scope": "shared_method_batch",
                "runtime_attribution": "shared_not_sequence_additive",
                "shared_batch_runtime_seconds": runtime_seconds,
            },
        )
        for sequence_id in evaluation.sequence_ids
    ]


def run_sequencewise_representation_benchmark(
    methods: Iterable[RepresentationMethod],
    train: SequenceBatch,
    evaluation: SequenceBatch,
    *,
    reference: SequenceBatch | None = None,
    neighborhood_k: int = 5,
) -> SequenceRepresentationBenchmarkResult:
    """Run methods with native per-sequence isolation when explicitly supported.

    Methods exposing ``embed_sequence(train, one_sequence_batch)`` are executed
    independently per evaluation trajectory, so one plugin failure cannot erase a
    successful sibling. Batch-only methods are executed once and split only after a
    complete successful embedding; a batch failure is conservatively recorded for
    every declared sequence.
    """
    methods = tuple(methods)
    if not methods:
        raise ValueError("at least one representation method is required")
    method_ids = tuple(method.method_id for method in methods)
    if any(not isinstance(method_id, str) or not method_id.strip() for method_id in method_ids):
        raise ValueError("every method must expose a nonblank method_id")
    if len(set(method_ids)) != len(method_ids):
        raise ValueError("representation method IDs must be unique")
    neighborhood_k = _positive_int(neighborhood_k, name="neighborhood_k")
    if train.feature_count != evaluation.feature_count:
        raise ValueError("train and evaluation feature dimensions must match")
    if reference is not None:
        if reference.sequence_ids != evaluation.sequence_ids:
            raise ValueError("reference sequence identity must exactly match evaluation identity")
        for source, reference_sequence in zip(
            evaluation.sequences,
            reference.sequences,
            strict=True,
        ):
            if source.shape[0] != reference_sequence.shape[0]:
                raise ValueError(
                    "reference and evaluation sequences must have matching timepoints"
                )

    outcomes: list[SequenceMethodOutcome] = []
    for method in methods:
        embed_sequence = getattr(method, "embed_sequence", None)
        if callable(embed_sequence):
            for index, sequence_id in enumerate(evaluation.sequence_ids):
                evaluation_one = _single_sequence(evaluation, index)
                reference_one = None if reference is None else _single_sequence(reference, index)
                started = perf_counter()
                try:
                    embedding = embed_sequence(train, evaluation_one)
                    runtime_seconds = max(0.0, perf_counter() - started)
                    if embedding.sequence_ids != (sequence_id,):
                        raise ValueError(
                            "native sequence embedding changed evaluation sequence identity"
                        )
                    source = evaluation_one.sequences[0]
                    latent = embedding.sequences[0]
                    if source.shape[0] != latent.shape[0]:
                        raise ValueError(
                            "representation output changed the evaluation timepoint count"
                        )
                    reference_array = (
                        None if reference_one is None else reference_one.sequences[0]
                    )
                    outcomes.append(
                        SequenceMethodOutcome(
                            method_id=method.method_id,
                            sequence_id=sequence_id,
                            fit_regime=method.fit_regime,
                            status=MethodStatus.OK,
                            embedding=embedding,
                            metrics=_score_sequence(
                                source,
                                latent,
                                reference_array,
                                neighborhood_k=neighborhood_k,
                            ),
                            metadata={
                                "execution_scope": "native_per_sequence",
                                "runtime_seconds": runtime_seconds,
                                "runtime_domain": "operational",
                            },
                        )
                    )
                except RepresentationUnavailableError as exc:
                    outcomes.append(
                        SequenceMethodOutcome(
                            method_id=method.method_id,
                            sequence_id=sequence_id,
                            fit_regime=method.fit_regime,
                            status=MethodStatus.UNAVAILABLE,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            metadata={
                                "execution_scope": "native_per_sequence",
                                "runtime_seconds": max(0.0, perf_counter() - started),
                                "runtime_domain": "operational",
                            },
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    outcomes.append(
                        SequenceMethodOutcome(
                            method_id=method.method_id,
                            sequence_id=sequence_id,
                            fit_regime=method.fit_regime,
                            status=MethodStatus.FAILED,
                            error_type=type(exc).__name__,
                            error_message=str(exc),
                            metadata={
                                "execution_scope": "native_per_sequence",
                                "runtime_seconds": max(0.0, perf_counter() - started),
                                "runtime_domain": "operational",
                            },
                        )
                    )
            continue

        started = perf_counter()
        try:
            embedding = method.embed(train, evaluation)
            runtime_seconds = max(0.0, perf_counter() - started)
            if embedding.sequence_ids != evaluation.sequence_ids:
                raise ValueError("representation output changed evaluation sequence identity")
            for index, sequence_id in enumerate(evaluation.sequence_ids):
                source = evaluation.sequences[index]
                latent = embedding.sequences[index]
                if source.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "representation output changed the evaluation timepoint count"
                    )
                reference_array = None if reference is None else reference.sequences[index]
                outcomes.append(
                    SequenceMethodOutcome(
                        method_id=method.method_id,
                        sequence_id=sequence_id,
                        fit_regime=method.fit_regime,
                        status=MethodStatus.OK,
                        embedding=_single_embedding(embedding, index),
                        metrics=_score_sequence(
                            source,
                            latent,
                            reference_array,
                            neighborhood_k=neighborhood_k,
                        ),
                        metadata={
                            "execution_scope": "shared_method_batch",
                            "runtime_attribution": "shared_not_sequence_additive",
                            "shared_batch_runtime_seconds": runtime_seconds,
                            "runtime_domain": "operational",
                        },
                    )
                )
        except RepresentationUnavailableError as exc:
            outcomes.extend(
                _failure_outcomes(
                    method,
                    evaluation,
                    status=MethodStatus.UNAVAILABLE,
                    error=exc,
                    runtime_seconds=max(0.0, perf_counter() - started),
                )
            )
        except Exception as exc:  # noqa: BLE001
            outcomes.extend(
                _failure_outcomes(
                    method,
                    evaluation,
                    status=MethodStatus.FAILED,
                    error=exc,
                    runtime_seconds=max(0.0, perf_counter() - started),
                )
            )

    return SequenceRepresentationBenchmarkResult(
        method_ids=method_ids,
        train_sequence_ids=train.sequence_ids,
        evaluation_sequence_ids=evaluation.sequence_ids,
        outcomes=tuple(outcomes),
        metadata={
            "ranking_policy": "none",
            "claim_scope": "representation_geometry_sequence_level",
            "sequence_authority": "method_x_evaluation_sequence_cartesian_complete",
            "native_sequence_capability_required_for_partial_failure_preservation": True,
            "reference_geometry": "provided" if reference is not None else "none",
        },
    )
