"""Case-level, failure-preserving representation benchmark authority."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np

from .contracts import (
    EvaluationScope,
    FitRegime,
    RepresentationMethod,
    RepresentationUnavailableError,
    SequenceBatch,
    _freeze_metadata,
    _strict_metric_value,
    _validated_array,
)
from .metrics import aggregate_geometry_metrics, aggregate_reference_metrics
from .pca import _positive_int


class CaseStatus(str, Enum):
    """Status for one declared method × sequence benchmark case."""

    OK = "ok"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    NONCONVERGED = "nonconverged"


class RepresentationNonconvergenceError(RuntimeError):
    """A representation method explicitly reported non-convergence."""


@dataclass(frozen=True, slots=True)
class RepresentationCaseOutcome:
    """Evidence for one method on one preserved evaluation sequence."""

    method_id: str
    sequence_id: str
    fit_regime: FitRegime
    evaluation_scope: EvaluationScope
    status: CaseStatus
    embedding: np.ndarray | None = None
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
        evaluation_scope = EvaluationScope(self.evaluation_scope)
        status = CaseStatus(self.status)

        if status is CaseStatus.OK:
            if self.embedding is None:
                raise ValueError("successful cases require an embedding")
            embedding = _validated_array(
                self.embedding,
                name=f"case embedding {self.method_id!r}/{self.sequence_id!r}",
                min_rows=3,
            )
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful cases cannot carry error evidence")
        else:
            if self.embedding is not None:
                raise ValueError("non-success cases cannot carry an embedding")
            embedding = None
            if not self.error_type or not self.error_message:
                raise ValueError("non-success cases require explicit error evidence")
            if self.metrics:
                raise ValueError("non-success cases cannot carry scientific metric values")

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
                    metric_values[key] = _strict_metric_value(
                        value,
                        name=f"metric {key!r}",
                    )

        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "evaluation_scope", evaluation_scope)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class MethodCaseSummary:
    """Derived method summary that keeps its denominator and failures visible."""

    method_id: str
    fit_regime: FitRegime
    evaluation_scope: EvaluationScope
    total_cases: int
    ok_cases: int
    failed_cases: int
    unavailable_cases: int
    nonconverged_cases: int
    metrics: Mapping[str, float | None]
    metric_n: Mapping[str, int]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        counts = (
            self.total_cases,
            self.ok_cases,
            self.failed_cases,
            self.unavailable_cases,
            self.nonconverged_cases,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, (int, np.integer))
            for value in counts
        ):
            raise TypeError("summary counts must be integers")
        counts = tuple(int(value) for value in counts)
        if counts[0] <= 0:
            raise ValueError("total_cases must be positive")
        if any(value < 0 for value in counts[1:]):
            raise ValueError("case counts cannot be negative")
        if sum(counts[1:]) != counts[0]:
            raise ValueError("case counts must sum exactly to total_cases")
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")

        metric_values: dict[str, float | None] = {}
        for key, value in dict(self.metrics).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metric IDs must be nonblank strings")
            if value is None:
                metric_values[key] = None
            else:
                metric_values[key] = _strict_metric_value(
                    value,
                    name=f"summary metric {key!r}",
                )

        metric_n: dict[str, int] = {}
        for key, value in dict(self.metric_n).items():
            if not isinstance(key, str) or not key.strip():
                raise ValueError("metric_n IDs must be nonblank strings")
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise TypeError("metric_n values must be integers")
            numeric = int(value)
            if numeric < 0 or numeric > counts[1]:
                raise ValueError("metric_n values must be between zero and ok_cases")
            metric_n[key] = numeric
        if set(metric_n) != set(metric_values):
            raise ValueError("metric_n keys must exactly match summary metric keys")

        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
        object.__setattr__(
            self,
            "evaluation_scope",
            EvaluationScope(self.evaluation_scope),
        )
        object.__setattr__(self, "total_cases", counts[0])
        object.__setattr__(self, "ok_cases", counts[1])
        object.__setattr__(self, "failed_cases", counts[2])
        object.__setattr__(self, "unavailable_cases", counts[3])
        object.__setattr__(self, "nonconverged_cases", counts[4])
        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
        object.__setattr__(self, "metric_n", MappingProxyType(metric_n))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def non_ok_rate(self) -> float:
        return float((self.total_cases - self.ok_cases) / self.total_cases)

    @property
    def failed_rate(self) -> float:
        return float(self.failed_cases / self.total_cases)

    @property
    def unavailable_rate(self) -> float:
        return float(self.unavailable_cases / self.total_cases)

    @property
    def nonconverged_rate(self) -> float:
        return float(self.nonconverged_cases / self.total_cases)

    @property
    def failure_rate(self) -> float:
        """Deprecated compatibility alias for the broader non-ok rate."""
        return self.non_ok_rate


@dataclass(frozen=True, slots=True)
class CasePreservingRepresentationResult:
    """Complete Cartesian method × sequence result set with no hidden omissions."""

    train_sequence_ids: tuple[str, ...]
    evaluation_sequence_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    cases: tuple[RepresentationCaseOutcome, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        train_ids = tuple(self.train_sequence_ids)
        evaluation_ids = tuple(self.evaluation_sequence_ids)
        method_ids = tuple(self.method_ids)
        cases = tuple(self.cases)
        if not train_ids:
            raise ValueError("train_sequence_ids cannot be empty")
        if not evaluation_ids:
            raise ValueError("evaluation_sequence_ids cannot be empty")
        if not method_ids:
            raise ValueError("method_ids cannot be empty")
        if len(set(train_ids)) != len(train_ids):
            raise ValueError("train sequence IDs must be unique")
        if len(set(evaluation_ids)) != len(evaluation_ids):
            raise ValueError("evaluation sequence IDs must be unique")
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("method IDs must be unique")
        if any(not isinstance(value, str) or not value.strip() for value in train_ids):
            raise ValueError("train sequence IDs must be nonblank strings")
        if any(not isinstance(value, str) or not value.strip() for value in evaluation_ids):
            raise ValueError("evaluation sequence IDs must be nonblank strings")
        if any(not isinstance(value, str) or not value.strip() for value in method_ids):
            raise ValueError("method IDs must be nonblank strings")

        expected = {
            (method_id, sequence_id)
            for method_id in method_ids
            for sequence_id in evaluation_ids
        }
        seen: set[tuple[str, str]] = set()
        regimes: dict[str, FitRegime] = {}
        scopes: dict[str, EvaluationScope] = {}
        for case in cases:
            key = (case.method_id, case.sequence_id)
            if key in seen:
                raise ValueError(f"duplicate representation case {key!r}")
            seen.add(key)
            existing = regimes.setdefault(case.method_id, case.fit_regime)
            if existing is not case.fit_regime:
                raise ValueError("all cases for a method must share one fit regime")
            existing_scope = scopes.setdefault(case.method_id, case.evaluation_scope)
            if existing_scope is not case.evaluation_scope:
                raise ValueError("all cases for a method must share one evaluation scope")
        missing = expected - seen
        extra = seen - expected
        if missing or extra:
            raise ValueError(
                "case result must contain the exact declared method × sequence "
                "Cartesian product; "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )

        object.__setattr__(self, "train_sequence_ids", train_ids)
        object.__setattr__(self, "evaluation_sequence_ids", evaluation_ids)
        object.__setattr__(self, "method_ids", method_ids)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def by_case(self) -> dict[tuple[str, str], RepresentationCaseOutcome]:
        return {(case.method_id, case.sequence_id): case for case in self.cases}

    def cases_for_method(self, method_id: str) -> tuple[RepresentationCaseOutcome, ...]:
        if method_id not in self.method_ids:
            raise KeyError(method_id)
        by_case = self.by_case()
        return tuple(
            by_case[(method_id, sequence_id)]
            for sequence_id in self.evaluation_sequence_ids
        )

    def summary_for_method(self, method_id: str) -> MethodCaseSummary:
        cases = self.cases_for_method(method_id)
        regime = cases[0].fit_regime
        evaluation_scope = cases[0].evaluation_scope
        counts = {status: 0 for status in CaseStatus}
        metric_values: dict[str, list[float]] = {}
        metric_schema: tuple[str, ...] | None = None
        for case in cases:
            counts[case.status] += 1
            if case.status is not CaseStatus.OK:
                continue
            case_schema = tuple(sorted(case.metrics))
            if metric_schema is None:
                metric_schema = case_schema
            elif case_schema != metric_schema:
                raise ValueError(
                    "successful cases for one method must expose an identical metric schema"
                )
            for key, value in case.metrics.items():
                if value is not None:
                    metric_values.setdefault(key, []).append(value)
        metric_schema = metric_schema or ()
        aggregated = {
            key: float(np.mean(metric_values[key])) if metric_values.get(key) else None
            for key in metric_schema
        }
        metric_n = {key: len(metric_values.get(key, ())) for key in metric_schema}
        total = len(cases)
        ok = counts[CaseStatus.OK]
        return MethodCaseSummary(
            method_id=method_id,
            fit_regime=regime,
            evaluation_scope=evaluation_scope,
            total_cases=total,
            ok_cases=ok,
            failed_cases=counts[CaseStatus.FAILED],
            unavailable_cases=counts[CaseStatus.UNAVAILABLE],
            nonconverged_cases=counts[CaseStatus.NONCONVERGED],
            metrics=aggregated,
            metric_n=metric_n,
            metadata={
                "aggregation_basis": "successful_cases_with_per_metric_denominator",
                "successful_metric_cases": ok,
                "declared_total_cases": total,
            },
        )

    def summaries(self) -> tuple[MethodCaseSummary, ...]:
        return tuple(self.summary_for_method(method_id) for method_id in self.method_ids)


class CasePreservingRepresentationBenchmark:
    """Benchmark representations while preserving every method × sequence case."""

    def __init__(
        self,
        methods: Iterable[RepresentationMethod],
        *,
        neighborhood_k: int = 5,
    ) -> None:
        methods = tuple(methods)
        if not methods:
            raise ValueError("at least one representation method is required")
        ids = [method.method_id for method in methods]
        if any(
            not isinstance(method_id, str) or not method_id.strip()
            for method_id in ids
        ):
            raise ValueError("every method must expose a nonblank method_id")
        if len(set(ids)) != len(ids):
            raise ValueError("representation method IDs must be unique")
        for method in methods:
            try:
                EvaluationScope(method.evaluation_scope)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"representation method {method.method_id!r} must declare a valid "
                    "evaluation_scope"
                ) from exc
        self.methods = methods
        self.neighborhood_k = _positive_int(neighborhood_k, name="neighborhood_k")

    @staticmethod
    def _validate_reference(
        evaluation: SequenceBatch,
        reference: SequenceBatch | None,
    ) -> None:
        if reference is None:
            return
        if reference.sequence_ids != evaluation.sequence_ids:
            raise ValueError(
                "reference sequence identity must exactly match evaluation identity"
            )
        for source, reference_sequence in zip(
            evaluation.sequences,
            reference.sequences,
            strict=True,
        ):
            if source.shape[0] != reference_sequence.shape[0]:
                raise ValueError(
                    "reference and evaluation sequences must have matching timepoints"
                )

    def _metrics(
        self,
        source: np.ndarray,
        latent: np.ndarray,
        reference: np.ndarray | None,
    ) -> dict[str, float | None]:
        metrics = aggregate_geometry_metrics(
            (source,),
            (latent,),
            k=self.neighborhood_k,
        )
        if reference is not None:
            metrics.update(
                aggregate_reference_metrics(
                    (reference,),
                    (latent,),
                    k=self.neighborhood_k,
                )
            )
        return metrics

    @staticmethod
    def _single_batch(batch: SequenceBatch, index: int) -> SequenceBatch:
        return SequenceBatch(
            sequences=(batch.sequences[index],),
            sequence_ids=(batch.sequence_ids[index],),
            metadata=batch.metadata,
        )

    @staticmethod
    def _failure_case(
        method: RepresentationMethod,
        sequence_id: str,
        status: CaseStatus,
        exc: Exception,
    ) -> RepresentationCaseOutcome:
        return RepresentationCaseOutcome(
            method_id=method.method_id,
            sequence_id=sequence_id,
            fit_regime=method.fit_regime,
            evaluation_scope=method.evaluation_scope,
            status=status,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )

    def _batch_inductive_cases(
        self,
        method: RepresentationMethod,
        train: SequenceBatch,
        evaluation: SequenceBatch,
        reference: SequenceBatch | None,
    ) -> list[RepresentationCaseOutcome]:
        try:
            embedding = method.embed(train, evaluation)
            if embedding.sequence_ids != evaluation.sequence_ids:
                raise ValueError(
                    "representation output sequence identity does not match evaluation batch"
                )
            if len(embedding.sequences) != len(evaluation.sequences):
                raise ValueError("representation output changed evaluation sequence count")
            for source, latent in zip(evaluation.sequences, embedding.sequences, strict=True):
                if source.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "representation output changed the evaluation timepoint count"
                    )
        except RepresentationUnavailableError as exc:
            return [self._failure_case(method, sequence_id, CaseStatus.UNAVAILABLE, exc) for sequence_id in evaluation.sequence_ids]
        except RepresentationNonconvergenceError as exc:
            return [self._failure_case(method, sequence_id, CaseStatus.NONCONVERGED, exc) for sequence_id in evaluation.sequence_ids]
        except Exception as exc:
            return [self._failure_case(method, sequence_id, CaseStatus.FAILED, exc) for sequence_id in evaluation.sequence_ids]

        cases: list[RepresentationCaseOutcome] = []
        for index, (sequence_id, source, latent) in enumerate(
            zip(evaluation.sequence_ids, evaluation.sequences, embedding.sequences, strict=True)
        ):
            reference_sequence = None if reference is None else reference.sequences[index]
            metrics = self._metrics(source, latent, reference_sequence)
            cases.append(
                RepresentationCaseOutcome(
                    method_id=method.method_id,
                    sequence_id=sequence_id,
                    fit_regime=method.fit_regime,
                    evaluation_scope=method.evaluation_scope,
                    status=CaseStatus.OK,
                    embedding=latent,
                    metrics=metrics,
                    metadata={
                        "metric_scope": "trajectory_local_rigid_transform_invariant",
                        "embedding_metadata": dict(embedding.metadata),
                        "evaluation_scope": EvaluationScope(method.evaluation_scope).value,
                    },
                )
            )
        return cases

    def _sequence_local_cases(
        self,
        method: RepresentationMethod,
        train: SequenceBatch,
        evaluation: SequenceBatch,
        reference: SequenceBatch | None,
    ) -> list[RepresentationCaseOutcome]:
        cases: list[RepresentationCaseOutcome] = []
        for index, sequence_id in enumerate(evaluation.sequence_ids):
            evaluation_case = self._single_batch(evaluation, index)
            try:
                embedding = method.embed(train, evaluation_case)
                if embedding.sequence_ids != (sequence_id,) or len(embedding.sequences) != 1:
                    raise ValueError(
                        "sequence-local representation output identity does not match case"
                    )
                latent = embedding.sequences[0]
                source = evaluation.sequences[index]
                if source.shape[0] != latent.shape[0]:
                    raise ValueError(
                        "representation output changed the evaluation timepoint count"
                    )
            except RepresentationUnavailableError as exc:
                cases.append(self._failure_case(method, sequence_id, CaseStatus.UNAVAILABLE, exc))
                continue
            except RepresentationNonconvergenceError as exc:
                cases.append(self._failure_case(method, sequence_id, CaseStatus.NONCONVERGED, exc))
                continue
            except Exception as exc:
                cases.append(self._failure_case(method, sequence_id, CaseStatus.FAILED, exc))
                continue

            reference_sequence = None if reference is None else reference.sequences[index]
            metrics = self._metrics(source, latent, reference_sequence)
            cases.append(
                RepresentationCaseOutcome(
                    method_id=method.method_id,
                    sequence_id=sequence_id,
                    fit_regime=method.fit_regime,
                    evaluation_scope=method.evaluation_scope,
                    status=CaseStatus.OK,
                    embedding=latent,
                    metrics=metrics,
                    metadata={
                        "metric_scope": "trajectory_local_rigid_transform_invariant",
                        "embedding_metadata": dict(embedding.metadata),
                        "evaluation_scope": EvaluationScope(method.evaluation_scope).value,
                    },
                )
            )
        return cases

    def run(
        self,
        train: SequenceBatch,
        evaluation: SequenceBatch,
        *,
        reference: SequenceBatch | None = None,
    ) -> CasePreservingRepresentationResult:
        if train.feature_count != evaluation.feature_count:
            raise ValueError("train and evaluation feature dimensions must match")
        self._validate_reference(evaluation, reference)

        cases: list[RepresentationCaseOutcome] = []
        for method in self.methods:
            scope = EvaluationScope(method.evaluation_scope)
            if scope is EvaluationScope.BATCH_TRANSFORM:
                cases.extend(self._batch_inductive_cases(method, train, evaluation, reference))
            elif scope is EvaluationScope.SEQUENCE_LOCAL:
                cases.extend(self._sequence_local_cases(method, train, evaluation, reference))
            else:  # pragma: no cover
                raise ValueError(f"unsupported evaluation scope {scope!r}")

        return CasePreservingRepresentationResult(
            train_sequence_ids=train.sequence_ids,
            evaluation_sequence_ids=evaluation.sequence_ids,
            method_ids=tuple(method.method_id for method in self.methods),
            cases=tuple(cases),
            metadata={
                "neighborhood_k": self.neighborhood_k,
                "ranking_policy": "none",
                "claim_scope": "representation_geometry",
                "case_authority": (
                    "complete_method_x_sequence_cartesian_product"
                ),
                "evaluation_scope_authority": (
                    "explicit_method_declared_batch_or_sequence_local"
                ),
                "reference_geometry": (
                    "provided" if reference is not None else "none"
                ),
            },
        )
