"""Failure-preserving, content-addressed representation evidence contracts.

This module deliberately does not execute representation methods. It records the
scientific evidence emitted by an already-declared benchmark or study while
preserving missingness, non-convergence, evaluation scope, fit regime, and the
per-metric denominator used by derived summaries.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from math import isfinite
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any

from .contracts import FitRegime


_SCHEMA_VERSION = 1
_DIGEST_DOMAIN = b"neuros.representation-evidence-grid.v1\0"


class EvaluationScope(str, Enum):
    """How a method consumed the declared evaluation data."""

    BATCH_TRANSFORM = "batch_transform"
    SEQUENCE_LOCAL = "sequence_local"


class CaseStatus(str, Enum):
    """Outcome of one declared method × evaluation-sequence case."""

    OK = "ok"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    NONCONVERGED = "nonconverged"


def _identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _finite_metric(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    numeric = float(value)
    if not isfinite(numeric):
        raise ValueError(f"{name} must be a finite real number")
    # JSON distinguishes -0.0 and 0.0 even though they are numerically equal.
    return 0.0 if numeric == 0.0 else numeric


def _count(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer")
    return int(value)


def _freeze_json(value: Any, *, path: str) -> Any:
    """Deep-freeze the portable JSON subset used by evidence metadata."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} mapping keys must be strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{path} must not contain unordered sets")
    raise TypeError(f"{path} contains unsupported value type {type(value).__name__}")


def _freeze_metadata(metadata: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if metadata is None:
        return MappingProxyType({})
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    frozen = _freeze_json(metadata, path="metadata")
    assert isinstance(frozen, Mapping)
    return frozen


def _portable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _portable(value[key]) for key in sorted(value)}
    if isinstance(value, tuple):
        return [_portable(item) for item in value]
    return value


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        _portable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class RepresentationCaseEvidence:
    """Evidence for one method on one preserved evaluation sequence."""

    method_id: str
    sequence_id: str
    fit_regime: FitRegime
    evaluation_scope: EvaluationScope
    status: CaseStatus
    metrics: Mapping[str, float | None] | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _identifier(self.method_id, name="method_id"))
        object.__setattr__(
            self,
            "sequence_id",
            _identifier(self.sequence_id, name="sequence_id"),
        )
        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
        object.__setattr__(self, "evaluation_scope", EvaluationScope(self.evaluation_scope))
        status = CaseStatus(self.status)
        object.__setattr__(self, "status", status)

        metric_values: dict[str, float | None] = {}
        if self.metrics is not None:
            if not isinstance(self.metrics, Mapping):
                raise TypeError("metrics must be a mapping")
            for key, value in self.metrics.items():
                metric_id = _identifier(key, name="metric_id")
                metric_values[metric_id] = (
                    None if value is None else _finite_metric(value, name=f"metric {metric_id!r}")
                )

        if status is CaseStatus.OK:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful cases cannot carry error evidence")
        else:
            if metric_values:
                raise ValueError("non-success cases cannot carry scientific metric values")
            _identifier(self.error_type, name="error_type")
            _identifier(self.error_message, name="error_message")

        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "sequence_id": self.sequence_id,
            "fit_regime": self.fit_regime.value,
            "evaluation_scope": self.evaluation_scope.value,
            "status": self.status.value,
            "metrics": _portable(self.metrics),
            "error_type": self.error_type,
            "error_message": self.error_message,
            "metadata": _portable(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class MethodEvidenceSummary:
    """Derived method summary with explicit status and metric denominators."""

    method_id: str
    fit_regime: FitRegime
    evaluation_scope: EvaluationScope
    total_cases: int
    ok_cases: int
    failed_cases: int
    unavailable_cases: int
    nonconverged_cases: int
    metric_mean: Mapping[str, float | None]
    metric_n: Mapping[str, int]

    def __post_init__(self) -> None:
        object.__setattr__(self, "method_id", _identifier(self.method_id, name="method_id"))
        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
        object.__setattr__(self, "evaluation_scope", EvaluationScope(self.evaluation_scope))

        counts = tuple(
            _count(value, name=name)
            for value, name in (
                (self.total_cases, "total_cases"),
                (self.ok_cases, "ok_cases"),
                (self.failed_cases, "failed_cases"),
                (self.unavailable_cases, "unavailable_cases"),
                (self.nonconverged_cases, "nonconverged_cases"),
            )
        )
        if counts[0] <= 0:
            raise ValueError("total_cases must be positive")
        if any(value < 0 for value in counts[1:]):
            raise ValueError("case counts cannot be negative")
        if sum(counts[1:]) != counts[0]:
            raise ValueError("status counts must sum exactly to total_cases")

        means: dict[str, float | None] = {}
        if not isinstance(self.metric_mean, Mapping):
            raise TypeError("metric_mean must be a mapping")
        for key, value in self.metric_mean.items():
            metric_id = _identifier(key, name="metric_id")
            means[metric_id] = (
                None
                if value is None
                else _finite_metric(value, name=f"metric mean {metric_id!r}")
            )

        denominators: dict[str, int] = {}
        if not isinstance(self.metric_n, Mapping):
            raise TypeError("metric_n must be a mapping")
        for key, value in self.metric_n.items():
            metric_id = _identifier(key, name="metric_n id")
            numeric = _count(value, name=f"metric_n {metric_id!r}")
            if numeric < 0 or numeric > counts[1]:
                raise ValueError("metric_n values must be between zero and ok_cases")
            denominators[metric_id] = numeric
        if set(denominators) != set(means):
            raise ValueError("metric_n keys must exactly match metric_mean keys")
        for key, mean in means.items():
            if denominators[key] == 0 and mean is not None:
                raise ValueError("metrics with metric_n=0 must have mean=None")
            if denominators[key] > 0 and mean is None:
                raise ValueError("metrics with metric_n>0 must have a finite mean")

        object.__setattr__(self, "total_cases", counts[0])
        object.__setattr__(self, "ok_cases", counts[1])
        object.__setattr__(self, "failed_cases", counts[2])
        object.__setattr__(self, "unavailable_cases", counts[3])
        object.__setattr__(self, "nonconverged_cases", counts[4])
        object.__setattr__(self, "metric_mean", MappingProxyType(means))
        object.__setattr__(self, "metric_n", MappingProxyType(denominators))

    @property
    def non_ok_rate(self) -> float:
        return (self.total_cases - self.ok_cases) / self.total_cases

    @property
    def failed_rate(self) -> float:
        return self.failed_cases / self.total_cases

    @property
    def unavailable_rate(self) -> float:
        return self.unavailable_cases / self.total_cases

    @property
    def nonconverged_rate(self) -> float:
        return self.nonconverged_cases / self.total_cases


@dataclass(frozen=True, slots=True)
class RepresentationEvidenceGrid:
    """Exact Cartesian method × evaluation-sequence evidence authority."""

    train_sequence_ids: tuple[str, ...]
    evaluation_sequence_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    cases: tuple[RepresentationCaseEvidence, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        train_ids = tuple(
            _identifier(value, name="train_sequence_id") for value in self.train_sequence_ids
        )
        evaluation_ids = tuple(
            _identifier(value, name="evaluation_sequence_id")
            for value in self.evaluation_sequence_ids
        )
        method_ids = tuple(_identifier(value, name="method_id") for value in self.method_ids)
        cases = tuple(self.cases)

        if not train_ids:
            raise ValueError("train_sequence_ids cannot be empty")
        if not evaluation_ids:
            raise ValueError("evaluation_sequence_ids cannot be empty")
        if not method_ids:
            raise ValueError("method_ids cannot be empty")
        if len(set(train_ids)) != len(train_ids):
            raise ValueError("train_sequence_ids must be unique")
        if len(set(evaluation_ids)) != len(evaluation_ids):
            raise ValueError("evaluation_sequence_ids must be unique")
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("method_ids must be unique")

        expected = {
            (method_id, sequence_id)
            for method_id in method_ids
            for sequence_id in evaluation_ids
        }
        seen: set[tuple[str, str]] = set()
        regimes: dict[str, FitRegime] = {}
        scopes: dict[str, EvaluationScope] = {}
        for case in cases:
            if not isinstance(case, RepresentationCaseEvidence):
                raise TypeError("cases must contain RepresentationCaseEvidence values")
            key = (case.method_id, case.sequence_id)
            if key in seen:
                raise ValueError(f"duplicate representation case {key!r}")
            seen.add(key)
            regimes.setdefault(case.method_id, case.fit_regime)
            scopes.setdefault(case.method_id, case.evaluation_scope)
            if regimes[case.method_id] is not case.fit_regime:
                raise ValueError("all cases for a method must share one fit_regime")
            if scopes[case.method_id] is not case.evaluation_scope:
                raise ValueError("all cases for a method must share one evaluation_scope")

        missing = expected - seen
        extra = seen - expected
        if missing or extra:
            raise ValueError(
                "cases must contain the exact declared method × evaluation-sequence grid; "
                f"missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )

        object.__setattr__(self, "train_sequence_ids", train_ids)
        object.__setattr__(self, "evaluation_sequence_ids", evaluation_ids)
        object.__setattr__(self, "method_ids", method_ids)
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def cases_for_method(self, method_id: str) -> tuple[RepresentationCaseEvidence, ...]:
        method_id = _identifier(method_id, name="method_id")
        if method_id not in self.method_ids:
            raise KeyError(method_id)
        by_case = {(case.method_id, case.sequence_id): case for case in self.cases}
        return tuple(
            by_case[(method_id, sequence_id)] for sequence_id in self.evaluation_sequence_ids
        )

    def summary_for_method(self, method_id: str) -> MethodEvidenceSummary:
        cases = self.cases_for_method(method_id)
        counts = {status: 0 for status in CaseStatus}
        metric_schema: set[str] = set()
        metric_values: dict[str, list[float]] = {}
        for case in cases:
            counts[case.status] += 1
            if case.status is not CaseStatus.OK:
                continue
            metric_schema.update(case.metrics)
            for key, value in case.metrics.items():
                if value is not None:
                    metric_values.setdefault(key, []).append(value)

        means: dict[str, float | None] = {}
        denominators: dict[str, int] = {}
        for key in sorted(metric_schema):
            samples = metric_values.get(key, [])
            denominators[key] = len(samples)
            means[key] = sum(samples) / len(samples) if samples else None

        first = cases[0]
        return MethodEvidenceSummary(
            method_id=method_id,
            fit_regime=first.fit_regime,
            evaluation_scope=first.evaluation_scope,
            total_cases=len(cases),
            ok_cases=counts[CaseStatus.OK],
            failed_cases=counts[CaseStatus.FAILED],
            unavailable_cases=counts[CaseStatus.UNAVAILABLE],
            nonconverged_cases=counts[CaseStatus.NONCONVERGED],
            metric_mean=means,
            metric_n=denominators,
        )

    def summaries(self) -> tuple[MethodEvidenceSummary, ...]:
        return tuple(self.summary_for_method(method_id) for method_id in self.method_ids)

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "train_sequence_ids": sorted(self.train_sequence_ids),
            "evaluation_sequence_ids": sorted(self.evaluation_sequence_ids),
            "method_ids": sorted(self.method_ids),
            "cases": [
                case.to_manifest()
                for case in sorted(self.cases, key=lambda item: (item.method_id, item.sequence_id))
            ],
            "metadata": _portable(self.metadata),
        }

    @property
    def evidence_sha256(self) -> str:
        payload = _DIGEST_DOMAIN + _canonical_json(self.to_manifest())
        return sha256(payload).hexdigest()
