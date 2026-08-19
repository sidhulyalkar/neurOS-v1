"""Independent-execution reproduction contracts for published scientific artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np


@dataclass(frozen=True, slots=True)
class ReproductionSnapshot:
    """The minimal evidence needed to compare two independent executions."""

    artifact_family: str
    study_fingerprint: str
    run_hash: str
    execution_id: str
    decision: str
    metrics: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in self.metrics.items():
            if not np.isfinite(float(value)):
                raise ValueError(f"reproduction metric {name!r} must be finite")
        if not all((self.artifact_family, self.study_fingerprint, self.run_hash, self.execution_id)):
            raise ValueError("reproduction identities must be non-empty")
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType({name: float(value) for name, value in self.metrics.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_family": self.artifact_family,
            "study_fingerprint": self.study_fingerprint,
            "run_hash": self.run_hash,
            "execution_id": self.execution_id,
            "decision": self.decision,
            "metrics": dict(self.metrics),
        }


@dataclass(frozen=True, slots=True)
class ReproductionMetricTolerance:
    """Preregistered absolute and relative tolerance for one scientific metric."""

    metric: str
    absolute: float = 1e-6
    relative: float = 1e-6

    def __post_init__(self) -> None:
        if not self.metric:
            raise ValueError("reproduction metric name must be non-empty")
        if self.absolute < 0.0 or self.relative < 0.0:
            raise ValueError("reproduction tolerances must be non-negative")

    def allowed_delta(self, reference: float) -> float:
        return float(self.absolute + self.relative * abs(reference))

    def to_dict(self) -> dict[str, float | str]:
        return {"metric": self.metric, "absolute": self.absolute, "relative": self.relative}


@dataclass(frozen=True, slots=True)
class ReproductionSpec:
    """Frozen qualitative decision and numerical tolerances for reproduction."""

    reproduction_id: str
    artifact_family: str
    required_decision: str
    metric_tolerances: tuple[ReproductionMetricTolerance, ...] = ()
    require_same_study_fingerprint: bool = True
    require_distinct_run_hash: bool = True
    require_distinct_execution_id: bool = True

    def __post_init__(self) -> None:
        if not self.reproduction_id or not self.artifact_family or not self.required_decision:
            raise ValueError("reproduction identifiers and decision must be non-empty")
        metrics = [item.metric for item in self.metric_tolerances]
        if len(metrics) != len(set(metrics)):
            raise ValueError("reproduction metric tolerances must have unique names")

    def to_dict(self) -> dict[str, Any]:
        return {
            "reproduction_id": self.reproduction_id,
            "artifact_family": self.artifact_family,
            "required_decision": self.required_decision,
            "metric_tolerances": [item.to_dict() for item in self.metric_tolerances],
            "require_same_study_fingerprint": self.require_same_study_fingerprint,
            "require_distinct_run_hash": self.require_distinct_run_hash,
            "require_distinct_execution_id": self.require_distinct_execution_id,
        }


@dataclass(frozen=True, slots=True)
class ReproductionMetricComparison:
    metric: str
    reference: float
    candidate: float
    absolute_delta: float
    allowed_delta: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "reference": self.reference,
            "candidate": self.candidate,
            "absolute_delta": self.absolute_delta,
            "allowed_delta": self.allowed_delta,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class ReproductionResult:
    spec: ReproductionSpec
    reference: ReproductionSnapshot
    candidate: ReproductionSnapshot
    metric_comparisons: tuple[ReproductionMetricComparison, ...]
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "spec": self.spec.to_dict(),
            "reference": self.reference.to_dict(),
            "candidate": self.candidate.to_dict(),
            "metric_comparisons": [item.to_dict() for item in self.metric_comparisons],
            "passed": self.passed,
            "reasons": list(self.reasons),
        }


def assess_independent_reproduction(
    spec: ReproductionSpec,
    reference: ReproductionSnapshot,
    candidate: ReproductionSnapshot,
) -> ReproductionResult:
    """Require an independent execution to recover the preregistered qualitative decision."""

    reasons: list[str] = []
    if reference.artifact_family != spec.artifact_family:
        reasons.append("reference artifact family does not match reproduction spec")
    if candidate.artifact_family != spec.artifact_family:
        reasons.append("candidate artifact family does not match reproduction spec")
    if reference.decision != spec.required_decision:
        reasons.append("reference execution does not have the preregistered decision")
    if candidate.decision != spec.required_decision:
        reasons.append("candidate execution does not reproduce the preregistered decision")
    if (
        spec.require_same_study_fingerprint
        and reference.study_fingerprint != candidate.study_fingerprint
    ):
        reasons.append("scientific fingerprints differ")
    if spec.require_distinct_run_hash and reference.run_hash == candidate.run_hash:
        reasons.append("run hashes are identical; independent execution is not established")
    if spec.require_distinct_execution_id and reference.execution_id == candidate.execution_id:
        reasons.append("execution IDs are identical; independent execution is not established")

    comparisons: list[ReproductionMetricComparison] = []
    for tolerance in spec.metric_tolerances:
        if tolerance.metric not in reference.metrics or tolerance.metric not in candidate.metrics:
            reasons.append(f"metric {tolerance.metric!r} missing from reproduction snapshot")
            continue
        reference_value = reference.metrics[tolerance.metric]
        candidate_value = candidate.metrics[tolerance.metric]
        delta = abs(candidate_value - reference_value)
        allowed = tolerance.allowed_delta(reference_value)
        passed = delta <= allowed
        comparisons.append(
            ReproductionMetricComparison(
                metric=tolerance.metric,
                reference=reference_value,
                candidate=candidate_value,
                absolute_delta=delta,
                allowed_delta=allowed,
                passed=passed,
            )
        )
        if not passed:
            reasons.append(
                f"metric {tolerance.metric!r} delta {delta:.6g} exceeds {allowed:.6g}"
            )

    return ReproductionResult(
        spec=spec,
        reference=reference,
        candidate=candidate,
        metric_comparisons=tuple(comparisons),
        passed=not reasons,
        reasons=tuple(reasons),
    )
