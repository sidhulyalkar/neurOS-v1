"""Failure-preserving multi-case sweeps for representation benchmarks."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .contracts import FitRegime, MethodOutcome, MethodStatus, RepresentationBenchmarkResult


def _freeze_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if value is None:
        return MappingProxyType({})
    return MappingProxyType(dict(value))


@dataclass(frozen=True, slots=True)
class SweepCase:
    """One predeclared controlled experiment coordinate."""

    noise_std: float
    seed: int

    def __post_init__(self) -> None:
        noise = float(self.noise_std)
        if not math.isfinite(noise) or noise < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        if isinstance(self.seed, bool):
            raise TypeError("seed must be an integer")
        seed = int(self.seed)
        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "seed", seed)

    @property
    def case_id(self) -> str:
        return f"noise={self.noise_std:.12g}|seed={self.seed}"

    def to_dict(self) -> dict[str, float | int | str]:
        return {"case_id": self.case_id, "noise_std": self.noise_std, "seed": self.seed}


@dataclass(frozen=True, slots=True)
class CaseMethodEvidence:
    """One method outcome at one declared sweep coordinate."""

    case: SweepCase
    method_id: str
    fit_regime: FitRegime
    status: MethodStatus
    metrics: Mapping[str, float | None] | None = None
    error_type: str | None = None
    error_message: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        regime = FitRegime(self.fit_regime)
        status = MethodStatus(self.status)
        metrics = dict(self.metrics or {})
        normalized: dict[str, float | None] = {}
        for name, value in metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("metric IDs must be nonblank strings")
            if value is None:
                normalized[name] = None
            else:
                numeric = float(value)
                if not math.isfinite(numeric):
                    raise ValueError("metric values must be finite or None")
                normalized[name] = numeric
        if status is MethodStatus.OK:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful case evidence cannot carry error fields")
        else:
            if normalized:
                raise ValueError("failed/unavailable case evidence cannot carry scientific metrics")
            if not self.error_type or not self.error_message:
                raise ValueError("failed/unavailable case evidence requires explicit error evidence")
        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", MappingProxyType(normalized))

    @classmethod
    def from_outcome(cls, case: SweepCase, outcome: MethodOutcome) -> CaseMethodEvidence:
        metrics = outcome.metrics if outcome.status is MethodStatus.OK else None
        return cls(
            case=case,
            method_id=outcome.method_id,
            fit_regime=outcome.fit_regime,
            status=outcome.status,
            metrics=metrics,
            error_type=outcome.error_type,
            error_message=outcome.error_message,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.case.to_dict(),
            "method_id": self.method_id,
            "fit_regime": self.fit_regime.value,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }


@dataclass(frozen=True, slots=True)
class MethodSweepSummary:
    """Aggregate diagnostics that retain explicit denominators and failure counts."""

    method_id: str
    declared_cases: int
    ok_cases: int
    failed_cases: int
    unavailable_cases: int
    metric_summaries: Mapping[str, Mapping[str, float | int]]

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        counts = tuple(
            int(value)
            for value in (
                self.declared_cases,
                self.ok_cases,
                self.failed_cases,
                self.unavailable_cases,
            )
        )
        if any(value < 0 for value in counts):
            raise ValueError("summary counts must be nonnegative")
        if counts[1] + counts[2] + counts[3] != counts[0]:
            raise ValueError("summary status counts must equal declared_cases")
        frozen = {
            key: MappingProxyType(dict(value)) for key, value in self.metric_summaries.items()
        }
        object.__setattr__(self, "metric_summaries", MappingProxyType(frozen))

    @property
    def success_rate(self) -> float:
        return self.ok_cases / self.declared_cases

    @property
    def failure_rate(self) -> float:
        return (self.failed_cases + self.unavailable_cases) / self.declared_cases

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "declared_cases": self.declared_cases,
            "ok_cases": self.ok_cases,
            "failed_cases": self.failed_cases,
            "unavailable_cases": self.unavailable_cases,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "metric_summaries": {
                key: dict(value) for key, value in self.metric_summaries.items()
            },
        }


@dataclass(frozen=True, slots=True)
class RepresentationSweepResult:
    """Complete method x case evidence with no ranking or winner field."""

    cases: tuple[SweepCase, ...]
    evidence: tuple[CaseMethodEvidence, ...]
    summaries: tuple[MethodSweepSummary, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        evidence = tuple(self.evidence)
        summaries = tuple(self.summaries)
        if not cases or not evidence or not summaries:
            raise ValueError("sweep result requires cases, evidence, and summaries")
        case_ids = [case.case_id for case in cases]
        if len(set(case_ids)) != len(case_ids):
            raise ValueError("sweep case IDs must be unique")
        method_ids = [summary.method_id for summary in summaries]
        if len(set(method_ids)) != len(method_ids):
            raise ValueError("sweep summaries must have unique method IDs")
        actual = [(row.case.case_id, row.method_id) for row in evidence]
        if len(set(actual)) != len(actual):
            raise ValueError("method x case evidence pairs must be unique")
        expected = {(case_id, method_id) for case_id in case_ids for method_id in method_ids}
        if set(actual) != expected:
            missing = sorted(expected - set(actual))
            extra = sorted(set(actual) - expected)
            raise ValueError(f"method x case evidence must be Cartesian complete: missing={missing}, extra={extra}")
        if any(summary.declared_cases != len(cases) for summary in summaries):
            raise ValueError("every summary denominator must equal the declared case count")
        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "summaries", summaries)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "evidence": [row.to_dict() for row in self.evidence],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "metadata": dict(self.metadata),
        }


def _metric_summaries(rows: tuple[CaseMethodEvidence, ...]) -> dict[str, dict[str, float | int]]:
    metric_ids = sorted({name for row in rows for name in row.metrics})
    summaries: dict[str, dict[str, float | int]] = {}
    for metric_id in metric_ids:
        values = np.asarray(
            [row.metrics[metric_id] for row in rows if row.metrics.get(metric_id) is not None],
            dtype=float,
        )
        if values.size == 0:
            continue
        summaries[metric_id] = {
            "n": int(values.size),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return summaries


def _summarize_method(method_id: str, rows: tuple[CaseMethodEvidence, ...]) -> MethodSweepSummary:
    return MethodSweepSummary(
        method_id=method_id,
        declared_cases=len(rows),
        ok_cases=sum(row.status is MethodStatus.OK for row in rows),
        failed_cases=sum(row.status is MethodStatus.FAILED for row in rows),
        unavailable_cases=sum(row.status is MethodStatus.UNAVAILABLE for row in rows),
        metric_summaries=_metric_summaries(rows),
    )


def build_representation_sweep(
    case_results: Iterable[tuple[SweepCase, RepresentationBenchmarkResult]],
    *,
    metadata: Mapping[str, Any] | None = None,
) -> RepresentationSweepResult:
    """Build Cartesian-complete case evidence from independent benchmark results."""
    rows = tuple(case_results)
    if not rows:
        raise ValueError("case_results must be non-empty")
    cases = tuple(case for case, _ in rows)
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("case_results contains duplicate sweep cases")

    first_methods = tuple(outcome.method_id for outcome in rows[0][1].outcomes)
    expected_methods = set(first_methods)
    if not first_methods:
        raise ValueError("benchmark results must contain methods")

    evidence: list[CaseMethodEvidence] = []
    regimes: dict[str, FitRegime] = {}
    for case, result in rows:
        by_method = result.by_method()
        if set(by_method) != expected_methods:
            raise ValueError("every sweep case must contain the same method ID set")
        for method_id in first_methods:
            outcome = by_method[method_id]
            prior_regime = regimes.setdefault(method_id, outcome.fit_regime)
            if prior_regime is not outcome.fit_regime:
                raise ValueError("a method cannot change fit regime across sweep cases")
            evidence.append(CaseMethodEvidence.from_outcome(case, outcome))

    evidence_tuple = tuple(evidence)
    summaries = tuple(
        _summarize_method(
            method_id,
            tuple(row for row in evidence_tuple if row.method_id == method_id),
        )
        for method_id in first_methods
    )
    combined_metadata = {
        "ranking_policy": "none",
        "claim_scope": "controlled_representation_geometry",
        "case_authority": "method_x_noise_x_seed_cartesian_complete",
        **dict(metadata or {}),
    }
    return RepresentationSweepResult(
        cases=cases,
        evidence=evidence_tuple,
        summaries=summaries,
        metadata=combined_metadata,
    )
