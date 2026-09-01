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


def _nonnegative_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(f"{name} must be a finite nonnegative real")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return numeric


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
    operational: Mapping[str, float] | None = None
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

        operational_values: dict[str, float] = {}
        for name, value in dict(self.operational or {}).items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("operational metric IDs must be nonblank strings")
            operational_values[name] = _nonnegative_real(
                value,
                name=f"operational metric {name!r}",
            )

        if status is MethodStatus.OK:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful case evidence cannot carry error fields")
        else:
            if normalized:
                raise ValueError("failed/unavailable case evidence cannot carry scientific metrics")
            if not self.error_type or not self.error_message:
                raise ValueError(
                    "failed/unavailable case evidence requires explicit error evidence"
                )
        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", MappingProxyType(normalized))
        object.__setattr__(self, "operational", MappingProxyType(operational_values))

    @classmethod
    def from_outcome(cls, case: SweepCase, outcome: MethodOutcome) -> CaseMethodEvidence:
        metrics = outcome.metrics if outcome.status is MethodStatus.OK else None
        operational: dict[str, float] = {}
        runtime = outcome.metadata.get("runtime_seconds")
        if runtime is not None:
            operational["runtime_seconds"] = float(runtime)
        return cls(
            case=case,
            method_id=outcome.method_id,
            fit_regime=outcome.fit_regime,
            status=outcome.status,
            metrics=metrics,
            operational=operational,
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
            "operational": dict(self.operational),
            "error_type": self.error_type,
            "error_message": self.error_message,
        }

    def to_scientific_dict(self) -> dict[str, Any]:
        row = self.to_dict()
        row.pop("operational")
        return row


def _validated_counts(
    declared_cases: int,
    ok_cases: int,
    failed_cases: int,
    unavailable_cases: int,
) -> tuple[int, int, int, int]:
    counts = tuple(
        int(value)
        for value in (
            declared_cases,
            ok_cases,
            failed_cases,
            unavailable_cases,
        )
    )
    if any(value < 0 for value in counts):
        raise ValueError("summary counts must be nonnegative")
    if counts[1] + counts[2] + counts[3] != counts[0]:
        raise ValueError("summary status counts must equal declared_cases")
    return counts


def _freeze_summaries(
    summaries: Mapping[str, Mapping[str, float | int]],
) -> Mapping[str, Mapping[str, float | int]]:
    return MappingProxyType(
        {key: MappingProxyType(dict(value)) for key, value in summaries.items()}
    )


@dataclass(frozen=True, slots=True)
class MethodSweepSummary:
    """Aggregate diagnostics that retain explicit denominators and failure counts."""

    method_id: str
    declared_cases: int
    ok_cases: int
    failed_cases: int
    unavailable_cases: int
    metric_summaries: Mapping[str, Mapping[str, float | int]]
    operational_summaries: Mapping[str, Mapping[str, float | int]]

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        _validated_counts(
            self.declared_cases,
            self.ok_cases,
            self.failed_cases,
            self.unavailable_cases,
        )
        object.__setattr__(self, "metric_summaries", _freeze_summaries(self.metric_summaries))
        object.__setattr__(
            self,
            "operational_summaries",
            _freeze_summaries(self.operational_summaries),
        )

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
            "operational_summaries": {
                key: dict(value) for key, value in self.operational_summaries.items()
            },
        }

    def to_scientific_dict(self) -> dict[str, Any]:
        row = self.to_dict()
        row.pop("operational_summaries")
        return row


@dataclass(frozen=True, slots=True)
class NoiseMethodSummary:
    """Method evidence at one noise coordinate with uncertainty across declared seeds."""

    method_id: str
    noise_std: float
    declared_seeds: int
    ok_seeds: int
    failed_seeds: int
    unavailable_seeds: int
    metric_summaries: Mapping[str, Mapping[str, float | int]]
    operational_summaries: Mapping[str, Mapping[str, float | int]]

    def __post_init__(self) -> None:
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        object.__setattr__(
            self,
            "noise_std",
            _nonnegative_real(self.noise_std, name="noise_std"),
        )
        _validated_counts(
            self.declared_seeds,
            self.ok_seeds,
            self.failed_seeds,
            self.unavailable_seeds,
        )
        object.__setattr__(self, "metric_summaries", _freeze_summaries(self.metric_summaries))
        object.__setattr__(
            self,
            "operational_summaries",
            _freeze_summaries(self.operational_summaries),
        )

    @property
    def success_rate(self) -> float:
        return self.ok_seeds / self.declared_seeds

    @property
    def failure_rate(self) -> float:
        return (self.failed_seeds + self.unavailable_seeds) / self.declared_seeds

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "noise_std": self.noise_std,
            "declared_seeds": self.declared_seeds,
            "ok_seeds": self.ok_seeds,
            "failed_seeds": self.failed_seeds,
            "unavailable_seeds": self.unavailable_seeds,
            "success_rate": self.success_rate,
            "failure_rate": self.failure_rate,
            "metric_summaries": {
                key: dict(value) for key, value in self.metric_summaries.items()
            },
            "operational_summaries": {
                key: dict(value) for key, value in self.operational_summaries.items()
            },
        }

    def to_scientific_dict(self) -> dict[str, Any]:
        row = self.to_dict()
        row.pop("operational_summaries")
        return row


@dataclass(frozen=True, slots=True)
class RepresentationSweepResult:
    """Complete method x noise x seed evidence with no ranking or winner field."""

    cases: tuple[SweepCase, ...]
    evidence: tuple[CaseMethodEvidence, ...]
    summaries: tuple[MethodSweepSummary, ...]
    noise_summaries: tuple[NoiseMethodSummary, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        cases = tuple(self.cases)
        evidence = tuple(self.evidence)
        summaries = tuple(self.summaries)
        noise_summaries = tuple(self.noise_summaries)
        if not cases or not evidence or not summaries or not noise_summaries:
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
            raise ValueError(
                "method x case evidence must be Cartesian complete: "
                f"missing={missing}, extra={extra}"
            )
        if any(summary.declared_cases != len(cases) for summary in summaries):
            raise ValueError("every summary denominator must equal the declared case count")

        noises = tuple(sorted({case.noise_std for case in cases}))
        seeds = tuple(sorted({case.seed for case in cases}))
        expected_cases = {(noise, seed) for noise in noises for seed in seeds}
        actual_cases = {(case.noise_std, case.seed) for case in cases}
        if actual_cases != expected_cases:
            missing = sorted(expected_cases - actual_cases)
            extra = sorted(actual_cases - expected_cases)
            raise ValueError(
                "noise x seed case grid must be Cartesian complete: "
                f"missing={missing}, extra={extra}"
            )

        expected_noise_summaries = {
            (method_id, noise) for method_id in method_ids for noise in noises
        }
        actual_noise_summaries = {
            (summary.method_id, summary.noise_std) for summary in noise_summaries
        }
        if len(actual_noise_summaries) != len(noise_summaries):
            raise ValueError("noise summaries must have unique method x noise identities")
        if actual_noise_summaries != expected_noise_summaries:
            raise ValueError("noise summaries must be method x noise Cartesian complete")
        if any(summary.declared_seeds != len(seeds) for summary in noise_summaries):
            raise ValueError("every noise summary denominator must equal the declared seed count")

        object.__setattr__(self, "cases", cases)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "summaries", summaries)
        object.__setattr__(self, "noise_summaries", noise_summaries)
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))

    def to_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "evidence": [row.to_dict() for row in self.evidence],
            "summaries": [summary.to_dict() for summary in self.summaries],
            "noise_summaries": [summary.to_dict() for summary in self.noise_summaries],
            "metadata": dict(self.metadata),
        }

    def to_scientific_dict(self) -> dict[str, Any]:
        return {
            "cases": [case.to_dict() for case in self.cases],
            "evidence": [row.to_scientific_dict() for row in self.evidence],
            "summaries": [summary.to_scientific_dict() for summary in self.summaries],
            "noise_summaries": [
                summary.to_scientific_dict() for summary in self.noise_summaries
            ],
            "metadata": dict(self.metadata),
        }


def _numeric_summary(values: np.ndarray) -> dict[str, float | int]:
    if values.size == 0:
        raise ValueError("cannot summarize an empty numeric vector")
    sample_std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    sem = sample_std / math.sqrt(values.size)
    return {
        "n": int(values.size),
        "mean": float(np.mean(values)),
        "std": sample_std,
        "sem": float(sem),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def _metric_summaries(
    rows: tuple[CaseMethodEvidence, ...],
) -> dict[str, dict[str, float | int]]:
    metric_ids = sorted({name for row in rows for name in row.metrics})
    summaries: dict[str, dict[str, float | int]] = {}
    for metric_id in metric_ids:
        values = np.asarray(
            [row.metrics[metric_id] for row in rows if row.metrics.get(metric_id) is not None],
            dtype=float,
        )
        if values.size:
            summaries[metric_id] = _numeric_summary(values)
    return summaries


def _operational_summaries(
    rows: tuple[CaseMethodEvidence, ...],
) -> dict[str, dict[str, float | int]]:
    metric_ids = sorted({name for row in rows for name in row.operational})
    summaries: dict[str, dict[str, float | int]] = {}
    for metric_id in metric_ids:
        values = np.asarray(
            [row.operational[metric_id] for row in rows if metric_id in row.operational],
            dtype=float,
        )
        if values.size:
            summaries[metric_id] = _numeric_summary(values)
    return summaries


def _summarize_method(
    method_id: str,
    rows: tuple[CaseMethodEvidence, ...],
) -> MethodSweepSummary:
    return MethodSweepSummary(
        method_id=method_id,
        declared_cases=len(rows),
        ok_cases=sum(row.status is MethodStatus.OK for row in rows),
        failed_cases=sum(row.status is MethodStatus.FAILED for row in rows),
        unavailable_cases=sum(row.status is MethodStatus.UNAVAILABLE for row in rows),
        metric_summaries=_metric_summaries(rows),
        operational_summaries=_operational_summaries(rows),
    )


def _summarize_method_noise(
    method_id: str,
    noise_std: float,
    rows: tuple[CaseMethodEvidence, ...],
) -> NoiseMethodSummary:
    return NoiseMethodSummary(
        method_id=method_id,
        noise_std=noise_std,
        declared_seeds=len(rows),
        ok_seeds=sum(row.status is MethodStatus.OK for row in rows),
        failed_seeds=sum(row.status is MethodStatus.FAILED for row in rows),
        unavailable_seeds=sum(row.status is MethodStatus.UNAVAILABLE for row in rows),
        metric_summaries=_metric_summaries(rows),
        operational_summaries=_operational_summaries(rows),
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

    noises = tuple(sorted({case.noise_std for case in cases}))
    seeds = tuple(sorted({case.seed for case in cases}))
    expected_cases = {(noise, seed) for noise in noises for seed in seeds}
    actual_cases = {(case.noise_std, case.seed) for case in cases}
    if actual_cases != expected_cases:
        missing = sorted(expected_cases - actual_cases)
        extra = sorted(actual_cases - expected_cases)
        raise ValueError(
            "case_results must form a noise x seed Cartesian grid: "
            f"missing={missing}, extra={extra}"
        )

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
    noise_summaries = tuple(
        _summarize_method_noise(
            method_id,
            noise,
            tuple(
                row
                for row in evidence_tuple
                if row.method_id == method_id and row.case.noise_std == noise
            ),
        )
        for method_id in first_methods
        for noise in noises
    )
    combined_metadata = {
        "ranking_policy": "none",
        "claim_scope": "controlled_representation_geometry",
        "case_authority": "method_x_noise_x_seed_cartesian_complete",
        "uncertainty_unit": "seed",
        "operational_domain": "separate_from_scientific_metrics",
        **dict(metadata or {}),
    }
    return RepresentationSweepResult(
        cases=cases,
        evidence=evidence_tuple,
        summaries=summaries,
        noise_summaries=noise_summaries,
        metadata=combined_metadata,
    )
