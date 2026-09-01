"""Controlled multi-seed noise sweeps for representation geometry."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .cases import CasePreservingRepresentationBenchmark, CaseStatus
from .contracts import FitRegime, RepresentationMethod, _freeze_metadata
from .synthetic import make_controlled_temporal_manifold


def _strict_nonnegative_real(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite nonnegative real")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return numeric


def _strict_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    return int(value)


@dataclass(frozen=True, slots=True)
class SweepCaseRecord:
    """Compact case evidence for one method × sequence × noise × seed point."""

    noise_std: float
    seed: int
    method_id: str
    sequence_id: str
    fit_regime: FitRegime
    status: CaseStatus
    metrics: Mapping[str, float | None] | None = None
    error_type: str | None = None
    error_message: str | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        noise = _strict_nonnegative_real(self.noise_std, name="noise_std")
        seed = _strict_int(self.seed, name="seed")
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a nonblank string")
        status = CaseStatus(self.status)
        regime = FitRegime(self.fit_regime)

        metric_values: dict[str, float | None] = {}
        if self.metrics is not None:
            for key, value in dict(self.metrics).items():
                if not isinstance(key, str) or not key.strip():
                    raise ValueError("metric IDs must be nonblank strings")
                if value is None:
                    metric_values[key] = None
                else:
                    numeric = float(value)
                    if not np.isfinite(numeric):
                        raise ValueError("metric values must be finite or None")
                    metric_values[key] = numeric

        if status is CaseStatus.OK:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError(
                    "successful sweep records cannot carry error evidence"
                )
        else:
            if metric_values:
                raise ValueError(
                    "non-success sweep records cannot carry scientific metrics"
                )
            if not self.error_type or not self.error_message:
                raise ValueError(
                    "non-success sweep records require explicit error evidence"
                )

        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", MappingProxyType(metric_values))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


def _finite_metric_mapping(
    values: Mapping[str, float | None],
) -> Mapping[str, float | None]:
    output: dict[str, float | None] = {}
    for key, value in dict(values).items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric IDs must be nonblank strings")
        if value is None:
            output[key] = None
        else:
            numeric = float(value)
            if not np.isfinite(numeric):
                raise ValueError("metric summary values must be finite or None")
            output[key] = numeric
    return MappingProxyType(output)


@dataclass(frozen=True, slots=True)
class NoiseLevelSummary:
    """Seed-level aggregate at one method/noise point with visible denominator."""

    method_id: str
    fit_regime: FitRegime
    noise_std: float
    total_cases: int
    ok_cases: int
    failed_cases: int
    unavailable_cases: int
    nonconverged_cases: int
    metric_mean: Mapping[str, float | None]
    metric_std: Mapping[str, float | None]
    metric_sem: Mapping[str, float | None]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        noise = _strict_nonnegative_real(self.noise_std, name="noise_std")
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
        if counts[0] <= 0 or any(value < 0 for value in counts[1:]):
            raise ValueError("summary case counts are invalid")
        if sum(counts[1:]) != counts[0]:
            raise ValueError("summary status counts must equal total_cases")
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        object.__setattr__(self, "noise_std", noise)
        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
        object.__setattr__(self, "total_cases", counts[0])
        object.__setattr__(self, "ok_cases", counts[1])
        object.__setattr__(self, "failed_cases", counts[2])
        object.__setattr__(self, "unavailable_cases", counts[3])
        object.__setattr__(self, "nonconverged_cases", counts[4])
        object.__setattr__(
            self, "metric_mean", _finite_metric_mapping(self.metric_mean)
        )
        object.__setattr__(
            self, "metric_std", _finite_metric_mapping(self.metric_std)
        )
        object.__setattr__(
            self, "metric_sem", _finite_metric_mapping(self.metric_sem)
        )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def failure_rate(self) -> float:
        return float((self.total_cases - self.ok_cases) / self.total_cases)


@dataclass(frozen=True, slots=True)
class ControlledNoiseSweepResult:
    """Complete controlled noise × seed × method × sequence evidence grid."""

    noise_levels: tuple[float, ...]
    seeds: tuple[int, ...]
    method_ids: tuple[str, ...]
    evaluation_sequence_ids: tuple[str, ...]
    records: tuple[SweepCaseRecord, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        noise_levels = tuple(
            _strict_nonnegative_real(value, name="noise level")
            for value in self.noise_levels
        )
        seeds = tuple(_strict_int(value, name="seed") for value in self.seeds)
        method_ids = tuple(self.method_ids)
        sequence_ids = tuple(self.evaluation_sequence_ids)
        records = tuple(self.records)
        if not noise_levels:
            raise ValueError("noise_levels must be nonempty")
        if len(set(noise_levels)) != len(noise_levels):
            raise ValueError("noise_levels must be unique")
        if not seeds or len(set(seeds)) != len(seeds):
            raise ValueError("seeds must be nonempty and unique")
        if not method_ids or len(set(method_ids)) != len(method_ids):
            raise ValueError("method_ids must be nonempty and unique")
        if not sequence_ids or len(set(sequence_ids)) != len(sequence_ids):
            raise ValueError(
                "evaluation_sequence_ids must be nonempty and unique"
            )
        if any(
            not isinstance(value, str) or not value.strip() for value in method_ids
        ):
            raise ValueError("method_ids must contain only nonblank strings")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in sequence_ids
        ):
            raise ValueError(
                "evaluation_sequence_ids must contain only nonblank strings"
            )
        if len(sequence_ids) != 1:
            raise ValueError(
                "controlled seed-level sweep authority currently requires exactly one "
                "evaluation trajectory per seed"
            )

        expected = {
            (noise, seed, method_id, sequence_id)
            for noise in noise_levels
            for seed in seeds
            for method_id in method_ids
            for sequence_id in sequence_ids
        }
        seen: set[tuple[float, int, str, str]] = set()
        regimes: dict[str, FitRegime] = {}
        for record in records:
            key = (
                record.noise_std,
                record.seed,
                record.method_id,
                record.sequence_id,
            )
            if key in seen:
                raise ValueError(f"duplicate sweep record {key!r}")
            seen.add(key)
            existing = regimes.setdefault(record.method_id, record.fit_regime)
            if existing is not record.fit_regime:
                raise ValueError("fit regime changed across sweep points")
        missing = expected - seen
        extra = seen - expected
        if missing or extra:
            raise ValueError(
                "sweep result must contain the exact declared noise × seed × method × "
                f"sequence grid; missing={sorted(missing)!r}, extra={sorted(extra)!r}"
            )

        object.__setattr__(self, "noise_levels", noise_levels)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "method_ids", method_ids)
        object.__setattr__(self, "evaluation_sequence_ids", sequence_ids)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def records_for(
        self,
        method_id: str,
        noise_std: float,
    ) -> tuple[SweepCaseRecord, ...]:
        noise_std = _strict_nonnegative_real(noise_std, name="noise_std")
        if method_id not in self.method_ids:
            raise KeyError(method_id)
        if noise_std not in self.noise_levels:
            raise KeyError(noise_std)
        return tuple(
            record
            for record in self.records
            if record.method_id == method_id and record.noise_std == noise_std
        )

    def summary(self, method_id: str, noise_std: float) -> NoiseLevelSummary:
        records = self.records_for(method_id, noise_std)
        counts = {status: 0 for status in CaseStatus}
        values: dict[str, list[float]] = {}
        metric_ids: set[str] = set()
        for record in records:
            counts[record.status] += 1
            if record.status is not CaseStatus.OK:
                continue
            metric_ids.update(record.metrics)
            for key, value in record.metrics.items():
                if value is not None:
                    values.setdefault(key, []).append(float(value))

        means: dict[str, float | None] = {}
        stds: dict[str, float | None] = {}
        sems: dict[str, float | None] = {}
        for key in sorted(metric_ids):
            samples = np.asarray(values.get(key, ()), dtype=float)
            if samples.size == 0:
                means[key] = stds[key] = sems[key] = None
                continue
            means[key] = float(np.mean(samples))
            if samples.size < 2:
                stds[key] = None
                sems[key] = None
            else:
                std = float(np.std(samples, ddof=1))
                stds[key] = std
                sems[key] = float(std / np.sqrt(samples.size))

        return NoiseLevelSummary(
            method_id=method_id,
            fit_regime=records[0].fit_regime,
            noise_std=noise_std,
            total_cases=len(records),
            ok_cases=counts[CaseStatus.OK],
            failed_cases=counts[CaseStatus.FAILED],
            unavailable_cases=counts[CaseStatus.UNAVAILABLE],
            nonconverged_cases=counts[CaseStatus.NONCONVERGED],
            metric_mean=means,
            metric_std=stds,
            metric_sem=sems,
            metadata={
                "uncertainty_unit": "independent_controlled_seed",
                "declared_seed_count": len(self.seeds),
                "successful_metric_cases": counts[CaseStatus.OK],
            },
        )

    def summaries(self) -> tuple[NoiseLevelSummary, ...]:
        return tuple(
            self.summary(method_id, noise)
            for method_id in self.method_ids
            for noise in self.noise_levels
        )


def _validated_noise_levels(values: Iterable[float]) -> tuple[float, ...]:
    output = tuple(
        _strict_nonnegative_real(value, name="noise level") for value in values
    )
    if not output or len(set(output)) != len(output):
        raise ValueError("noise levels must be nonempty and unique")
    return output


def _validated_seeds(values: Iterable[int]) -> tuple[int, ...]:
    output = tuple(_strict_int(value, name="seed") for value in values)
    if not output or len(set(output)) != len(output):
        raise ValueError("seeds must be nonempty and unique")
    return output


def run_controlled_noise_sweep(
    method_factory: Callable[[], Iterable[RepresentationMethod]],
    *,
    noise_levels: Iterable[float],
    seeds: Iterable[int],
    neighborhood_k: int = 5,
) -> ControlledNoiseSweepResult:
    """Run a deterministic controlled sweep without retaining large embeddings."""

    noise_levels = _validated_noise_levels(noise_levels)
    seeds = _validated_seeds(seeds)
    records: list[SweepCaseRecord] = []
    declared_method_ids: tuple[str, ...] | None = None
    declared_regimes: dict[str, FitRegime] | None = None
    evaluation_ids: tuple[str, ...] | None = None

    for noise_std in noise_levels:
        for seed in seeds:
            data = make_controlled_temporal_manifold(
                noise_std=noise_std,
                seed=seed,
            )
            methods = tuple(method_factory())
            if not methods:
                raise ValueError("method_factory must return at least one method")
            method_ids = tuple(method.method_id for method in methods)
            regimes = {
                method.method_id: FitRegime(method.fit_regime) for method in methods
            }
            if declared_method_ids is None:
                declared_method_ids = method_ids
                declared_regimes = regimes
            elif method_ids != declared_method_ids or regimes != declared_regimes:
                raise ValueError(
                    "method_factory must return the same ordered method IDs and fit "
                    "regimes at every sweep point"
                )
            if evaluation_ids is None:
                evaluation_ids = data.evaluation.sequence_ids
            elif data.evaluation.sequence_ids != evaluation_ids:
                raise ValueError("controlled generator changed evaluation identity")

            result = CasePreservingRepresentationBenchmark(
                methods,
                neighborhood_k=neighborhood_k,
            ).run(
                data.train,
                data.evaluation,
                reference=data.reference,
            )
            for case in result.cases:
                records.append(
                    SweepCaseRecord(
                        noise_std=noise_std,
                        seed=seed,
                        method_id=case.method_id,
                        sequence_id=case.sequence_id,
                        fit_regime=case.fit_regime,
                        status=case.status,
                        metrics=case.metrics,
                        error_type=case.error_type,
                        error_message=case.error_message,
                        metadata={
                            "case_metadata": dict(case.metadata),
                            "generator_metadata": dict(data.metadata),
                        },
                    )
                )

    assert declared_method_ids is not None
    assert evaluation_ids is not None
    return ControlledNoiseSweepResult(
        noise_levels=noise_levels,
        seeds=seeds,
        method_ids=declared_method_ids,
        evaluation_sequence_ids=evaluation_ids,
        records=tuple(records),
        metadata={
            "schema": "neuros.representation.controlled_noise_sweep.v1",
            "ranking_policy": "none",
            "claim_scope": "representation_geometry",
            "uncertainty_unit": "independent_controlled_seed",
            "generator": "controlled_temporal_manifold.v2",
        },
    )
