"""Failure-preserving controlled temporal-corruption ablation sweeps."""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np

from .cases import CasePreservingRepresentationBenchmark, CaseStatus
from .contracts import (
    EvaluationScope,
    FitRegime,
    RepresentationMethod,
    _freeze_metadata,
    _strict_metric_value,
)
from .corruptions import TemporalCorruption, make_controlled_corruption_manifold


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
    numeric = int(value)
    if numeric < 0:
        raise ValueError(f"{name} must be nonnegative")
    return numeric


def _metric_mapping(
    values: Mapping[str, float | None],
) -> Mapping[str, float | None]:
    output: dict[str, float | None] = {}
    for key, value in dict(values).items():
        if not isinstance(key, str) or not key.strip():
            raise ValueError("metric IDs must be nonblank strings")
        output[key] = (
            None
            if value is None
            else _strict_metric_value(value, name=f"metric {key!r}")
        )
    return MappingProxyType(output)


@dataclass(frozen=True, slots=True)
class TemporalAblationRecord:
    corruption: TemporalCorruption
    corruption_scale: float
    seed: int
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
        kind = TemporalCorruption(self.corruption)
        scale = _strict_nonnegative_real(
            self.corruption_scale,
            name="corruption_scale",
        )
        seed = _strict_int(self.seed, name="seed")
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")
        if not isinstance(self.sequence_id, str) or not self.sequence_id.strip():
            raise ValueError("sequence_id must be a nonblank string")
        regime = FitRegime(self.fit_regime)
        scope = EvaluationScope(self.evaluation_scope)
        status = CaseStatus(self.status)
        metrics = _metric_mapping(self.metrics or {})
        if status is CaseStatus.OK:
            if self.error_type is not None or self.error_message is not None:
                raise ValueError("successful records cannot carry error evidence")
        else:
            if metrics:
                raise ValueError(
                    "non-success records cannot carry scientific metrics"
                )
            if not self.error_type or not self.error_message:
                raise ValueError(
                    "non-success records require explicit error evidence"
                )

        object.__setattr__(self, "corruption", kind)
        object.__setattr__(self, "corruption_scale", scale)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "fit_regime", regime)
        object.__setattr__(self, "evaluation_scope", scope)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TemporalAblationSummary:
    method_id: str
    corruption: TemporalCorruption
    corruption_scale: float
    fit_regime: FitRegime
    evaluation_scope: EvaluationScope
    total_cases: int
    ok_cases: int
    failed_cases: int
    unavailable_cases: int
    nonconverged_cases: int
    metric_mean: Mapping[str, float | None]
    metric_std: Mapping[str, float | None]
    metric_sem: Mapping[str, float | None]
    metric_n: Mapping[str, int]

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
        normalized = tuple(int(value) for value in counts)
        if normalized[0] <= 0 or any(value < 0 for value in normalized[1:]):
            raise ValueError("summary case counts are invalid")
        if sum(normalized[1:]) != normalized[0]:
            raise ValueError("summary status counts must equal total_cases")
        if not isinstance(self.method_id, str) or not self.method_id.strip():
            raise ValueError("method_id must be a nonblank string")

        mean = _metric_mapping(self.metric_mean)
        std = _metric_mapping(self.metric_std)
        sem = _metric_mapping(self.metric_sem)
        if set(std) != set(mean) or set(sem) != set(mean):
            raise ValueError("summary metric schemas must match exactly")

        metric_n: dict[str, int] = {}
        for key, value in dict(self.metric_n).items():
            if key not in mean:
                raise ValueError(
                    "metric_n keys must exactly match metric means"
                )
            if isinstance(value, bool) or not isinstance(
                value,
                (int, np.integer),
            ):
                raise TypeError("metric_n values must be integers")
            numeric = int(value)
            if numeric < 0 or numeric > normalized[1]:
                raise ValueError(
                    "metric_n must be between zero and ok_cases"
                )
            metric_n[key] = numeric
        if set(metric_n) != set(mean):
            raise ValueError("metric_n keys must exactly match metric means")

        object.__setattr__(
            self,
            "corruption",
            TemporalCorruption(self.corruption),
        )
        object.__setattr__(
            self,
            "corruption_scale",
            _strict_nonnegative_real(
                self.corruption_scale,
                name="corruption_scale",
            ),
        )
        object.__setattr__(self, "fit_regime", FitRegime(self.fit_regime))
        object.__setattr__(
            self,
            "evaluation_scope",
            EvaluationScope(self.evaluation_scope),
        )
        object.__setattr__(self, "total_cases", normalized[0])
        object.__setattr__(self, "ok_cases", normalized[1])
        object.__setattr__(self, "failed_cases", normalized[2])
        object.__setattr__(self, "unavailable_cases", normalized[3])
        object.__setattr__(self, "nonconverged_cases", normalized[4])
        object.__setattr__(self, "metric_mean", mean)
        object.__setattr__(self, "metric_std", std)
        object.__setattr__(self, "metric_sem", sem)
        object.__setattr__(self, "metric_n", MappingProxyType(metric_n))

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


@dataclass(frozen=True, slots=True)
class ControlledTemporalAblationResult:
    corruptions: tuple[TemporalCorruption, ...]
    corruption_scales: tuple[float, ...]
    seeds: tuple[int, ...]
    method_ids: tuple[str, ...]
    evaluation_sequence_ids: tuple[str, ...]
    records: tuple[TemporalAblationRecord, ...]
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        corruptions = tuple(
            TemporalCorruption(value) for value in self.corruptions
        )
        scales = tuple(
            _strict_nonnegative_real(value, name="corruption scale")
            for value in self.corruption_scales
        )
        seeds = tuple(_strict_int(value, name="seed") for value in self.seeds)
        method_ids = tuple(self.method_ids)
        sequence_ids = tuple(self.evaluation_sequence_ids)
        records = tuple(self.records)
        for name, values in (
            ("corruptions", corruptions),
            ("corruption_scales", scales),
            ("seeds", seeds),
            ("method_ids", method_ids),
            ("evaluation_sequence_ids", sequence_ids),
        ):
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in method_ids
        ):
            raise ValueError("method_ids must contain nonblank strings")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in sequence_ids
        ):
            raise ValueError(
                "evaluation_sequence_ids must contain nonblank strings"
            )

        expected = {
            (kind, scale, seed, method_id, sequence_id)
            for kind in corruptions
            for scale in scales
            for seed in seeds
            for method_id in method_ids
            for sequence_id in sequence_ids
        }
        seen: set[tuple[TemporalCorruption, float, int, str, str]] = set()
        regimes: dict[str, FitRegime] = {}
        scopes: dict[str, EvaluationScope] = {}
        for record in records:
            key = (
                record.corruption,
                record.corruption_scale,
                record.seed,
                record.method_id,
                record.sequence_id,
            )
            if key in seen:
                raise ValueError(
                    f"duplicate temporal ablation record {key!r}"
                )
            seen.add(key)
            regime = regimes.setdefault(record.method_id, record.fit_regime)
            if regime is not record.fit_regime:
                raise ValueError("fit regime changed across ablation points")
            scope = scopes.setdefault(
                record.method_id,
                record.evaluation_scope,
            )
            if scope is not record.evaluation_scope:
                raise ValueError(
                    "evaluation scope changed across ablation points"
                )
        missing = expected - seen
        extra = seen - expected
        if missing or extra:
            raise ValueError(
                "temporal ablation result must contain the exact declared "
                "corruption × scale × seed × method × sequence grid; "
                f"missing={sorted(missing, key=repr)!r}, "
                f"extra={sorted(extra, key=repr)!r}"
            )

        object.__setattr__(self, "corruptions", corruptions)
        object.__setattr__(self, "corruption_scales", scales)
        object.__setattr__(self, "seeds", seeds)
        object.__setattr__(self, "method_ids", method_ids)
        object.__setattr__(self, "evaluation_sequence_ids", sequence_ids)
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    def records_for(
        self,
        method_id: str,
        corruption: TemporalCorruption | str,
        corruption_scale: float,
    ) -> tuple[TemporalAblationRecord, ...]:
        kind = TemporalCorruption(corruption)
        scale = _strict_nonnegative_real(
            corruption_scale,
            name="corruption_scale",
        )
        if method_id not in self.method_ids:
            raise KeyError(method_id)
        if kind not in self.corruptions:
            raise KeyError(kind)
        if scale not in self.corruption_scales:
            raise KeyError(scale)
        return tuple(
            record
            for record in self.records
            if record.method_id == method_id
            and record.corruption is kind
            and record.corruption_scale == scale
        )

    def summary(
        self,
        method_id: str,
        corruption: TemporalCorruption | str,
        corruption_scale: float,
    ) -> TemporalAblationSummary:
        records = self.records_for(method_id, corruption, corruption_scale)
        counts = {status: 0 for status in CaseStatus}
        values: dict[str, list[float]] = {}
        metric_schema: tuple[str, ...] | None = None
        for record in records:
            counts[record.status] += 1
            if record.status is not CaseStatus.OK:
                continue
            schema = tuple(sorted(record.metrics))
            if metric_schema is None:
                metric_schema = schema
            elif schema != metric_schema:
                raise ValueError(
                    "successful ablation records must expose identical "
                    "metric schemas"
                )
            for key, value in record.metrics.items():
                if value is not None:
                    values.setdefault(key, []).append(value)

        means: dict[str, float | None] = {}
        stds: dict[str, float | None] = {}
        sems: dict[str, float | None] = {}
        metric_n: dict[str, int] = {}
        for key in metric_schema or ():
            samples = np.asarray(values.get(key, ()), dtype=float)
            metric_n[key] = int(samples.size)
            if samples.size == 0:
                means[key] = stds[key] = sems[key] = None
            elif samples.size == 1:
                means[key] = float(samples[0])
                stds[key] = sems[key] = None
            else:
                means[key] = float(np.mean(samples))
                std = float(np.std(samples, ddof=1))
                stds[key] = std
                sems[key] = float(std / np.sqrt(samples.size))

        return TemporalAblationSummary(
            method_id=method_id,
            corruption=records[0].corruption,
            corruption_scale=records[0].corruption_scale,
            fit_regime=records[0].fit_regime,
            evaluation_scope=records[0].evaluation_scope,
            total_cases=len(records),
            ok_cases=counts[CaseStatus.OK],
            failed_cases=counts[CaseStatus.FAILED],
            unavailable_cases=counts[CaseStatus.UNAVAILABLE],
            nonconverged_cases=counts[CaseStatus.NONCONVERGED],
            metric_mean=means,
            metric_std=stds,
            metric_sem=sems,
            metric_n=metric_n,
        )

    def summaries(self) -> tuple[TemporalAblationSummary, ...]:
        return tuple(
            self.summary(method_id, kind, scale)
            for method_id in self.method_ids
            for kind in self.corruptions
            for scale in self.corruption_scales
        )


def run_controlled_temporal_ablation(
    method_factory: Callable[[], Iterable[RepresentationMethod]],
    *,
    corruptions: Iterable[TemporalCorruption | str],
    corruption_scales: Iterable[float],
    seeds: Iterable[int],
    neighborhood_k: int = 5,
) -> ControlledTemporalAblationResult:
    """Run a deterministic factorial temporal-corruption pilot."""

    kinds = tuple(TemporalCorruption(value) for value in corruptions)
    scales = tuple(
        _strict_nonnegative_real(value, name="corruption scale")
        for value in corruption_scales
    )
    seed_values = tuple(_strict_int(value, name="seed") for value in seeds)
    for name, values in (
        ("corruptions", kinds),
        ("corruption_scales", scales),
        ("seeds", seed_values),
    ):
        if not values or len(set(values)) != len(values):
            raise ValueError(f"{name} must be nonempty and unique")

    records: list[TemporalAblationRecord] = []
    declared_method_ids: tuple[str, ...] | None = None
    declared_regimes: dict[str, FitRegime] | None = None
    declared_scopes: dict[str, EvaluationScope] | None = None
    evaluation_ids: tuple[str, ...] | None = None

    for kind in kinds:
        for scale in scales:
            for seed in seed_values:
                data = make_controlled_corruption_manifold(
                    corruption=kind,
                    corruption_scale=scale,
                    seed=seed,
                )
                methods = tuple(method_factory())
                if not methods:
                    raise ValueError(
                        "method_factory must return at least one method"
                    )
                method_ids = tuple(method.method_id for method in methods)
                regimes = {
                    method.method_id: FitRegime(method.fit_regime)
                    for method in methods
                }
                scopes = {
                    method.method_id: EvaluationScope(method.evaluation_scope)
                    for method in methods
                }
                if declared_method_ids is None:
                    declared_method_ids = method_ids
                    declared_regimes = regimes
                    declared_scopes = scopes
                elif (
                    method_ids != declared_method_ids
                    or regimes != declared_regimes
                    or scopes != declared_scopes
                ):
                    raise ValueError(
                        "method_factory must return the same ordered method IDs, "
                        "fit regimes, and evaluation scopes at every ablation point"
                    )

                if evaluation_ids is None:
                    evaluation_ids = data.evaluation.sequence_ids
                elif data.evaluation.sequence_ids != evaluation_ids:
                    raise ValueError(
                        "controlled corruption generator changed evaluation identity"
                    )

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
                        TemporalAblationRecord(
                            corruption=kind,
                            corruption_scale=scale,
                            seed=seed,
                            method_id=case.method_id,
                            sequence_id=case.sequence_id,
                            fit_regime=case.fit_regime,
                            evaluation_scope=case.evaluation_scope,
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
    return ControlledTemporalAblationResult(
        corruptions=kinds,
        corruption_scales=scales,
        seeds=seed_values,
        method_ids=declared_method_ids,
        evaluation_sequence_ids=evaluation_ids,
        records=tuple(records),
        metadata={
            "schema": (
                "neuros.representation.controlled_temporal_ablation.v1"
            ),
            "ranking_policy": "none",
            "claim_scope": "representation_geometry_temporal_ablation",
            "uncertainty_unit": "independent_controlled_seed",
            "generator": "controlled_temporal_corruption.v1",
        },
    )
