"""Claim-aware hierarchical replication and uncertainty for mechanistic evidence."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from neuros_mechint.core.manifest import stable_hash

REPLICATION_ARTIFACT_SCHEMA = "neuros-mechint.hierarchical-replication-artifact.v1"
REPLICATION_STUDY_SCHEMA = "neuros-mechint.hierarchical-replication-study.v1"


class ReplicationAxis(str, Enum):
    """Scientific hierarchy axes supported by v0.9."""

    DATASET = "dataset"
    MODEL_SEED = "model_seed"
    CHECKPOINT = "checkpoint"
    DICTIONARY = "dictionary"
    PROJECTOR = "projector"
    SUBJECT = "subject"
    SESSION = "session"
    TRIAL = "trial"


DEFAULT_HIERARCHY = (
    ReplicationAxis.DATASET,
    ReplicationAxis.MODEL_SEED,
    ReplicationAxis.CHECKPOINT,
    ReplicationAxis.DICTIONARY,
    ReplicationAxis.PROJECTOR,
    ReplicationAxis.SUBJECT,
    ReplicationAxis.SESSION,
    ReplicationAxis.TRIAL,
)


@dataclass(frozen=True, slots=True)
class ReplicationCoordinates:
    """Coordinates locating one observation in the scientific hierarchy."""

    dataset_id: str | None = None
    model_seed: str | int | None = None
    checkpoint: str | None = None
    dictionary_id: str | None = None
    projector_id: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    trial_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def value(self, axis: ReplicationAxis | str) -> str | None:
        axis = ReplicationAxis(axis)
        raw: Any
        if axis is ReplicationAxis.DATASET:
            raw = self.dataset_id
        elif axis is ReplicationAxis.MODEL_SEED:
            raw = self.model_seed
        elif axis is ReplicationAxis.CHECKPOINT:
            raw = self.checkpoint
        elif axis is ReplicationAxis.DICTIONARY:
            raw = self.dictionary_id
        elif axis is ReplicationAxis.PROJECTOR:
            raw = self.projector_id
        elif axis is ReplicationAxis.SUBJECT:
            raw = self.subject_id
        elif axis is ReplicationAxis.SESSION:
            raw = self.session_id
        else:
            raw = self.trial_id
        return None if raw is None else str(raw)

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint": self.checkpoint,
            "dataset_id": self.dataset_id,
            "dictionary_id": self.dictionary_id,
            "metadata": dict(self.metadata),
            "model_seed": None if self.model_seed is None else str(self.model_seed),
            "projector_id": self.projector_id,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "trial_id": self.trial_id,
        }


@dataclass(frozen=True, slots=True)
class ReplicationObservation:
    """One observed mechanistic result with explicit hierarchical coordinates."""

    observation_id: str
    family_id: str
    coordinates: ReplicationCoordinates
    metrics: Mapping[str, float]
    estimable: bool = True
    rejection_reasons: tuple[str, ...] = ()
    source_study_fingerprint: str | None = None
    source_kind: str = "generic"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.observation_id or not self.family_id:
            raise ValueError("observation_id and family_id must be non-empty")
        metrics = {str(key): float(value) for key, value in self.metrics.items()}
        if not metrics:
            raise ValueError("replication observation requires at least one metric")
        if not np.isfinite(np.asarray(list(metrics.values()), dtype=np.float64)).all():
            raise ValueError("replication metrics must be finite")
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        object.__setattr__(
            self,
            "rejection_reasons",
            tuple(str(item) for item in self.rejection_reasons),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": self.coordinates.to_dict(),
            "estimable": self.estimable,
            "family_id": self.family_id,
            "metadata": dict(self.metadata),
            "metrics": dict(self.metrics),
            "observation_id": self.observation_id,
            "rejection_reasons": list(self.rejection_reasons),
            "source_kind": self.source_kind,
            "source_study_fingerprint": self.source_study_fingerprint,
        }


@dataclass(frozen=True, slots=True)
class HierarchicalReplicationPolicy:
    """Preregistered estimability and replication thresholds."""

    min_independent_units: int = 3
    bootstrap_samples: int = 2000
    confidence_level: float = 0.95
    min_sign_agreement: float = 0.75
    min_estimable_fraction: float = 0.80
    require_ci_excludes_null: bool = True
    min_absolute_effect: float = 0.0

    def __post_init__(self) -> None:
        if self.min_independent_units < 2:
            raise ValueError("min_independent_units must be at least 2")
        if self.bootstrap_samples < 100:
            raise ValueError("bootstrap_samples must be at least 100")
        for name in ("confidence_level", "min_sign_agreement", "min_estimable_fraction"):
            value = float(getattr(self, name))
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        if self.confidence_level >= 1.0:
            raise ValueError("confidence_level must be less than 1")
        if self.min_absolute_effect < 0.0 or not np.isfinite(self.min_absolute_effect):
            raise ValueError("min_absolute_effect must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class HierarchicalReplicationSpec:
    """Frozen design for one claim-aware replication analysis."""

    study_id: str
    family_id: str
    claim_axis: ReplicationAxis | str
    primary_metric: str
    hierarchy: tuple[ReplicationAxis | str, ...] = DEFAULT_HIERARCHY
    null_value: float = 0.0
    expected_direction: int = 0
    seed: int = 0
    policy: HierarchicalReplicationPolicy = field(default_factory=HierarchicalReplicationPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = REPLICATION_STUDY_SCHEMA

    def __post_init__(self) -> None:
        if not self.study_id or not self.family_id or not self.primary_metric:
            raise ValueError("study_id, family_id, and primary_metric must be non-empty")
        claim_axis = ReplicationAxis(self.claim_axis)
        hierarchy = tuple(ReplicationAxis(item) for item in self.hierarchy)
        if len(hierarchy) != len(set(hierarchy)):
            raise ValueError("hierarchy axes must be unique")
        if claim_axis not in hierarchy:
            raise ValueError("claim_axis must be present in hierarchy")
        if self.expected_direction not in {-1, 0, 1}:
            raise ValueError("expected_direction must be -1, 0, or 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if not np.isfinite(float(self.null_value)):
            raise ValueError("null_value must be finite")
        object.__setattr__(self, "claim_axis", claim_axis)
        object.__setattr__(self, "hierarchy", hierarchy)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def active_hierarchy(self) -> tuple[ReplicationAxis, ...]:
        start = self.hierarchy.index(self.claim_axis)
        return self.hierarchy[start:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_axis": self.claim_axis.value,
            "expected_direction": self.expected_direction,
            "family_id": self.family_id,
            "hierarchy": [item.value for item in self.hierarchy],
            "metadata": dict(self.metadata),
            "null_value": self.null_value,
            "policy": self.policy.to_dict(),
            "primary_metric": self.primary_metric,
            "schema_version": self.schema_version,
            "seed": self.seed,
            "study_id": self.study_id,
        }


@dataclass(frozen=True, slots=True)
class IndependentUnitSummary:
    """Balanced estimate for one independent unit at the claim axis."""

    unit_id: str
    observation_count: int
    estimable_count: int
    metrics: Mapping[str, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "metrics",
            MappingProxyType({str(k): float(v) for k, v in self.metrics.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimable_count": self.estimable_count,
            "metrics": dict(self.metrics),
            "observation_count": self.observation_count,
            "unit_id": self.unit_id,
        }


@dataclass(frozen=True, slots=True)
class MetricReplicationEstimate:
    """Uncertainty summary for one metric at the declared claim level."""

    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    null_value: float
    independent_units: int
    between_unit_std: float
    sign_agreement: float
    bootstrap_samples: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReplicationDecision:
    estimable: bool
    replicated: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "estimable": self.estimable,
            "reasons": list(self.reasons),
            "replicated": self.replicated,
        }


@dataclass(frozen=True, slots=True)
class HierarchicalReplicationResult:
    """Complete v0.9 claim-aware replication result."""

    spec: HierarchicalReplicationSpec
    observations: tuple[ReplicationObservation, ...]
    independent_units: tuple[IndependentUnitSummary, ...]
    metric_estimates: tuple[MetricReplicationEstimate, ...]
    decision: ReplicationDecision

    @property
    def primary_estimate(self) -> MetricReplicationEstimate | None:
        return next(
            (item for item in self.metric_estimates if item.metric == self.spec.primary_metric),
            None,
        )

    @property
    def study_fingerprint(self) -> str:
        return stable_hash(
            {
                "decision": self.decision.to_dict(),
                "independent_units": [item.to_dict() for item in self.independent_units],
                "metric_estimates": [item.to_dict() for item in self.metric_estimates],
                "observations": [item.to_dict() for item in self.observations],
                "spec": self.spec.to_dict(),
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.to_dict(),
            "independent_units": [item.to_dict() for item in self.independent_units],
            "metric_estimates": [item.to_dict() for item in self.metric_estimates],
            "observations": [item.to_dict() for item in self.observations],
            "schema_version": self.spec.schema_version,
            "spec": self.spec.to_dict(),
            "study_fingerprint": self.study_fingerprint,
        }


def _balanced_metric(
    observations: Sequence[ReplicationObservation],
    *,
    metric: str,
    hierarchy: Sequence[ReplicationAxis],
) -> float:
    if not observations:
        raise ValueError("cannot aggregate an empty observation set")
    if not hierarchy:
        return float(np.mean([item.metrics[metric] for item in observations]))
    axis = hierarchy[0]
    groups: dict[str, list[ReplicationObservation]] = defaultdict(list)
    for observation in observations:
        value = observation.coordinates.value(axis)
        if value is None:
            raise ValueError(f"missing {axis.value} coordinate during hierarchical aggregation")
        groups[value].append(observation)
    child_values = [
        _balanced_metric(group, metric=metric, hierarchy=hierarchy[1:])
        for _, group in sorted(groups.items())
    ]
    return float(np.mean(child_values))


def _has_complete_hierarchy(
    observation: ReplicationObservation,
    hierarchy: Sequence[ReplicationAxis],
) -> bool:
    return all(observation.coordinates.value(axis) is not None for axis in hierarchy)


def _unit_summaries(
    observations: Sequence[ReplicationObservation],
    *,
    spec: HierarchicalReplicationSpec,
    metrics: Sequence[str],
) -> tuple[IndependentUnitSummary, ...]:
    groups: dict[str, list[ReplicationObservation]] = defaultdict(list)
    for observation in observations:
        unit_id = observation.coordinates.value(spec.claim_axis)
        if unit_id is None:
            continue
        groups[unit_id].append(observation)
    lower_hierarchy = spec.active_hierarchy[1:]
    result = []
    for unit_id, group in sorted(groups.items()):
        estimable = [item for item in group if item.estimable]
        complete_estimable = [
            item for item in estimable if _has_complete_hierarchy(item, lower_hierarchy)
        ]
        metric_values = {}
        if estimable and len(complete_estimable) == len(estimable):
            for metric in metrics:
                metric_values[metric] = _balanced_metric(
                    complete_estimable,
                    metric=metric,
                    hierarchy=lower_hierarchy,
                )
        result.append(
            IndependentUnitSummary(
                unit_id=unit_id,
                observation_count=len(group),
                estimable_count=len(estimable),
                metrics=metric_values,
            )
        )
    return tuple(result)


def _hierarchical_bootstrap_once(
    observations: Sequence[ReplicationObservation],
    *,
    metric: str,
    hierarchy: Sequence[ReplicationAxis],
    rng: np.random.Generator,
) -> float:
    if not hierarchy:
        values = np.asarray([item.metrics[metric] for item in observations], dtype=np.float64)
        indices = rng.integers(0, len(values), size=len(values))
        return float(np.mean(values[indices]))
    axis = hierarchy[0]
    groups: dict[str, list[ReplicationObservation]] = defaultdict(list)
    for observation in observations:
        value = observation.coordinates.value(axis)
        if value is None:
            raise ValueError(f"missing {axis.value} coordinate during bootstrap")
        groups[value].append(observation)
    ordered = [group for _, group in sorted(groups.items())]
    sampled_indices = rng.integers(0, len(ordered), size=len(ordered))
    sampled = [
        _hierarchical_bootstrap_once(
            ordered[index],
            metric=metric,
            hierarchy=hierarchy[1:],
            rng=rng,
        )
        for index in sampled_indices
    ]
    return float(np.mean(sampled))


def _bootstrap_interval(
    observations: Sequence[ReplicationObservation],
    *,
    metric: str,
    spec: HierarchicalReplicationSpec,
    rng: np.random.Generator,
) -> tuple[float, float]:
    values = np.asarray(
        [
            _hierarchical_bootstrap_once(
                observations,
                metric=metric,
                hierarchy=spec.active_hierarchy,
                rng=rng,
            )
            for _ in range(spec.policy.bootstrap_samples)
        ],
        dtype=np.float64,
    )
    alpha = (1.0 - spec.policy.confidence_level) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def _sign_agreement(values: Sequence[float], *, null_value: float, direction: int) -> float:
    centered = np.asarray(values, dtype=np.float64) - float(null_value)
    if direction == 0:
        aggregate = float(np.mean(centered))
        direction = 1 if aggregate >= 0.0 else -1
    if direction > 0:
        return float(np.mean(centered > 0.0))
    return float(np.mean(centered < 0.0))


def analyze_hierarchical_replication(
    spec: HierarchicalReplicationSpec,
    observations: Sequence[ReplicationObservation],
) -> HierarchicalReplicationResult:
    """Estimate a claim at its declared independent replication level."""

    observations = tuple(observations)
    reasons: list[str] = []
    if not observations:
        reasons.append("no observations were supplied")
    families = {item.family_id for item in observations}
    if families and families != {spec.family_id}:
        reasons.append("observations contain a replication family other than spec.family_id")
    if any(spec.primary_metric not in item.metrics for item in observations):
        reasons.append(f"primary metric {spec.primary_metric!r} is missing from one or more observations")

    for axis in spec.active_hierarchy:
        missing = [
            item.observation_id
            for item in observations
            if item.coordinates.value(axis) is None
        ]
        if missing:
            reasons.append(f"axis {axis.value!r} is missing for {len(missing)} observation(s)")

    all_metrics = (
        sorted(set.intersection(*(set(item.metrics) for item in observations)))
        if observations
        else []
    )
    valid = [item for item in observations if item.estimable]
    estimable_fraction = 0.0 if not observations else len(valid) / len(observations)
    if estimable_fraction < spec.policy.min_estimable_fraction:
        reasons.append(
            "estimable observation fraction "
            f"{estimable_fraction:.3f} is below {spec.policy.min_estimable_fraction:.3f}"
        )

    independent_ids = {
        item.coordinates.value(spec.claim_axis)
        for item in valid
        if item.coordinates.value(spec.claim_axis) is not None
    }
    if len(independent_ids) < spec.policy.min_independent_units:
        reasons.append(
            f"claim axis {spec.claim_axis.value!r} has {len(independent_ids)} independent unit(s); "
            f"requires {spec.policy.min_independent_units}"
        )

    scientific_estimable = not reasons
    unit_summaries = _unit_summaries(observations, spec=spec, metrics=all_metrics)
    estimates: list[MetricReplicationEstimate] = []
    if scientific_estimable:
        rng = np.random.default_rng(spec.seed)
        for metric in all_metrics:
            estimate = _balanced_metric(valid, metric=metric, hierarchy=spec.active_hierarchy)
            ci_low, ci_high = _bootstrap_interval(valid, metric=metric, spec=spec, rng=rng)
            unit_values = [
                item.metrics[metric]
                for item in unit_summaries
                if item.estimable_count > 0 and metric in item.metrics
            ]
            estimates.append(
                MetricReplicationEstimate(
                    metric=metric,
                    estimate=estimate,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    null_value=spec.null_value,
                    independent_units=len(unit_values),
                    between_unit_std=(
                        0.0 if len(unit_values) < 2 else float(np.std(unit_values, ddof=1))
                    ),
                    sign_agreement=_sign_agreement(
                        unit_values,
                        null_value=spec.null_value,
                        direction=spec.expected_direction,
                    ),
                    bootstrap_samples=spec.policy.bootstrap_samples,
                )
            )

    replicated = False
    if scientific_estimable:
        primary = next(item for item in estimates if item.metric == spec.primary_metric)
        decision_reasons: list[str] = []
        centered = primary.estimate - spec.null_value
        if abs(centered) < spec.policy.min_absolute_effect:
            decision_reasons.append(
                f"absolute primary effect {abs(centered):.6g} is below "
                f"{spec.policy.min_absolute_effect:.6g}"
            )
        if primary.sign_agreement < spec.policy.min_sign_agreement:
            decision_reasons.append(
                f"independent-unit sign agreement {primary.sign_agreement:.3f} is below "
                f"{spec.policy.min_sign_agreement:.3f}"
            )
        wrong_direction = (
            spec.expected_direction > 0
            and centered <= 0.0
            or spec.expected_direction < 0
            and centered >= 0.0
        )
        if wrong_direction:
            decision_reasons.append("primary estimate has the wrong direction")
        if (
            spec.policy.require_ci_excludes_null
            and primary.ci_low <= spec.null_value <= primary.ci_high
        ):
            decision_reasons.append("confidence interval includes the preregistered null")
        reasons.extend(decision_reasons)
        replicated = not decision_reasons

    return HierarchicalReplicationResult(
        spec=spec,
        observations=observations,
        independent_units=unit_summaries,
        metric_estimates=tuple(estimates),
        decision=ReplicationDecision(
            estimable=scientific_estimable,
            replicated=replicated,
            reasons=tuple(reasons),
        ),
    )


def observation_from_correspondence(
    result: Any,
    *,
    observation_id: str,
    family_id: str,
    coordinates: ReplicationCoordinates,
    metadata: Mapping[str, Any] | None = None,
) -> ReplicationObservation:
    """Convert one v0.8 correspondence result into a v0.9 replication observation."""

    metrics = result.validation_metrics
    return ReplicationObservation(
        observation_id=observation_id,
        family_id=family_id,
        coordinates=coordinates,
        metrics={
            "causal_recovery": metrics.median_causal_recovery,
            "causal_score": metrics.median_causal_score,
            "predictive_r2": metrics.predictive_r2,
            "random_margin": metrics.random_margin,
            "shuffled_margin": metrics.shuffled_margin,
            "source_effect": metrics.median_source_effect,
            "target_effect": metrics.median_target_effect,
        },
        estimable=True,
        rejection_reasons=(),
        source_study_fingerprint=result.study_fingerprint,
        source_kind="feature_correspondence",
        metadata={"promotion_passed": result.promotion.passed, **dict(metadata or {})},
    )


def observation_from_factorial_contrast(
    report: Any,
    *,
    contrast_id: str,
    observation_id: str,
    family_id: str,
    coordinates: ReplicationCoordinates,
    metadata: Mapping[str, Any] | None = None,
) -> ReplicationObservation:
    """Convert one v0.7 factorial contrast into a v0.9 replication observation."""

    contrast = next((item for item in report.contrasts if item.contrast_id == contrast_id), None)
    if contrast is None:
        raise ValueError(f"factorial report does not contain contrast {contrast_id!r}")
    metrics = dict(contrast.outcome_effects) or {"nonestimable_placeholder": 0.0}
    return ReplicationObservation(
        observation_id=observation_id,
        family_id=family_id,
        coordinates=coordinates,
        metrics=metrics,
        estimable=contrast.estimable,
        rejection_reasons=contrast.reasons,
        source_study_fingerprint=report.study_fingerprint,
        source_kind="factorial_contrast",
        metadata={"contrast_id": contrast_id, **dict(metadata or {})},
    )


def write_replication_artifact(
    result: HierarchicalReplicationResult,
    path: str | Path,
) -> Path:
    """Write a self-checking v0.9 hierarchical replication artifact."""

    path = Path(path)
    payload = result.to_dict()
    envelope = {
        "artifact_schema": REPLICATION_ARTIFACT_SCHEMA,
        "integrity_hash": stable_hash(payload),
        "result": payload,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(envelope, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_replication_artifact(path: str | Path) -> Mapping[str, Any]:
    """Read and verify a v0.9 hierarchical replication artifact."""

    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(envelope, dict):
        raise ValueError("replication artifact must contain a JSON object")
    if envelope.get("artifact_schema") != REPLICATION_ARTIFACT_SCHEMA:
        raise ValueError("unsupported replication artifact schema")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise ValueError("replication artifact result must be an object")
    if envelope.get("integrity_hash") != stable_hash(result):
        raise ValueError("replication artifact integrity hash mismatch")
    return MappingProxyType(result)
