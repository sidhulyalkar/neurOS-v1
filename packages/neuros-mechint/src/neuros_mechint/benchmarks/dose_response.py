"""Controlled intervention dose-response and manifold-assumption contracts."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any

import numpy as np


class InterventionManifoldKind(str, Enum):
    """How replacement values are expected to relate to the observed activation manifold."""

    ZERO = "zero"
    MEAN = "mean"
    EMPIRICAL_DONOR = "empirical_donor"
    NEAREST_NEIGHBOR = "nearest_neighbor"
    QUANTILE_MATCHED = "quantile_matched"
    CONDITIONAL_RESAMPLE = "conditional_resample"
    GENERATIVE = "generative"
    CAUSAL_SCRUBBING = "causal_scrubbing"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class InterventionManifoldAssumption:
    """Explicit provenance for the activation-manifold assumption of an intervention."""

    kind: InterventionManifoldKind | str
    description: str
    donor_pool_id: str | None = None
    fitted_on_partition_id: str | None = None
    expected_in_manifold: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        kind = InterventionManifoldKind(self.kind)
        if not self.description:
            raise ValueError("manifold assumption description must be non-empty")
        donor_kinds = {
            InterventionManifoldKind.EMPIRICAL_DONOR,
            InterventionManifoldKind.NEAREST_NEIGHBOR,
            InterventionManifoldKind.QUANTILE_MATCHED,
            InterventionManifoldKind.CONDITIONAL_RESAMPLE,
            InterventionManifoldKind.GENERATIVE,
            InterventionManifoldKind.CAUSAL_SCRUBBING,
        }
        if kind in donor_kinds and not self.donor_pool_id:
            raise ValueError(f"{kind.value} manifold assumption requires donor_pool_id")
        fitted_kinds = {
            InterventionManifoldKind.CONDITIONAL_RESAMPLE,
            InterventionManifoldKind.GENERATIVE,
        }
        if kind in fitted_kinds and not self.fitted_on_partition_id:
            raise ValueError(f"{kind.value} manifold assumption requires fitted_on_partition_id")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "donor_pool_id": self.donor_pool_id,
            "expected_in_manifold": self.expected_in_manifold,
            "fitted_on_partition_id": self.fitted_on_partition_id,
            "kind": self.kind.value,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class DoseResponseObservation:
    """One metric observation at one intervention dose."""

    unit_id: str
    dose: float
    metric: float
    semantic_trial_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.unit_id:
            raise ValueError("dose-response unit_id must be non-empty")
        if not 0.0 <= float(self.dose) <= 1.0:
            raise ValueError("dose must lie in [0, 1]")
        if not np.isfinite(float(self.metric)):
            raise ValueError("dose-response metric must be finite")
        object.__setattr__(self, "dose", float(self.dose))
        object.__setattr__(self, "metric", float(self.metric))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "dose": self.dose,
            "metadata": dict(self.metadata),
            "metric": self.metric,
            "semantic_trial_id": self.semantic_trial_id,
            "unit_id": self.unit_id,
        }


@dataclass(frozen=True, slots=True)
class DoseResponsePolicy:
    min_doses: int = 5
    min_units: int = 3
    min_monotonic_fraction: float = 0.75
    min_endpoint_effect: float = 1e-6
    require_endpoints: bool = True
    require_common_grid: bool = True

    def __post_init__(self) -> None:
        if self.min_doses < 3:
            raise ValueError("min_doses must be at least 3")
        if self.min_units < 2:
            raise ValueError("min_units must be at least 2")
        if not 0.0 <= self.min_monotonic_fraction <= 1.0:
            raise ValueError("min_monotonic_fraction must lie in [0, 1]")
        if self.min_endpoint_effect < 0.0 or not np.isfinite(self.min_endpoint_effect):
            raise ValueError("min_endpoint_effect must be finite and non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class DoseResponseSpec:
    study_id: str
    intervention_id: str
    expected_direction: int
    manifold: InterventionManifoldAssumption
    policy: DoseResponsePolicy = field(default_factory=DoseResponsePolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.study_id or not self.intervention_id:
            raise ValueError("study_id and intervention_id must be non-empty")
        if self.expected_direction not in {-1, 1}:
            raise ValueError("expected_direction must be -1 or 1")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected_direction": self.expected_direction,
            "intervention_id": self.intervention_id,
            "manifold": self.manifold.to_dict(),
            "metadata": dict(self.metadata),
            "policy": self.policy.to_dict(),
            "study_id": self.study_id,
        }


@dataclass(frozen=True, slots=True)
class DoseResponseUnitSummary:
    unit_id: str
    doses: tuple[float, ...]
    metrics: tuple[float, ...]
    endpoint_effect: float
    monotonic_fraction: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "doses": list(self.doses),
            "endpoint_effect": self.endpoint_effect,
            "metrics": list(self.metrics),
            "monotonic_fraction": self.monotonic_fraction,
            "unit_id": self.unit_id,
        }


@dataclass(frozen=True, slots=True)
class DoseResponseResult:
    spec: DoseResponseSpec
    unit_summaries: tuple[DoseResponseUnitSummary, ...]
    aggregate_doses: tuple[float, ...]
    aggregate_metrics: tuple[float, ...]
    endpoint_effect: float
    mean_monotonic_fraction: float
    normalized_auc: float
    passed: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregate_doses": list(self.aggregate_doses),
            "aggregate_metrics": list(self.aggregate_metrics),
            "endpoint_effect": self.endpoint_effect,
            "mean_monotonic_fraction": self.mean_monotonic_fraction,
            "normalized_auc": self.normalized_auc,
            "passed": self.passed,
            "reasons": list(self.reasons),
            "spec": self.spec.to_dict(),
            "unit_summaries": [item.to_dict() for item in self.unit_summaries],
        }


def _monotonic_fraction(values: Sequence[float], direction: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        return 0.0
    differences = np.diff(values) * float(direction)
    return float(np.mean(differences >= -1e-12))


def _trapezoid(values: np.ndarray, x: Sequence[float]) -> float:
    """Integrate with the trapezoid rule while retaining NumPy 1.24 support."""

    y = np.asarray(values, dtype=np.float64)
    coordinates = np.asarray(x, dtype=np.float64)
    if y.size != coordinates.size:
        raise ValueError("trapezoid values and coordinates must have equal length")
    if y.size < 2:
        return 0.0
    widths = np.diff(coordinates)
    return float(np.sum(0.5 * (y[:-1] + y[1:]) * widths))


def analyze_dose_response(
    spec: DoseResponseSpec,
    observations: Sequence[DoseResponseObservation],
) -> DoseResponseResult:
    """Analyze a preregistered dose-response without pooling repeated units as replicas."""

    groups: dict[str, list[DoseResponseObservation]] = defaultdict(list)
    for observation in observations:
        groups[observation.unit_id].append(observation)
    reasons: list[str] = []
    if len(groups) < spec.policy.min_units:
        reasons.append(
            f"dose-response has {len(groups)} independent unit(s); requires {spec.policy.min_units}"
        )

    summaries: list[DoseResponseUnitSummary] = []
    grids: list[tuple[float, ...]] = []
    for unit_id, group in sorted(groups.items()):
        by_dose: dict[float, list[float]] = defaultdict(list)
        for observation in group:
            by_dose[observation.dose].append(observation.metric)
        doses = tuple(sorted(by_dose))
        metrics = tuple(float(np.mean(by_dose[dose])) for dose in doses)
        if len(doses) < spec.policy.min_doses:
            reasons.append(
                f"unit {unit_id!r} has {len(doses)} unique doses; requires {spec.policy.min_doses}"
            )
        if spec.policy.require_endpoints and (not doses or doses[0] != 0.0 or doses[-1] != 1.0):
            reasons.append(f"unit {unit_id!r} is missing dose 0 or dose 1")
        endpoint = 0.0 if len(metrics) < 2 else (metrics[-1] - metrics[0]) * spec.expected_direction
        summaries.append(
            DoseResponseUnitSummary(
                unit_id=unit_id,
                doses=doses,
                metrics=metrics,
                endpoint_effect=float(endpoint),
                monotonic_fraction=_monotonic_fraction(metrics, spec.expected_direction),
            )
        )
        grids.append(doses)

    if spec.policy.require_common_grid and grids and len(set(grids)) != 1:
        reasons.append("independent dose-response units do not share a common dose grid")

    aggregate_doses: tuple[float, ...] = ()
    aggregate_metrics: tuple[float, ...] = ()
    endpoint_effect = 0.0
    monotonicity = 0.0
    normalized_auc = 0.0
    common_grid = summaries and all(item.doses == summaries[0].doses for item in summaries)
    if common_grid:
        aggregate_doses = summaries[0].doses
        matrix = np.asarray([item.metrics for item in summaries], dtype=np.float64)
        aggregate_metrics = tuple(float(value) for value in matrix.mean(axis=0))
        endpoint_effect = float(
            (aggregate_metrics[-1] - aggregate_metrics[0]) * spec.expected_direction
        )
        monotonicity = float(np.mean([item.monotonic_fraction for item in summaries]))
        baseline = aggregate_metrics[0]
        oriented = (np.asarray(aggregate_metrics) - baseline) * spec.expected_direction
        endpoint_scale = max(abs(endpoint_effect), 1e-12)
        normalized_auc = _trapezoid(oriented / endpoint_scale, aggregate_doses)

    if endpoint_effect < spec.policy.min_endpoint_effect:
        reasons.append(
            f"oriented endpoint effect {endpoint_effect:.6g} is below "
            f"{spec.policy.min_endpoint_effect:.6g}"
        )
    if monotonicity < spec.policy.min_monotonic_fraction:
        reasons.append(
            f"mean monotonic fraction {monotonicity:.3f} is below "
            f"{spec.policy.min_monotonic_fraction:.3f}"
        )

    return DoseResponseResult(
        spec=spec,
        unit_summaries=tuple(summaries),
        aggregate_doses=aggregate_doses,
        aggregate_metrics=aggregate_metrics,
        endpoint_effect=endpoint_effect,
        mean_monotonic_fraction=monotonicity,
        normalized_auc=normalized_auc,
        passed=not reasons,
        reasons=tuple(reasons),
    )
