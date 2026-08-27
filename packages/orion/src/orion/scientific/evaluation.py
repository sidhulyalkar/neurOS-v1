"""Metric, repeated-measures, and failure-preservation authority."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .common import (
    CaseStatus,
    FailureAggregationPolicy,
    MetricDirection,
    ProbabilityRequirement,
    canonical_sha256,
    display_fingerprint,
    freeze_json,
    nonempty,
    strings,
    thaw_json,
)


@dataclass(frozen=True, slots=True)
class MetricSpec:
    metric_id: str
    version: str
    direction: MetricDirection
    averaging: str
    class_semantics: str
    probability_requirement: ProbabilityRequirement
    estimator: str
    estimator_version: str
    aggregation_unit: str
    failure_policy: FailureAggregationPolicy
    uncertainty_method: str
    primary: bool = False
    positive_class: str | None = None
    target_value: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("MetricSpec schema_version must be 2")
        if not isinstance(self.direction, MetricDirection):
            raise TypeError("direction must be MetricDirection")
        if not isinstance(self.probability_requirement, ProbabilityRequirement):
            raise TypeError("probability_requirement must be ProbabilityRequirement")
        if not isinstance(self.failure_policy, FailureAggregationPolicy):
            raise TypeError("failure_policy must be FailureAggregationPolicy")
        for name in (
            "metric_id",
            "version",
            "averaging",
            "class_semantics",
            "estimator",
            "estimator_version",
            "aggregation_unit",
            "uncertainty_method",
        ):
            object.__setattr__(self, name, nonempty(name, getattr(self, name)))
        if not isinstance(self.primary, bool):
            raise ValueError("primary must be boolean")
        if self.positive_class is not None:
            object.__setattr__(self, "positive_class", nonempty("positive_class", self.positive_class))
        if self.direction is MetricDirection.TARGET_IS_BEST:
            if (
                self.target_value is None
                or isinstance(self.target_value, bool)
                or not isinstance(self.target_value, (int, float, np.number))
                or not math.isfinite(float(self.target_value))
            ):
                raise ValueError("target_is_best metrics require a finite numeric target_value")
            object.__setattr__(self, "target_value", float(self.target_value))
        elif self.target_value is not None:
            raise ValueError("target_value is only valid for target_is_best metrics")
        if self.probability_requirement is not ProbabilityRequirement.NONE:
            semantics = self.class_semantics.lower()
            if "prob" not in semantics and "calibr" not in semantics:
                # This does not attempt to infer correctness. It merely forces the
                # probability requirement to be visible in metric semantics.
                raise ValueError(
                    "probability-requiring metrics must describe probability/calibration semantics"
                )
        metadata = freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def metric_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.metric_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "metric_id": self.metric_id,
            "version": self.version,
            "direction": self.direction.value,
            "averaging": self.averaging,
            "class_semantics": self.class_semantics,
            "positive_class": self.positive_class,
            "probability_requirement": self.probability_requirement.value,
            "estimator": self.estimator,
            "estimator_version": self.estimator_version,
            "aggregation_unit": self.aggregation_unit,
            "failure_policy": self.failure_policy.value,
            "uncertainty_method": self.uncertainty_method,
            "primary": self.primary,
            "target_value": self.target_value,
            "metadata": thaw_json(self.metadata),
        }
        if include_identity:
            payload["metric_sha256"] = self.metric_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class RepeatedMeasuresAuthority:
    hierarchy: tuple[str, ...]
    independent_unit: str
    case_unit: str
    cluster_units: tuple[str, ...]
    inference_method: str
    strata: tuple[str, ...] = ()
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("RepeatedMeasuresAuthority schema_version must be 2")
        hierarchy = strings("hierarchy", self.hierarchy, allow_empty=False)
        independent = nonempty("independent_unit", self.independent_unit)
        case_unit = nonempty("case_unit", self.case_unit)
        clusters = strings("cluster_units", self.cluster_units, allow_empty=False)
        strata = strings("strata", self.strata)
        if independent not in hierarchy:
            raise ValueError("independent_unit must be present in hierarchy")
        if any(unit not in hierarchy for unit in clusters):
            raise ValueError("every cluster unit must be present in hierarchy")
        if independent not in clusters:
            raise ValueError(
                "cluster_units must include the declared independent experimental unit"
            )
        object.__setattr__(self, "hierarchy", hierarchy)
        object.__setattr__(self, "independent_unit", independent)
        object.__setattr__(self, "case_unit", case_unit)
        object.__setattr__(self, "cluster_units", clusters)
        object.__setattr__(self, "inference_method", nonempty("inference_method", self.inference_method))
        object.__setattr__(self, "strata", strata)

    @property
    def authority_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.authority_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "hierarchy": list(self.hierarchy),
            "independent_unit": self.independent_unit,
            "case_unit": self.case_unit,
            "cluster_units": list(self.cluster_units),
            "inference_method": self.inference_method,
            "strata": list(self.strata),
        }
        if include_identity:
            payload["authority_sha256"] = self.authority_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class CaseOutcome:
    case_id: str
    method_id: str
    status: CaseStatus
    metrics: Mapping[str, float] = field(default_factory=dict)
    reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", nonempty("case_id", self.case_id))
        object.__setattr__(self, "method_id", nonempty("method_id", self.method_id))
        if not isinstance(self.status, CaseStatus):
            raise TypeError("status must be CaseStatus")
        if not isinstance(self.metrics, Mapping):
            raise TypeError("metrics must be a mapping")
        metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            name = nonempty("metric name", key)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"metric {name!r} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"metric {name!r} must be finite")
            metrics[name] = number
        if self.status is CaseStatus.OK:
            if self.reason is not None:
                raise ValueError("successful rows cannot carry a failure reason")
            if not metrics:
                raise ValueError("successful rows require at least one metric")
        else:
            if metrics:
                raise ValueError(
                    "non-success rows cannot carry scientific metric values; "
                    "store partial diagnostics in metadata"
                )
            object.__setattr__(self, "reason", nonempty("reason", self.reason or ""))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        metadata = freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "method_id": self.method_id,
            "status": self.status.value,
            "metrics": dict(self.metrics),
            "reason": self.reason,
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FailurePreservingResultSet:
    """Complete method x case matrix where difficult cases cannot disappear."""

    declared_case_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    rows: tuple[CaseOutcome, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("FailurePreservingResultSet schema_version must be 2")
        cases = strings("declared_case_ids", self.declared_case_ids, allow_empty=False)
        methods = strings("method_ids", self.method_ids, allow_empty=False)
        rows = tuple(self.rows)
        if any(not isinstance(row, CaseOutcome) for row in rows):
            raise TypeError("rows must contain only CaseOutcome objects")
        expected = {(method, case) for method in methods for case in cases}
        actual = {(row.method_id, row.case_id) for row in rows}
        if len(actual) != len(rows):
            raise ValueError("result rows cannot duplicate a method/case pair")
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                "result set must preserve the complete declared method/case matrix; "
                f"missing={missing[:8]}, extra={extra[:8]}"
            )
        object.__setattr__(self, "declared_case_ids", cases)
        object.__setattr__(self, "method_ids", methods)
        object.__setattr__(self, "rows", rows)

    @property
    def result_sha256(self) -> str:
        return canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return display_fingerprint(self.result_sha256)

    def status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in CaseStatus}
        for row in self.rows:
            counts[row.status.value] += 1
        return counts

    def require_metric_specs(self, specs: tuple[MetricSpec, ...]) -> None:
        specs_tuple = tuple(specs)
        if any(not isinstance(spec, MetricSpec) for spec in specs_tuple):
            raise TypeError("metric specifications must contain only MetricSpec objects")
        declared = {spec.metric_id for spec in specs_tuple}
        if not declared:
            raise ValueError("at least one metric spec is required")
        for row in self.rows:
            unknown = set(row.metrics) - declared
            if unknown:
                raise ValueError(
                    f"row {row.method_id}/{row.case_id} reports undeclared metrics {sorted(unknown)}"
                )
            if row.status is CaseStatus.OK and not set(row.metrics).issuperset(declared):
                missing = sorted(declared - set(row.metrics))
                raise ValueError(
                    f"successful row {row.method_id}/{row.case_id} is missing declared metrics {missing}"
                )

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "declared_case_ids": list(self.declared_case_ids),
            "method_ids": list(self.method_ids),
            "rows": [row.to_dict() for row in self.rows],
            "status_counts": self.status_counts(),
        }
        if include_identity:
            payload["result_sha256"] = self.result_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload
