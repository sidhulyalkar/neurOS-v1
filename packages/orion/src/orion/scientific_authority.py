"""Scientific authority for leakage-aware neural evaluation.

This module governs the information *around* model training and final assessment:
dataset/model lineage, pretraining overlap, observation roles, fitted preprocessing,
metric semantics, repeated-measures inference, failure preservation, and claim scope.

The contracts are deliberately dependency-light. Dataset-specific packages may
serialize their own split authorities and bind them here without ORION becoming
dependent on a particular benchmark library.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific authority cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("scientific authority cannot contain object arrays")
        return _canonical_json(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            normalized_key = str(key)
            if not normalized_key.strip():
                raise ValueError("scientific authority mapping keys must be non-empty")
            if normalized_key in normalized:
                raise ValueError(
                    "scientific authority mapping keys collide after string normalization: "
                    f"{normalized_key!r}"
                )
            normalized[normalized_key] = _canonical_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered sets are not valid scientific authority values")
    raise TypeError(
        "scientific authority values must be deterministic JSON-compatible primitives, "
        f"NumPy scalars/arrays, mappings, lists, or tuples; got {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    normalized = _canonical_json(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        _canonical_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return normalized


def _optional_sha256(name: str, value: str | None) -> str | None:
    return None if value is None else _require_sha256(name, value)


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _strings(name: str, values: Iterable[str], *, allow_empty: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(_nonempty(name, value) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def _display(full_sha256: str) -> str:
    return _require_sha256("full scientific identity", full_sha256)[:16]


class LineageCompleteness(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


class IdentityAvailability(str, Enum):
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class OverlapStatus(str, Enum):
    DISJOINT_VERIFIED = "disjoint_verified"
    OVERLAP_DETECTED = "overlap_detected"
    POSSIBLE_OVERLAP = "possible_overlap"
    UNKNOWN_LINEAGE = "unknown_lineage"


class ObservationRole(str, Enum):
    PRETRAINING = "pretraining"
    SUPERVISED_TRAINING = "supervised_training"
    SOURCE_HISTORY = "source_history"
    LABELED_TARGET_CALIBRATION = "labeled_target_calibration"
    UNLABELED_TARGET_OBSERVATION = "unlabeled_target_observation"
    QUALIFICATION = "qualification"
    MECHANISTIC_DISCOVERY = "mechanistic_discovery"
    FINAL_ASSESSMENT = "final_assessment"


class OperationKind(str, Enum):
    PRETRAINING = "pretraining"
    PREPROCESSING_FIT = "preprocessing_fit"
    MODEL_TRAINING = "model_training"
    ADAPTATION = "adaptation"
    MODEL_SELECTION = "model_selection"
    MECHANISTIC_DISCOVERY = "mechanistic_discovery"
    FINAL_ASSESSMENT = "final_assessment"


_ALLOWED_ROLES: Mapping[OperationKind, frozenset[ObservationRole]] = {
    OperationKind.PRETRAINING: frozenset({ObservationRole.PRETRAINING}),
    OperationKind.PREPROCESSING_FIT: frozenset(
        {
            ObservationRole.PRETRAINING,
            ObservationRole.SUPERVISED_TRAINING,
            ObservationRole.SOURCE_HISTORY,
            ObservationRole.LABELED_TARGET_CALIBRATION,
            ObservationRole.UNLABELED_TARGET_OBSERVATION,
        }
    ),
    OperationKind.MODEL_TRAINING: frozenset(
        {
            ObservationRole.PRETRAINING,
            ObservationRole.SUPERVISED_TRAINING,
            ObservationRole.SOURCE_HISTORY,
            ObservationRole.LABELED_TARGET_CALIBRATION,
            ObservationRole.UNLABELED_TARGET_OBSERVATION,
        }
    ),
    OperationKind.ADAPTATION: frozenset(
        {
            ObservationRole.SOURCE_HISTORY,
            ObservationRole.LABELED_TARGET_CALIBRATION,
            ObservationRole.UNLABELED_TARGET_OBSERVATION,
        }
    ),
    OperationKind.MODEL_SELECTION: frozenset({ObservationRole.QUALIFICATION}),
    OperationKind.MECHANISTIC_DISCOVERY: frozenset({ObservationRole.MECHANISTIC_DISCOVERY}),
    OperationKind.FINAL_ASSESSMENT: frozenset({ObservationRole.FINAL_ASSESSMENT}),
}


class TransformFitKind(str, Enum):
    PREDECLARED_FIXED = "predeclared_fixed"
    DATA_FITTED = "data_fitted"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    TARGET_IS_BEST = "target_is_best"


class ProbabilityRequirement(str, Enum):
    NONE = "none"
    PROBABILITY = "probability"
    CALIBRATED_PROBABILITY = "calibrated_probability"


class FailureAggregationPolicy(str, Enum):
    PRESERVE = "preserve"
    FAIL_STUDY = "fail_study"
    PENALIZE_PREDECLARED = "penalize_predeclared"


class CaseStatus(str, Enum):
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"
    OOM = "oom"
    NONCONVERGED = "nonconverged"
    UNAVAILABLE = "unavailable"


class EvidenceDomain(str, Enum):
    TASK_UTILITY = "task_utility"
    REPRESENTATION_GEOMETRY = "representation_geometry"
    MECHANISM = "mechanism"
    RUNTIME = "runtime"
    HARDWARE = "hardware"
    CLINICAL = "clinical"


class ClaimQualification(str, Enum):
    CLEAN = "clean"
    CONTAMINATED_PRETRAINING_OVERLAP = "contaminated_pretraining_overlap"
    UNKNOWN_PRETRAINING_LINEAGE = "unknown_pretraining_lineage"
    DESCRIPTIVE_ONLY = "descriptive_only"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class IdentitySet:
    """Identifiers available at one lineage level, or why they are unavailable."""

    level: str
    availability: IdentityAvailability
    identifiers: tuple[str, ...] = ()
    unavailable_reason: str | None = None

    def __post_init__(self) -> None:
        level = _nonempty("identity level", self.level)
        identifiers = _strings("identity identifiers", self.identifiers)
        reason = self.unavailable_reason
        if self.availability is IdentityAvailability.AVAILABLE:
            if not identifiers:
                raise ValueError("available identity sets require at least one identifier")
            if reason is not None:
                raise ValueError("available identity sets cannot carry unavailable_reason")
        elif self.availability is IdentityAvailability.UNAVAILABLE:
            if identifiers:
                raise ValueError("unavailable identity sets cannot carry identifiers")
            reason = _nonempty("unavailable_reason", reason or "")
        else:
            raise ValueError("availability must be an IdentityAvailability")
        object.__setattr__(self, "level", level)
        object.__setattr__(self, "identifiers", identifiers)
        object.__setattr__(self, "unavailable_reason", reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "availability": self.availability.value,
            "identifiers": list(self.identifiers),
            "unavailable_reason": self.unavailable_reason,
        }


@dataclass(frozen=True, slots=True)
class DatasetLineage:
    """Immutable lineage for one downstream or pretraining dataset/domain."""

    dataset_id: str
    upstream_source: str
    version: str | None = None
    revision: str | None = None
    content_sha256: str | None = None
    parent_dataset_ids: tuple[str, ...] = ()
    identity_sets: tuple[IdentitySet, ...] = ()
    preprocessing_history: tuple[str, ...] = ()
    sampling_assumptions: Mapping[str, Any] = field(default_factory=dict)
    license: str | None = None
    citation: str | None = None
    lineage_completeness: LineageCompleteness = LineageCompleteness.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("DatasetLineage schema_version must be 2")
        dataset_id = _nonempty("dataset_id", self.dataset_id)
        upstream_source = _nonempty("upstream_source", self.upstream_source)
        parents = _strings("parent_dataset_ids", self.parent_dataset_ids)
        if dataset_id in parents:
            raise ValueError("dataset cannot list itself as a parent")
        if len({item.level for item in self.identity_sets}) != len(self.identity_sets):
            raise ValueError("identity_sets cannot repeat an identity level")
        history = _strings("preprocessing_history", self.preprocessing_history)
        sampling = _freeze_json(self.sampling_assumptions)
        metadata = _freeze_json(self.metadata)
        if not isinstance(sampling, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("sampling_assumptions and metadata must be mappings")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "upstream_source", upstream_source)
        object.__setattr__(self, "content_sha256", _optional_sha256("content_sha256", self.content_sha256))
        object.__setattr__(self, "parent_dataset_ids", parents)
        object.__setattr__(self, "preprocessing_history", history)
        object.__setattr__(self, "sampling_assumptions", sampling)
        object.__setattr__(self, "metadata", metadata)

    @property
    def lineage_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return _display(self.lineage_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "upstream_source": self.upstream_source,
            "version": self.version,
            "revision": self.revision,
            "content_sha256": self.content_sha256,
            "parent_dataset_ids": list(self.parent_dataset_ids),
            "identity_sets": [item.to_dict() for item in self.identity_sets],
            "preprocessing_history": list(self.preprocessing_history),
            "sampling_assumptions": _thaw_json(self.sampling_assumptions),
            "license": self.license,
            "citation": self.citation,
            "lineage_completeness": self.lineage_completeness.value,
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["lineage_sha256"] = self.lineage_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class ModelLineage:
    """Immutable lineage for a trained/pretrained model or representation artifact."""

    model_id: str
    upstream_source: str
    version: str | None = None
    revision: str | None = None
    checkpoint_sha256: str | None = None
    pretraining_dataset_ids: tuple[str, ...] = ()
    pretraining_lineage_completeness: LineageCompleteness = LineageCompleteness.UNKNOWN
    input_assumptions: Mapping[str, Any] = field(default_factory=dict)
    license: str | None = None
    citation: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ModelLineage schema_version must be 2")
        model_id = _nonempty("model_id", self.model_id)
        source = _nonempty("upstream_source", self.upstream_source)
        datasets = _strings("pretraining_dataset_ids", self.pretraining_dataset_ids)
        assumptions = _freeze_json(self.input_assumptions)
        metadata = _freeze_json(self.metadata)
        if not isinstance(assumptions, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("input_assumptions and metadata must be mappings")
        object.__setattr__(self, "model_id", model_id)
        object.__setattr__(self, "upstream_source", source)
        object.__setattr__(self, "checkpoint_sha256", _optional_sha256("checkpoint_sha256", self.checkpoint_sha256))
        object.__setattr__(self, "pretraining_dataset_ids", datasets)
        object.__setattr__(self, "input_assumptions", assumptions)
        object.__setattr__(self, "metadata", metadata)

    @property
    def lineage_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return _display(self.lineage_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "upstream_source": self.upstream_source,
            "version": self.version,
            "revision": self.revision,
            "checkpoint_sha256": self.checkpoint_sha256,
            "pretraining_dataset_ids": list(self.pretraining_dataset_ids),
            "pretraining_lineage_completeness": self.pretraining_lineage_completeness.value,
            "input_assumptions": _thaw_json(self.input_assumptions),
            "license": self.license,
            "citation": self.citation,
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["lineage_sha256"] = self.lineage_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class PretrainingOverlapAudit:
    status: OverlapStatus
    model_id: str
    evaluation_dataset_id: str
    model_lineage_sha256: str
    evaluation_dataset_lineage_sha256: str
    matched_dataset_ids: tuple[str, ...] = ()
    scope: str = "dataset_domain"
    reason: str = ""
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("PretrainingOverlapAudit schema_version must be 2")
        object.__setattr__(self, "model_id", _nonempty("model_id", self.model_id))
        object.__setattr__(self, "evaluation_dataset_id", _nonempty("evaluation_dataset_id", self.evaluation_dataset_id))
        object.__setattr__(self, "model_lineage_sha256", _require_sha256("model_lineage_sha256", self.model_lineage_sha256))
        object.__setattr__(self, "evaluation_dataset_lineage_sha256", _require_sha256("evaluation_dataset_lineage_sha256", self.evaluation_dataset_lineage_sha256))
        object.__setattr__(self, "matched_dataset_ids", _strings("matched_dataset_ids", self.matched_dataset_ids))
        object.__setattr__(self, "scope", _nonempty("scope", self.scope))
        object.__setattr__(self, "reason", _nonempty("reason", self.reason))

    @property
    def audit_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return _display(self.audit_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "status": self.status.value,
            "model_id": self.model_id,
            "evaluation_dataset_id": self.evaluation_dataset_id,
            "model_lineage_sha256": self.model_lineage_sha256,
            "evaluation_dataset_lineage_sha256": self.evaluation_dataset_lineage_sha256,
            "matched_dataset_ids": list(self.matched_dataset_ids),
            "scope": self.scope,
            "reason": self.reason,
        }
        if include_identity:
            payload["audit_sha256"] = self.audit_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


def _dataset_domain_closure(
    dataset: DatasetLineage,
    known_datasets: Mapping[str, DatasetLineage],
) -> tuple[str, ...]:
    found: list[str] = []
    visiting: set[str] = set()

    def visit(current: DatasetLineage) -> None:
        if current.dataset_id in visiting:
            raise ValueError("dataset lineage contains a parent cycle")
        if current.dataset_id in found:
            return
        visiting.add(current.dataset_id)
        found.append(current.dataset_id)
        for parent_id in current.parent_dataset_ids:
            if parent_id not in found:
                found.append(parent_id)
            parent = known_datasets.get(parent_id)
            if parent is not None:
                visit(parent)
        visiting.remove(current.dataset_id)

    visit(dataset)
    return tuple(found)


def audit_pretraining_overlap(
    model: ModelLineage,
    evaluation_dataset: DatasetLineage,
    *,
    known_datasets: Mapping[str, DatasetLineage] | None = None,
) -> PretrainingOverlapAudit:
    """Assess dataset-domain overlap without turning unknown lineage into disjointness."""

    known = dict(known_datasets or {})
    known.setdefault(evaluation_dataset.dataset_id, evaluation_dataset)
    evaluation_domains = set(_dataset_domain_closure(evaluation_dataset, known))
    pretraining_domains = set(model.pretraining_dataset_ids)
    matched = tuple(sorted(evaluation_domains & pretraining_domains))

    if matched:
        status = OverlapStatus.OVERLAP_DETECTED
        reason = "pretraining dataset/domain identity intersects evaluation dataset ancestry"
    elif (
        model.pretraining_lineage_completeness is LineageCompleteness.UNKNOWN
        or evaluation_dataset.lineage_completeness is LineageCompleteness.UNKNOWN
    ):
        status = OverlapStatus.UNKNOWN_LINEAGE
        reason = "model or evaluation dataset lineage is unknown; disjointness cannot be established"
    elif (
        model.pretraining_lineage_completeness is LineageCompleteness.COMPLETE
        and evaluation_dataset.lineage_completeness is LineageCompleteness.COMPLETE
    ):
        status = OverlapStatus.DISJOINT_VERIFIED
        reason = "complete declared pretraining domains are disjoint from evaluation dataset ancestry"
    else:
        status = OverlapStatus.POSSIBLE_OVERLAP
        reason = "declared domains do not overlap, but at least one lineage is partial"

    return PretrainingOverlapAudit(
        status=status,
        model_id=model.model_id,
        evaluation_dataset_id=evaluation_dataset.dataset_id,
        model_lineage_sha256=model.lineage_sha256,
        evaluation_dataset_lineage_sha256=evaluation_dataset.lineage_sha256,
        matched_dataset_ids=matched,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class ObservationSetAuthority:
    authority_id: str
    dataset_lineage_sha256: str
    role: ObservationRole
    observation_ids: tuple[str, ...]
    domain_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ObservationSetAuthority schema_version must be 2")
        object.__setattr__(self, "authority_id", _nonempty("authority_id", self.authority_id))
        object.__setattr__(self, "dataset_lineage_sha256", _require_sha256("dataset_lineage_sha256", self.dataset_lineage_sha256))
        object.__setattr__(self, "domain_id", _nonempty("domain_id", self.domain_id))
        object.__setattr__(self, "observation_ids", _strings("observation_ids", self.observation_ids))
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def authority_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return _display(self.authority_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "authority_id": self.authority_id,
            "dataset_lineage_sha256": self.dataset_lineage_sha256,
            "role": self.role.value,
            "observation_ids": list(self.observation_ids),
            "domain_id": self.domain_id,
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["authority_sha256"] = self.authority_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class TargetObservationBudget:
    """Separate labeled and unlabeled target-observation budgets."""

    labeled_examples: int = 0
    unlabeled_examples: int = 0
    unlabeled_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("labeled_examples", self.labeled_examples),
            ("unlabeled_examples", self.unlabeled_examples),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.unlabeled_seconds is not None:
            value = self.unlabeled_seconds
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError("unlabeled_seconds must be numeric or None")
            number = float(value)
            if not math.isfinite(number) or number < 0:
                raise ValueError("unlabeled_seconds must be finite and non-negative")
            object.__setattr__(self, "unlabeled_seconds", number)

    def to_dict(self) -> dict[str, Any]:
        return {
            "labeled_examples": self.labeled_examples,
            "unlabeled_examples": self.unlabeled_examples,
            "unlabeled_seconds": self.unlabeled_seconds,
        }


@dataclass(frozen=True, slots=True)
class ObservationConsumption:
    """Exact observation authorities consumed by one state-changing operation."""

    operation_id: str
    operation: OperationKind
    observation_authority_sha256s: tuple[str, ...]
    roles: tuple[ObservationRole, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ObservationConsumption schema_version must be 2")
        object.__setattr__(self, "operation_id", _nonempty("operation_id", self.operation_id))
        shas = tuple(_require_sha256("observation authority SHA-256", value) for value in self.observation_authority_sha256s)
        if len(set(shas)) != len(shas):
            raise ValueError("observation_authority_sha256s cannot contain duplicates")
        if len(shas) != len(self.roles):
            raise ValueError("roles must align one-to-one with observation authorities")
        allowed = _ALLOWED_ROLES[self.operation]
        unauthorized = sorted({role.value for role in self.roles if role not in allowed})
        if unauthorized:
            raise ValueError(
                f"{self.operation.value} cannot consume observation roles {unauthorized}; "
                f"allowed={sorted(role.value for role in allowed)}"
            )
        object.__setattr__(self, "observation_authority_sha256s", shas)

    @classmethod
    def bind(
        cls,
        *,
        operation_id: str,
        operation: OperationKind,
        observations: Sequence[ObservationSetAuthority],
    ) -> "ObservationConsumption":
        return cls(
            operation_id=operation_id,
            operation=operation,
            observation_authority_sha256s=tuple(item.authority_sha256 for item in observations),
            roles=tuple(item.role for item in observations),
        )

    @property
    def consumption_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "operation": self.operation.value,
            "observation_authority_sha256s": list(self.observation_authority_sha256s),
            "roles": [role.value for role in self.roles],
        }
        if include_identity:
            payload["consumption_sha256"] = self.consumption_sha256
            payload["display_fingerprint"] = _display(self.consumption_sha256)
        return payload


@dataclass(frozen=True, slots=True)
class PreprocessingFitAuthority:
    transform_id: str
    fit_kind: TransformFitKind
    implementation: str
    implementation_version: str
    state_sha256: str
    consumption: ObservationConsumption | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("PreprocessingFitAuthority schema_version must be 2")
        object.__setattr__(self, "transform_id", _nonempty("transform_id", self.transform_id))
        object.__setattr__(self, "implementation", _nonempty("implementation", self.implementation))
        object.__setattr__(self, "implementation_version", _nonempty("implementation_version", self.implementation_version))
        object.__setattr__(self, "state_sha256", _require_sha256("state_sha256", self.state_sha256))
        if self.fit_kind is TransformFitKind.PREDECLARED_FIXED:
            if self.consumption is not None:
                raise ValueError("predeclared fixed transforms cannot claim data-fitted consumption")
        elif self.fit_kind is TransformFitKind.DATA_FITTED:
            if self.consumption is None:
                raise ValueError("data-fitted transforms require observation consumption authority")
            if self.consumption.operation is not OperationKind.PREPROCESSING_FIT:
                raise ValueError("data-fitted preprocessing must use preprocessing_fit consumption")
        else:
            raise ValueError("fit_kind must be a TransformFitKind")
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def authority_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "transform_id": self.transform_id,
            "fit_kind": self.fit_kind.value,
            "implementation": self.implementation,
            "implementation_version": self.implementation_version,
            "state_sha256": self.state_sha256,
            "consumption": None if self.consumption is None else self.consumption.to_dict(),
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["authority_sha256"] = self.authority_sha256
            payload["display_fingerprint"] = _display(self.authority_sha256)
        return payload


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
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        if not isinstance(self.primary, bool):
            raise ValueError("primary must be boolean")
        if self.direction is MetricDirection.TARGET_IS_BEST:
            if self.target_value is None or not math.isfinite(float(self.target_value)):
                raise ValueError("target_is_best metrics require a finite target_value")
        elif self.target_value is not None:
            raise ValueError("target_value is only valid for target_is_best metrics")
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def metric_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

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
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["metric_sha256"] = self.metric_sha256
            payload["display_fingerprint"] = _display(self.metric_sha256)
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
        hierarchy = _strings("hierarchy", self.hierarchy, allow_empty=False)
        independent = _nonempty("independent_unit", self.independent_unit)
        case_unit = _nonempty("case_unit", self.case_unit)
        clusters = _strings("cluster_units", self.cluster_units, allow_empty=False)
        strata = _strings("strata", self.strata)
        if independent not in hierarchy:
            raise ValueError("independent_unit must be present in hierarchy")
        if any(unit not in hierarchy for unit in clusters):
            raise ValueError("every cluster unit must be present in hierarchy")
        object.__setattr__(self, "hierarchy", hierarchy)
        object.__setattr__(self, "independent_unit", independent)
        object.__setattr__(self, "case_unit", case_unit)
        object.__setattr__(self, "cluster_units", clusters)
        object.__setattr__(self, "inference_method", _nonempty("inference_method", self.inference_method))
        object.__setattr__(self, "strata", strata)

    @property
    def authority_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

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
            payload["display_fingerprint"] = _display(self.authority_sha256)
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
        object.__setattr__(self, "case_id", _nonempty("case_id", self.case_id))
        object.__setattr__(self, "method_id", _nonempty("method_id", self.method_id))
        metrics: dict[str, float] = {}
        for key, value in self.metrics.items():
            name = _nonempty("metric name", key)
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
                raise ValueError(f"metric {name!r} must be numeric")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(f"metric {name!r} must be finite")
            metrics[name] = number
        if self.status is CaseStatus.OK:
            if self.reason is not None:
                raise ValueError("successful rows cannot carry a failure reason")
        else:
            object.__setattr__(self, "reason", _nonempty("reason", self.reason or ""))
        object.__setattr__(self, "metrics", MappingProxyType(metrics))
        metadata = _freeze_json(self.metadata)
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
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FailurePreservingResultSet:
    """A complete method x case matrix where difficult cases cannot disappear."""

    declared_case_ids: tuple[str, ...]
    method_ids: tuple[str, ...]
    rows: tuple[CaseOutcome, ...]
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("FailurePreservingResultSet schema_version must be 2")
        cases = _strings("declared_case_ids", self.declared_case_ids, allow_empty=False)
        methods = _strings("method_ids", self.method_ids, allow_empty=False)
        expected = {(method, case) for method in methods for case in cases}
        actual = {(row.method_id, row.case_id) for row in self.rows}
        if len(actual) != len(self.rows):
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

    @property
    def result_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    def status_counts(self) -> dict[str, int]:
        counts = {status.value: 0 for status in CaseStatus}
        for row in self.rows:
            counts[row.status.value] += 1
        return counts

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
            payload["display_fingerprint"] = _display(self.result_sha256)
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceClaim:
    claim_id: str
    domain: EvidenceDomain
    scope: str
    qualification: ClaimQualification
    evidence_sha256s: tuple[str, ...]
    model_id: str | None = None
    evaluation_dataset_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("EvidenceClaim schema_version must be 2")
        object.__setattr__(self, "claim_id", _nonempty("claim_id", self.claim_id))
        object.__setattr__(self, "scope", _nonempty("scope", self.scope))
        shas = tuple(_require_sha256("evidence SHA-256", value) for value in self.evidence_sha256s)
        if not shas:
            raise ValueError("evidence_sha256s must be non-empty")
        if len(set(shas)) != len(shas):
            raise ValueError("evidence_sha256s cannot contain duplicates")
        object.__setattr__(self, "evidence_sha256s", shas)
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "domain": self.domain.value,
            "scope": self.scope,
            "qualification": self.qualification.value,
            "evidence_sha256s": list(self.evidence_sha256s),
            "model_id": self.model_id,
            "evaluation_dataset_id": self.evaluation_dataset_id,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScientificStudyAuthority:
    """Top-level immutable study ledger for promoted neural comparisons."""

    study_id: str
    protocol_sha256: str
    datasets: tuple[DatasetLineage, ...]
    models: tuple[ModelLineage, ...]
    observations: tuple[ObservationSetAuthority, ...]
    preprocessing: tuple[PreprocessingFitAuthority, ...]
    metrics: tuple[MetricSpec, ...]
    repeated_measures: RepeatedMeasuresAuthority
    overlap_audits: tuple[PretrainingOverlapAudit, ...] = ()
    claims: tuple[EvidenceClaim, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ScientificStudyAuthority schema_version must be 2")
        object.__setattr__(self, "study_id", _nonempty("study_id", self.study_id))
        object.__setattr__(self, "protocol_sha256", _require_sha256("protocol_sha256", self.protocol_sha256))
        for name, values, key in (
            ("datasets", self.datasets, lambda item: item.dataset_id),
            ("models", self.models, lambda item: item.model_id),
            ("observations", self.observations, lambda item: item.authority_id),
            ("preprocessing", self.preprocessing, lambda item: item.transform_id),
            ("metrics", self.metrics, lambda item: item.metric_id),
            ("claims", self.claims, lambda item: item.claim_id),
        ):
            keys = [key(item) for item in values]
            if len(set(keys)) != len(keys):
                raise ValueError(f"{name} cannot contain duplicate identities")
        if not self.datasets:
            raise ValueError("a scientific study requires at least one dataset lineage")
        if not self.metrics:
            raise ValueError("a scientific study requires at least one metric spec")
        if sum(1 for metric in self.metrics if metric.primary) != 1:
            raise ValueError("a promoted scientific study requires exactly one primary metric")

        audits = {(item.model_id, item.evaluation_dataset_id): item for item in self.overlap_audits}
        if len(audits) != len(self.overlap_audits):
            raise ValueError("overlap_audits cannot repeat a model/evaluation-dataset pair")
        for claim in self.claims:
            if claim.model_id is None or claim.evaluation_dataset_id is None:
                continue
            audit = audits.get((claim.model_id, claim.evaluation_dataset_id))
            if audit is None:
                if claim.qualification is ClaimQualification.CLEAN:
                    raise ValueError("clean model/evaluation claims require an explicit overlap audit")
                continue
            if audit.status is OverlapStatus.OVERLAP_DETECTED:
                if claim.qualification is not ClaimQualification.CONTAMINATED_PRETRAINING_OVERLAP:
                    raise ValueError("overlap-detected claims must be labeled contaminated_pretraining_overlap")
            elif audit.status in {OverlapStatus.UNKNOWN_LINEAGE, OverlapStatus.POSSIBLE_OVERLAP}:
                if claim.qualification is not ClaimQualification.UNKNOWN_PRETRAINING_LINEAGE:
                    raise ValueError("unknown/possible-overlap claims must be labeled unknown_pretraining_lineage")
            elif audit.status is OverlapStatus.DISJOINT_VERIFIED:
                if claim.qualification is ClaimQualification.CLEAN:
                    pass
                elif claim.qualification not in {ClaimQualification.DESCRIPTIVE_ONLY, ClaimQualification.NOT_APPLICABLE}:
                    raise ValueError("disjoint verified claims cannot be labeled as overlap-contaminated")

        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def study_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return _display(self.study_sha256)

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "protocol_sha256": self.protocol_sha256,
            "datasets": [item.to_dict() for item in self.datasets],
            "models": [item.to_dict() for item in self.models],
            "observations": [item.to_dict() for item in self.observations],
            "preprocessing": [item.to_dict() for item in self.preprocessing],
            "metrics": [item.to_dict() for item in self.metrics],
            "repeated_measures": self.repeated_measures.to_dict(),
            "overlap_audits": [item.to_dict() for item in self.overlap_audits],
            "claims": [item.to_dict() for item in self.claims],
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["study_sha256"] = self.study_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload

    def report(self) -> dict[str, Any]:
        grouped = {domain.value: [] for domain in EvidenceDomain}
        for claim in self.claims:
            grouped[claim.domain.value].append(claim.to_dict())
        return {
            "schema": "orion.scientific_authority.v2",
            "study_id": self.study_id,
            "study_sha256": self.study_sha256,
            "display_fingerprint": self.display_fingerprint,
            "protocol_sha256": self.protocol_sha256,
            "pretraining_overlap": [item.to_dict() for item in self.overlap_audits],
            "metric_specs": [item.to_dict() for item in self.metrics],
            "repeated_measures": self.repeated_measures.to_dict(),
            "evidence_domains": grouped,
            "claim_scope": [item.to_dict() for item in self.claims],
        }


def bind_longitudinal_case_authority(
    case_payload: Mapping[str, Any],
    *,
    dataset_lineage: DatasetLineage,
    calibration_per_class: int,
    unlabeled_target_observation_indices: Sequence[int] = (),
) -> tuple[tuple[ObservationSetAuthority, ...], TargetObservationBudget]:
    """Bind an existing #26/#27 longitudinal case manifest into ORION v2 roles.

    This consumes the serialized `LongitudinalCaseAuthority` shape rather than
    importing neuros-foundation. The existing split/calibration authority remains
    the source of row identity; ORION adds role and information-budget governance.
    """

    if str(case_payload.get("dataset_id")) != dataset_lineage.dataset_id:
        raise ValueError("longitudinal case dataset_id does not match dataset lineage")
    if isinstance(calibration_per_class, bool) or not isinstance(calibration_per_class, int) or calibration_per_class < 0:
        raise ValueError("calibration_per_class must be a non-negative integer")

    case_id = _nonempty("case_id", str(case_payload.get("case_id", "")))
    source_indices = tuple(int(value) for value in case_payload["source_train_indices"])
    evaluation_indices = tuple(int(value) for value in case_payload["evaluation_indices"])
    calibration_by_class = dict(case_payload["calibration_order_by_class"])
    calibration_indices: list[int] = []
    for label in sorted(calibration_by_class, key=str):
        ordered = tuple(int(value) for value in calibration_by_class[label])
        if len(ordered) < calibration_per_class:
            raise ValueError(
                f"case {case_id!r} has fewer than {calibration_per_class} calibration rows "
                f"for class {label!r}"
            )
        calibration_indices.extend(ordered[:calibration_per_class])

    unlabeled = tuple(int(value) for value in unlabeled_target_observation_indices)
    for name, values in (
        ("source", source_indices),
        ("evaluation", evaluation_indices),
        ("calibration", tuple(calibration_indices)),
        ("unlabeled target", unlabeled),
    ):
        if any(value < 0 for value in values):
            raise ValueError(f"{name} indices cannot be negative")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} indices cannot contain duplicates")

    source_set = set(source_indices)
    evaluation_set = set(evaluation_indices)
    calibration_set = set(calibration_indices)
    unlabeled_set = set(unlabeled)
    if source_set & evaluation_set or source_set & calibration_set:
        raise ValueError("source history overlaps target calibration/final assessment")
    if calibration_set & evaluation_set:
        raise ValueError("target calibration overlaps final assessment")
    if unlabeled_set & evaluation_set:
        raise ValueError(
            "unlabeled target observations cannot borrow untouched final-assessment rows"
        )
    if unlabeled_set & calibration_set:
        raise ValueError("unlabeled and labeled target-observation authorities must be disjoint")

    domain = f"{dataset_lineage.dataset_id}:{case_id}"
    common_metadata = {
        "source_authority_fingerprint": case_payload.get("authority_fingerprint"),
        "partition_fingerprint": case_payload.get("partition_fingerprint"),
        "calibration_split_fingerprint": case_payload.get("calibration_split_fingerprint"),
        "processed_data_sha256": case_payload.get("processed_data_sha256"),
        "history_policy": case_payload.get("history_policy"),
        "held_out_values": case_payload.get("held_out_values"),
    }
    observations = [
        ObservationSetAuthority(
            authority_id=f"{case_id}:source-history",
            dataset_lineage_sha256=dataset_lineage.lineage_sha256,
            role=ObservationRole.SOURCE_HISTORY,
            observation_ids=tuple(str(value) for value in source_indices),
            domain_id=domain,
            metadata=common_metadata,
        ),
        ObservationSetAuthority(
            authority_id=f"{case_id}:labeled-target:{calibration_per_class}",
            dataset_lineage_sha256=dataset_lineage.lineage_sha256,
            role=ObservationRole.LABELED_TARGET_CALIBRATION,
            observation_ids=tuple(str(value) for value in calibration_indices),
            domain_id=domain,
            metadata={**common_metadata, "calibration_per_class": calibration_per_class},
        ),
        ObservationSetAuthority(
            authority_id=f"{case_id}:final-assessment",
            dataset_lineage_sha256=dataset_lineage.lineage_sha256,
            role=ObservationRole.FINAL_ASSESSMENT,
            observation_ids=tuple(str(value) for value in evaluation_indices),
            domain_id=domain,
            metadata=common_metadata,
        ),
    ]
    if unlabeled:
        observations.append(
            ObservationSetAuthority(
                authority_id=f"{case_id}:unlabeled-target",
                dataset_lineage_sha256=dataset_lineage.lineage_sha256,
                role=ObservationRole.UNLABELED_TARGET_OBSERVATION,
                observation_ids=tuple(str(value) for value in unlabeled),
                domain_id=domain,
                metadata=common_metadata,
            )
        )
    return (
        tuple(observations),
        TargetObservationBudget(
            labeled_examples=len(calibration_indices),
            unlabeled_examples=len(unlabeled),
        ),
    )
