"""Shared deterministic primitives for ORION Scientific Authority v2."""

from __future__ import annotations

import hashlib
import json
import math
import re
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import numpy as np

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("scientific authority cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return canonical_json(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("scientific authority cannot contain object arrays")
        return canonical_json(value.tolist())
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
            normalized[normalized_key] = canonical_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered sets are not valid scientific authority values")
    raise TypeError(
        "scientific authority values must be deterministic JSON-compatible primitives, "
        f"NumPy scalars/arrays, mappings, lists, or tuples; got {type(value).__name__}"
    )


def freeze_json(value: Any) -> Any:
    normalized = canonical_json(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: freeze_json(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(freeze_json(item) for item in normalized)
    return normalized


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def canonical_sha256(payload: Any) -> str:
    raw = json.dumps(
        canonical_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return normalized


def optional_sha256(name: str, value: str | None) -> str | None:
    return None if value is None else require_sha256(name, value)


def nonempty(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def strings(name: str, values: Iterable[str], *, allow_empty: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    result = tuple(nonempty(name, value) for value in values)
    if not allow_empty and not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


def display_fingerprint(full_sha256: str) -> str:
    return require_sha256("full scientific identity", full_sha256)[:16]


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


ALLOWED_ROLES: Mapping[OperationKind, frozenset[ObservationRole]] = {
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
