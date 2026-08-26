"""Final-assessment governance for adaptive and frozen neural states.

This module closes the evidence loop after state selection. Adaptation authority
may use calibration data to update state and qualification data to decide
retain/rollback. Final-assessment authority is deliberately separate: its rows
and metric scorecard are frozen, and an assessment record can only be created
for an already-selected state.

The contracts are dependency-light and dataset-agnostic. Dataset-specific
packages may derive these authorities from stronger upstream evidence contracts
without ORION depending on those packages.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .adaptation import AdaptationAuthority, AdaptationOutcome, ArtifactIdentity

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> Any:
    """Normalize supported evidence values without silent key collisions."""

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("assessment evidence cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if isinstance(value, np.ndarray):
        return _canonical_json(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            normalized_key = str(key)
            if not normalized_key.strip():
                raise ValueError("assessment evidence mapping keys must be non-empty")
            if normalized_key in normalized:
                raise ValueError(
                    "assessment evidence mapping keys collide after string normalization: "
                    f"{normalized_key!r}"
                )
            normalized[normalized_key] = _canonical_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    raise TypeError(
        "assessment evidence must be composed of JSON-compatible primitives, "
        f"NumPy scalars/arrays, mappings, lists, or tuples; got {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    normalized = _canonical_json(value)
    if isinstance(normalized, dict):
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in normalized.items()}
        )
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        _canonical_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _sha256(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return normalized


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _indices(name: str, values: Any) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.dtype == np.bool_:
        raise ValueError(f"{name} must contain integer sample indices, not booleans")
    try:
        integer = array.astype(np.int64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain integer sample indices") from exc
    if not np.array_equal(array, integer):
        raise ValueError(f"{name} must contain integer sample indices")
    result = tuple(int(value) for value in integer.tolist())
    if not result:
        raise ValueError(f"{name} must be non-empty")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} cannot contain negative indices")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate indices")
    return result


def _metric_names(values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError("metric_names must be a sequence of strings, not one string")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError("metric_names must be an iterable of strings") from exc
    if not raw:
        raise ValueError("metric_names must contain at least one metric name")
    if any(not isinstance(value, str) for value in raw):
        raise ValueError("metric_names must contain strings only")
    names = tuple(value.strip() for value in raw)
    if any(not value for value in names):
        raise ValueError("metric_names cannot contain empty names")
    if len(set(names)) != len(names):
        raise ValueError("metric_names cannot contain duplicates")
    return names


def _metrics(values: Mapping[str, Any]) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError("final-assessment metrics must be a mapping")
    normalized: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(key, str):
            raise ValueError("final-assessment metric names must be strings")
        name = key.strip()
        if not name:
            raise ValueError("final-assessment metric names must be non-empty")
        if name in normalized:
            raise ValueError("final-assessment metric names cannot contain duplicates")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
            raise ValueError(f"final-assessment metric {name!r} must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"final-assessment metric {name!r} must be finite")
        normalized[name] = number
    if not normalized:
        raise ValueError("final assessment requires at least one metric")
    return MappingProxyType(normalized)


class SelectionKind(str, Enum):
    """How a state became frozen before final assessment."""

    FROZEN = "frozen"
    ADAPTED = "adapted"


@dataclass(frozen=True, slots=True)
class FinalAssessmentAuthority:
    """Frozen untouched rows and scorecard for final scientific assessment."""

    authority_id: str
    dataset_id: str
    split_unit: str
    assessment_indices: tuple[int, ...]
    processed_data_sha256: str
    n_samples: int
    source_authority_fingerprint: str
    metric_names: tuple[str, ...]
    protocol_fingerprint: str | None = None
    seed: int = 0
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("FinalAssessmentAuthority schema_version must be 1")
        authority_id = _nonempty("authority_id", self.authority_id)
        dataset_id = _nonempty("dataset_id", self.dataset_id)
        split_unit = _nonempty("split_unit", self.split_unit)
        source = _nonempty(
            "source_authority_fingerprint", self.source_authority_fingerprint
        )
        if self.protocol_fingerprint is not None:
            _nonempty("protocol_fingerprint", self.protocol_fingerprint)
        if (
            isinstance(self.n_samples, bool)
            or not isinstance(self.n_samples, int)
            or self.n_samples < 1
        ):
            raise ValueError("n_samples must be a positive integer")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        indices = _indices("assessment_indices", self.assessment_indices)
        if max(indices) >= self.n_samples:
            raise ValueError("final-assessment authority contains out-of-range indices")
        names = _metric_names(self.metric_names)
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        object.__setattr__(self, "authority_id", authority_id)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "split_unit", split_unit)
        object.__setattr__(self, "assessment_indices", indices)
        object.__setattr__(
            self,
            "processed_data_sha256",
            _sha256("processed_data_sha256", self.processed_data_sha256),
        )
        object.__setattr__(self, "source_authority_fingerprint", source)
        object.__setattr__(self, "metric_names", names)
        object.__setattr__(self, "metadata", metadata)

    @property
    def authority_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def require_assessment_indices(self, values: Any) -> tuple[int, ...]:
        actual = _indices("final-assessment indices", values)
        if actual != self.assessment_indices:
            raise ValueError(
                "final assessment must use the complete frozen assessment indices in exact order"
            )
        return actual

    def require_metrics(self, values: Mapping[str, Any]) -> Mapping[str, float]:
        metrics = _metrics(values)
        if set(metrics) != set(self.metric_names):
            raise ValueError(
                "final assessment must report the exact predeclared metric scorecard; "
                f"expected={list(self.metric_names)}, actual={sorted(metrics)}"
            )
        return metrics

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "authority_id": self.authority_id,
            "dataset_id": self.dataset_id,
            "split_unit": self.split_unit,
            "assessment_indices": list(self.assessment_indices),
            "processed_data_sha256": self.processed_data_sha256,
            "n_samples": self.n_samples,
            "source_authority_fingerprint": self.source_authority_fingerprint,
            "metric_names": list(self.metric_names),
            "protocol_fingerprint": self.protocol_fingerprint,
            "seed": self.seed,
            "metadata": _thaw_json(self.metadata),
        }
        if include_fingerprint:
            payload["authority_fingerprint"] = self.authority_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class SelectedState:
    """One immutable model/representation state selected before final scoring."""

    selection_id: str
    source_authority_fingerprint: str
    artifact: ArtifactIdentity
    kind: SelectionKind
    adaptation_authority_fingerprint: str | None = None
    selection_evidence_fingerprint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("SelectedState schema_version must be 1")
        if not isinstance(self.kind, SelectionKind):
            raise ValueError("kind must be a SelectionKind")
        selection_id = _nonempty("selection_id", self.selection_id)
        source = _nonempty(
            "source_authority_fingerprint", self.source_authority_fingerprint
        )
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")

        if self.kind is SelectionKind.ADAPTED:
            _nonempty(
                "adaptation_authority_fingerprint",
                self.adaptation_authority_fingerprint or "",
            )
            _nonempty(
                "selection_evidence_fingerprint",
                self.selection_evidence_fingerprint or "",
            )
        else:
            if self.adaptation_authority_fingerprint is not None:
                raise ValueError(
                    "frozen selected states cannot claim an adaptation authority"
                )
            if self.selection_evidence_fingerprint is not None:
                raise ValueError(
                    "frozen selected states cannot claim adaptation selection evidence"
                )

        object.__setattr__(self, "selection_id", selection_id)
        object.__setattr__(self, "source_authority_fingerprint", source)
        object.__setattr__(self, "metadata", metadata)

    @classmethod
    def frozen(
        cls,
        *,
        selection_id: str,
        source_authority_fingerprint: str,
        artifact: ArtifactIdentity,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SelectedState":
        """Freeze a predeclared/no-update baseline without inventing adaptation."""

        return cls(
            selection_id=selection_id,
            source_authority_fingerprint=source_authority_fingerprint,
            artifact=artifact,
            kind=SelectionKind.FROZEN,
            metadata={} if metadata is None else metadata,
        )

    @classmethod
    def from_adaptation_outcome(
        cls,
        outcome: AdaptationOutcome,
        *,
        selection_id: str,
        source_authority_fingerprint: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SelectedState":
        """Freeze the exact artifact selected by a retain/rollback outcome."""

        return cls(
            selection_id=selection_id,
            source_authority_fingerprint=source_authority_fingerprint,
            artifact=outcome.active_artifact,
            kind=SelectionKind.ADAPTED,
            adaptation_authority_fingerprint=outcome.authority_fingerprint,
            selection_evidence_fingerprint=outcome.outcome_fingerprint,
            metadata={} if metadata is None else metadata,
        )

    @property
    def selection_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "selection_id": self.selection_id,
            "source_authority_fingerprint": self.source_authority_fingerprint,
            "kind": self.kind.value,
            "artifact": self.artifact.to_dict(),
            "adaptation_authority_fingerprint": self.adaptation_authority_fingerprint,
            "selection_evidence_fingerprint": self.selection_evidence_fingerprint,
            "metadata": _thaw_json(self.metadata),
        }
        if include_fingerprint:
            payload["selection_fingerprint"] = self.selection_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class AdaptiveStudyAuthority:
    """Bind calibration/qualification governance to one untouched final authority."""

    study_id: str
    source_authority_fingerprint: str
    adaptation_authority: AdaptationAuthority
    final_assessment_authority: FinalAssessmentAuthority
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("AdaptiveStudyAuthority schema_version must be 1")
        study_id = _nonempty("study_id", self.study_id)
        source = _nonempty(
            "source_authority_fingerprint", self.source_authority_fingerprint
        )
        adaptation = self.adaptation_authority
        final = self.final_assessment_authority

        if adaptation.source_authority_fingerprint != source:
            raise ValueError(
                "adaptation authority must be derived from the same source study authority"
            )
        if final.source_authority_fingerprint != source:
            raise ValueError(
                "final-assessment authority must be derived from the same source study authority"
            )
        if adaptation.dataset_id != final.dataset_id:
            raise ValueError("adaptation and final-assessment dataset identities differ")
        if adaptation.split_unit != final.split_unit:
            raise ValueError("adaptation and final-assessment split units differ")
        if adaptation.processed_data_sha256 != final.processed_data_sha256:
            raise ValueError("adaptation and final-assessment processed-data identities differ")
        if adaptation.n_samples != final.n_samples:
            raise ValueError("adaptation and final-assessment sample counts differ")
        if adaptation.seed != final.seed:
            raise ValueError("adaptation and final-assessment seeds differ")
        if adaptation.protocol_fingerprint != final.protocol_fingerprint:
            raise ValueError("adaptation and final-assessment protocol fingerprints differ")

        adaptation_set = set(adaptation.adaptation_indices)
        qualification_set = set(adaptation.evaluation_indices)
        final_set = set(final.assessment_indices)
        if adaptation_set & final_set:
            raise ValueError("final-assessment rows overlap adaptation/calibration rows")
        if qualification_set & final_set:
            raise ValueError("final-assessment rows overlap qualification rows")

        object.__setattr__(self, "study_id", study_id)
        object.__setattr__(self, "source_authority_fingerprint", source)

    @property
    def study_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def select_outcome(
        self,
        outcome: AdaptationOutcome,
        *,
        selection_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> SelectedState:
        if outcome.authority_fingerprint != self.adaptation_authority.authority_fingerprint:
            raise ValueError("adaptation outcome does not belong to this study authority")
        return SelectedState.from_adaptation_outcome(
            outcome,
            selection_id=selection_id,
            source_authority_fingerprint=self.source_authority_fingerprint,
            metadata={} if metadata is None else metadata,
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "study_id": self.study_id,
            "source_authority_fingerprint": self.source_authority_fingerprint,
            "adaptation_authority": self.adaptation_authority.to_dict(),
            "final_assessment_authority": self.final_assessment_authority.to_dict(),
        }
        if include_fingerprint:
            payload["study_fingerprint"] = self.study_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class FinalAssessmentRecord:
    """Final metrics for one already-selected state under frozen authority."""

    authority_fingerprint: str
    source_authority_fingerprint: str
    selected_state_fingerprint: str
    selected_artifact: ArtifactIdentity
    assessment_indices: tuple[int, ...]
    metrics: Mapping[str, float]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("FinalAssessmentRecord schema_version must be 1")
        _nonempty("authority_fingerprint", self.authority_fingerprint)
        _nonempty(
            "source_authority_fingerprint", self.source_authority_fingerprint
        )
        _nonempty("selected_state_fingerprint", self.selected_state_fingerprint)
        object.__setattr__(
            self,
            "assessment_indices",
            _indices("assessment_indices", self.assessment_indices),
        )
        object.__setattr__(self, "metrics", _metrics(self.metrics))

    @classmethod
    def record(
        cls,
        selected_state: SelectedState,
        *,
        authority: FinalAssessmentAuthority,
        assessment_indices: Any,
        metrics: Mapping[str, Any],
    ) -> "FinalAssessmentRecord":
        if (
            selected_state.source_authority_fingerprint
            != authority.source_authority_fingerprint
        ):
            raise ValueError(
                "selected state and final assessment do not share the same source authority"
            )
        indices = authority.require_assessment_indices(assessment_indices)
        validated_metrics = authority.require_metrics(metrics)
        return cls(
            authority_fingerprint=authority.authority_fingerprint,
            source_authority_fingerprint=authority.source_authority_fingerprint,
            selected_state_fingerprint=selected_state.selection_fingerprint,
            selected_artifact=selected_state.artifact,
            assessment_indices=indices,
            metrics=validated_metrics,
        )

    @property
    def assessment_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "authority_fingerprint": self.authority_fingerprint,
            "source_authority_fingerprint": self.source_authority_fingerprint,
            "selected_state_fingerprint": self.selected_state_fingerprint,
            "selected_artifact": self.selected_artifact.to_dict(),
            "assessment_indices": list(self.assessment_indices),
            "metrics": dict(self.metrics),
        }
        if include_fingerprint:
            payload["assessment_fingerprint"] = self.assessment_fingerprint
        return payload
