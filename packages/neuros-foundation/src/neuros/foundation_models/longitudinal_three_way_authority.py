"""Serializable v2 authority for three-way longitudinal adaptation studies.

The v1 ``LongitudinalCaseAuthority`` remains unchanged and replayable. This v2
contract freezes a distinct qualification/state-selection set and an untouched
final-assessment set so adaptive methods cannot reuse state-selection evidence
as final efficacy evidence.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from .benchmark import SplitUnit
from .longitudinal import ordered_group_values
from .longitudinal_authority import processed_data_sha256
from .longitudinal_three_way import ThreeWayCalibrationSplit
from .real_world import EvaluationPartition, GroupedEvaluationData

HistoryPolicy = Literal["prior", "all-other", "custom"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _indices_tuple(name: str, values: Any, *, allow_empty: bool = False) -> tuple[int, ...]:
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
    if not allow_empty and not result:
        raise ValueError(f"{name} must be non-empty")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} cannot contain negative indices")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate indices")
    return result


def _fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return result


def _sha256(name: str, value: str) -> str:
    normalized = str(value).strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return normalized


def _ordered_values(values: np.ndarray) -> tuple[str, ...]:
    return tuple(dict.fromkeys(np.asarray(values).astype(str).tolist()))


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _index_set_sha256(kind: str, processed_sha256: str, indices: tuple[int, ...]) -> str:
    return _canonical_sha256(
        {
            "kind": kind,
            "processed_data_sha256": processed_sha256,
            "indices": list(indices),
        }
    )


@dataclass(frozen=True, slots=True)
class ThreeWayLongitudinalCaseAuthority:
    """Frozen v2 sample authority for one target deployment unit.

    ``qualification_indices`` may influence state selection, such as a
    retain/rollback decision. ``final_assessment_indices`` must not influence
    adaptation, hyperparameters, thresholds, or retain/rollback policy.
    """

    dataset_id: str
    case_id: str
    split_unit: SplitUnit
    held_out_values: tuple[str, ...]
    history_policy: HistoryPolicy
    observed_group_order: tuple[str, ...]
    source_group_values: tuple[str, ...]
    source_train_indices: tuple[int, ...]
    qualification_indices: tuple[int, ...]
    final_assessment_indices: tuple[int, ...]
    calibration_order_by_class: Mapping[str, tuple[int, ...]]
    qualification_fraction: float
    final_assessment_fraction: float
    seed: int
    partition_fingerprint: str
    three_way_split_fingerprint: str
    processed_data_sha256: str
    n_samples: int
    input_shape: tuple[int, ...]
    case_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ThreeWayLongitudinalCaseAuthority schema_version must be 2")
        if not self.dataset_id.strip() or not self.case_id.strip():
            raise ValueError("dataset_id and case_id must be non-empty")
        if self.split_unit == "sample":
            raise ValueError("three-way longitudinal authority requires a deployment-unit split")
        if not self.held_out_values:
            raise ValueError("held_out_values must be non-empty")
        if self.history_policy not in {"prior", "all-other", "custom"}:
            raise ValueError(f"unsupported history_policy={self.history_policy!r}")
        if isinstance(self.n_samples, bool) or not isinstance(self.n_samples, int) or self.n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        if not self.input_shape or self.input_shape[0] != self.n_samples:
            raise ValueError("input_shape must begin with n_samples")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        processed_sha = _sha256("processed_data_sha256", self.processed_data_sha256)
        qualification_fraction = _fraction(
            "qualification_fraction", self.qualification_fraction
        )
        final_fraction = _fraction(
            "final_assessment_fraction", self.final_assessment_fraction
        )
        if qualification_fraction + final_fraction >= 1.0:
            raise ValueError(
                "qualification_fraction + final_assessment_fraction must be strictly less than 1"
            )

        source = _indices_tuple("source_train_indices", self.source_train_indices)
        qualification = _indices_tuple("qualification_indices", self.qualification_indices)
        final_assessment = _indices_tuple(
            "final_assessment_indices", self.final_assessment_indices
        )
        calibration = {
            str(label): _indices_tuple(
                f"calibration_order_by_class[{label!r}]", values, allow_empty=True
            )
            for label, values in self.calibration_order_by_class.items()
        }
        if not calibration:
            raise ValueError("calibration_order_by_class must be non-empty")

        all_sets: list[tuple[str, set[int]]] = [
            ("source", set(source)),
            ("qualification", set(qualification)),
            ("final_assessment", set(final_assessment)),
        ] + [
            (f"calibration[{label}]", set(values))
            for label, values in sorted(calibration.items())
        ]
        for index, (left_name, left) in enumerate(all_sets):
            if left and max(left) >= self.n_samples:
                raise ValueError(f"{left_name} contains out-of-range sample indices")
            for right_name, right in all_sets[index + 1 :]:
                overlap = left & right
                if overlap:
                    raise ValueError(
                        f"authority index sets must be pairwise disjoint: {left_name} vs "
                        f"{right_name}, overlap={sorted(overlap)[:8]}"
                    )

        object.__setattr__(self, "source_train_indices", source)
        object.__setattr__(self, "qualification_indices", qualification)
        object.__setattr__(self, "final_assessment_indices", final_assessment)
        object.__setattr__(self, "calibration_order_by_class", MappingProxyType(calibration))
        object.__setattr__(self, "processed_data_sha256", processed_sha)
        object.__setattr__(self, "qualification_fraction", qualification_fraction)
        object.__setattr__(self, "final_assessment_fraction", final_fraction)
        object.__setattr__(self, "case_metadata", MappingProxyType(dict(self.case_metadata)))

    @classmethod
    def from_split(
        cls,
        split: ThreeWayCalibrationSplit,
        *,
        case_id: str,
        history_policy: HistoryPolicy,
        observed_group_order: tuple[str, ...] | None = None,
        case_metadata: Mapping[str, Any] | None = None,
    ) -> "ThreeWayLongitudinalCaseAuthority":
        data = split.partition.data
        unit = split.partition.split_unit
        if unit == "sample":
            raise ValueError("three-way longitudinal authority requires a deployment-unit split")
        group = np.asarray(data.groups[unit]).astype(str)
        observed = (
            ordered_group_values(data, split_unit=unit)
            if observed_group_order is None
            else observed_group_order
        )
        source_values = _ordered_values(group[split.source_train_indices])
        return cls(
            dataset_id=data.dataset_id,
            case_id=case_id,
            split_unit=unit,
            held_out_values=tuple(str(v) for v in split.partition.held_out_values),
            history_policy=history_policy,
            observed_group_order=tuple(str(v) for v in observed),
            source_group_values=source_values,
            source_train_indices=_indices_tuple(
                "source_train_indices", split.source_train_indices
            ),
            qualification_indices=_indices_tuple(
                "qualification_indices", split.qualification_indices
            ),
            final_assessment_indices=_indices_tuple(
                "final_assessment_indices", split.final_assessment_indices
            ),
            calibration_order_by_class={
                label: _indices_tuple(
                    f"calibration_order_by_class[{label!r}]", values, allow_empty=True
                )
                for label, values in split.calibration_order_by_class.items()
            },
            qualification_fraction=float(split.qualification_fraction),
            final_assessment_fraction=float(split.final_assessment_fraction),
            seed=int(split.seed),
            partition_fingerprint=split.partition.fingerprint,
            three_way_split_fingerprint=split.fingerprint,
            processed_data_sha256=processed_data_sha256(data),
            n_samples=int(len(data.X)),
            input_shape=tuple(int(v) for v in np.asarray(data.X).shape),
            case_metadata={} if case_metadata is None else case_metadata,
        )

    @property
    def authority_fingerprint(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    @property
    def qualification_set_sha256(self) -> str:
        return _index_set_sha256(
            "qualification", self.processed_data_sha256, self.qualification_indices
        )

    @property
    def final_assessment_set_sha256(self) -> str:
        return _index_set_sha256(
            "final-assessment", self.processed_data_sha256, self.final_assessment_indices
        )

    @property
    def max_budget_per_class(self) -> int:
        return min(len(values) for values in self.calibration_order_by_class.values())

    def calibration_indices(self, per_class: int) -> tuple[int, ...]:
        if isinstance(per_class, bool) or not isinstance(per_class, (int, np.integer)):
            raise ValueError("calibration budget must be a non-negative integer")
        budget = int(per_class)
        if budget < 0:
            raise ValueError("calibration budget must be non-negative")
        if budget > self.max_budget_per_class:
            raise ValueError(
                f"budget {budget} exceeds balanced maximum {self.max_budget_per_class}"
            )
        if budget == 0:
            return ()
        selected: list[int] = []
        for _, values in sorted(self.calibration_order_by_class.items()):
            selected.extend(values[:budget])
        return tuple(sorted(selected))

    def calibration_budget_sha256(self, per_class: int) -> str:
        indices = self.calibration_indices(per_class)
        budget = int(per_class)
        return _index_set_sha256(
            f"calibration-budget-{budget}",
            self.processed_data_sha256,
            indices,
        )

    def require_calibration_indices(self, per_class: int, values: Any) -> tuple[int, ...]:
        expected = self.calibration_indices(per_class)
        actual = _indices_tuple(
            "calibration indices", values, allow_empty=len(expected) == 0
        )
        if actual != expected:
            raise ValueError(
                "calibration must use the exact frozen budget indices in canonical order"
            )
        return actual

    def require_qualification_indices(self, values: Any) -> tuple[int, ...]:
        actual = _indices_tuple("qualification indices", values)
        if actual != self.qualification_indices:
            raise ValueError(
                "qualification must use the complete frozen qualification set in exact order"
            )
        return actual

    def require_final_assessment_indices(self, values: Any) -> tuple[int, ...]:
        actual = _indices_tuple("final-assessment indices", values)
        if actual != self.final_assessment_indices:
            raise ValueError(
                "final assessment must use the complete frozen final-assessment set in exact order"
            )
        return actual

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "kind": "three_way_longitudinal_case_authority",
            "dataset_id": self.dataset_id,
            "case_id": self.case_id,
            "split_unit": self.split_unit,
            "held_out_values": list(self.held_out_values),
            "history_policy": self.history_policy,
            "observed_group_order": list(self.observed_group_order),
            "source_group_values": list(self.source_group_values),
            "source_train_indices": list(self.source_train_indices),
            "qualification_indices": list(self.qualification_indices),
            "final_assessment_indices": list(self.final_assessment_indices),
            "calibration_order_by_class": {
                key: list(values)
                for key, values in sorted(self.calibration_order_by_class.items())
            },
            "qualification_fraction": self.qualification_fraction,
            "final_assessment_fraction": self.final_assessment_fraction,
            "seed": self.seed,
            "partition_fingerprint": self.partition_fingerprint,
            "three_way_split_fingerprint": self.three_way_split_fingerprint,
            "processed_data_sha256": self.processed_data_sha256,
            "qualification_set_sha256": self.qualification_set_sha256,
            "final_assessment_set_sha256": self.final_assessment_set_sha256,
            "n_samples": self.n_samples,
            "input_shape": list(self.input_shape),
            "case_metadata": dict(self.case_metadata),
        }
        if include_fingerprint:
            payload["authority_fingerprint"] = self.authority_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ThreeWayLongitudinalCaseAuthority":
        if int(payload.get("schema_version", 0)) != 2:
            raise ValueError("three-way longitudinal authority requires schema_version=2")
        value = cls(
            dataset_id=str(payload["dataset_id"]),
            case_id=str(payload["case_id"]),
            split_unit=str(payload["split_unit"]),  # type: ignore[arg-type]
            held_out_values=tuple(str(v) for v in payload["held_out_values"]),
            history_policy=str(payload["history_policy"]),  # type: ignore[arg-type]
            observed_group_order=tuple(str(v) for v in payload["observed_group_order"]),
            source_group_values=tuple(str(v) for v in payload["source_group_values"]),
            source_train_indices=tuple(int(v) for v in payload["source_train_indices"]),
            qualification_indices=tuple(int(v) for v in payload["qualification_indices"]),
            final_assessment_indices=tuple(
                int(v) for v in payload["final_assessment_indices"]
            ),
            calibration_order_by_class={
                str(key): tuple(int(v) for v in values)
                for key, values in dict(payload["calibration_order_by_class"]).items()
            },
            qualification_fraction=float(payload["qualification_fraction"]),
            final_assessment_fraction=float(payload["final_assessment_fraction"]),
            seed=int(payload["seed"]),
            partition_fingerprint=str(payload["partition_fingerprint"]),
            three_way_split_fingerprint=str(payload["three_way_split_fingerprint"]),
            processed_data_sha256=str(payload["processed_data_sha256"]),
            n_samples=int(payload["n_samples"]),
            input_shape=tuple(int(v) for v in payload["input_shape"]),
            case_metadata=dict(payload.get("case_metadata", {})),
            schema_version=2,
        )
        expected_authority = payload.get("authority_fingerprint")
        if expected_authority is not None and str(expected_authority) != value.authority_fingerprint:
            raise ValueError("authority_fingerprint does not match serialized content")
        expected_qualification = payload.get("qualification_set_sha256")
        if expected_qualification is not None and str(expected_qualification) != value.qualification_set_sha256:
            raise ValueError("qualification_set_sha256 does not match serialized content")
        expected_final = payload.get("final_assessment_set_sha256")
        if expected_final is not None and str(expected_final) != value.final_assessment_set_sha256:
            raise ValueError("final_assessment_set_sha256 does not match serialized content")
        return value

    def restore(self, data: GroupedEvaluationData) -> ThreeWayCalibrationSplit:
        """Validate loaded processed data and reconstruct the exact v2 split."""
        if data.dataset_id != self.dataset_id:
            raise ValueError(
                f"dataset mismatch: authority={self.dataset_id!r}, loaded={data.dataset_id!r}"
            )
        if len(data.X) != self.n_samples or tuple(np.asarray(data.X).shape) != self.input_shape:
            raise ValueError("processed dataset sample count/shape differs from authority")
        actual_sha = processed_data_sha256(data)
        if actual_sha != self.processed_data_sha256:
            raise ValueError("processed neural data SHA-256 differs from authority")
        if self.split_unit not in data.groups:
            raise ValueError(f"loaded dataset has no {self.split_unit!r} group")

        n = len(data.X)
        source = np.asarray(self.source_train_indices, dtype=np.int64)
        qualification = np.asarray(self.qualification_indices, dtype=np.int64)
        final_assessment = np.asarray(self.final_assessment_indices, dtype=np.int64)
        calibration = {
            label: np.asarray(values, dtype=np.int64)
            for label, values in self.calibration_order_by_class.items()
        }
        all_indices = [source, qualification, final_assessment, *calibration.values()]
        if any(np.any(values < 0) or np.any(values >= n) for values in all_indices):
            raise ValueError("authority contains out-of-range sample indices")

        calibration_flat = (
            np.concatenate(list(calibration.values()))
            if calibration
            else np.asarray([], dtype=np.int64)
        )
        test = np.sort(
            np.concatenate([qualification, final_assessment, calibration_flat]).astype(
                np.int64, copy=False
            )
        )
        partition = EvaluationPartition(
            data=data,
            split_unit=self.split_unit,
            train_indices=source,
            test_indices=test,
            held_out_values=self.held_out_values,
        )
        if partition.fingerprint != self.partition_fingerprint:
            raise ValueError("partition fingerprint differs from authority")

        split = ThreeWayCalibrationSplit(
            partition=partition,
            qualification_indices=qualification,
            final_assessment_indices=final_assessment,
            calibration_order_by_class=calibration,
            qualification_fraction=self.qualification_fraction,
            final_assessment_fraction=self.final_assessment_fraction,
            seed=self.seed,
        )
        if split.fingerprint != self.three_way_split_fingerprint:
            raise ValueError("three-way split fingerprint differs from authority")

        group = np.asarray(data.groups[self.split_unit]).astype(str)
        observed = ordered_group_values(data, split_unit=self.split_unit)
        if observed != self.observed_group_order:
            raise ValueError(
                "deployment-unit order differs from authority; "
                f"authority={self.observed_group_order}, loaded={observed}"
            )
        if set(group[test].tolist()) != set(self.held_out_values):
            raise ValueError("held-out indices do not match authority held_out_values")
        actual_source_values = _ordered_values(group[source])
        if actual_source_values != self.source_group_values:
            raise ValueError("source deployment-unit identities differ from authority")

        source_set = set(source.tolist())
        expected_source = set(
            np.flatnonzero(np.isin(group, np.asarray(self.source_group_values, dtype=str))).tolist()
        )
        if source_set != expected_source:
            raise ValueError("authority source indices do not cover source groups exactly")

        if self.history_policy == "prior":
            if len(self.held_out_values) != 1:
                raise ValueError("prior authority requires exactly one held-out group")
            held = self.held_out_values[0]
            if held not in observed:
                raise ValueError("held-out group missing from observed chronology")
            expected_values = observed[: observed.index(held)]
            if self.source_group_values != expected_values:
                raise ValueError(
                    "prior authority source groups are not the complete chronological prefix"
                )
        elif self.history_policy == "all-other":
            expected_values = tuple(v for v in observed if v not in self.held_out_values)
            if self.source_group_values != expected_values:
                raise ValueError("all-other authority does not contain every non-held-out group")

        return split
