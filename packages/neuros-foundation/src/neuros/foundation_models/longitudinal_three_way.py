"""Three-way target-session authority for unbiased adaptive longitudinal studies.

This module is additive to the v1 ``NestedCalibrationSplit`` contract. It keeps
calibration budgets nested while separating two held-out roles:

* ``qualification_indices`` may be used for retain/rollback or other frozen
  state-selection policy;
* ``final_assessment_indices`` must remain untouched until model state and
  selection policy are frozen.

The distinction prevents state-selection data from being reported later as an
untouched final test set.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np

from .real_world import EvaluationPartition


def _fraction(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or not 0.0 < result < 1.0:
        raise ValueError(f"{name} must lie strictly between 0 and 1")
    return result


def _seed(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


def _readonly_indices(name: str, values: Any, *, allow_empty: bool) -> np.ndarray:
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
    if not allow_empty and integer.size == 0:
        raise ValueError(f"{name} must be non-empty")
    if np.any(integer < 0):
        raise ValueError(f"{name} cannot contain negative indices")
    if len(set(int(v) for v in integer.tolist())) != integer.size:
        raise ValueError(f"{name} cannot contain duplicate indices")
    result = np.ascontiguousarray(integer.copy())
    result.setflags(write=False)
    return result


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True)
class ThreeWayCalibrationSplit:
    """Frozen source/calibration/qualification/final-assessment partition.

    Calibration order remains randomized per class so balanced budgets are
    nested. Qualification and final-assessment rows are immutable across every
    calibration budget.
    """

    partition: EvaluationPartition
    qualification_indices: np.ndarray
    final_assessment_indices: np.ndarray
    calibration_order_by_class: Mapping[str, np.ndarray]
    qualification_fraction: float
    final_assessment_fraction: float
    seed: int
    schema_version: int = 2

    def __post_init__(self) -> None:
        if self.schema_version != 2:
            raise ValueError("ThreeWayCalibrationSplit schema_version must be 2")
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
        seed = _seed(self.seed)

        qualification = _readonly_indices(
            "qualification_indices", self.qualification_indices, allow_empty=False
        )
        final_assessment = _readonly_indices(
            "final_assessment_indices", self.final_assessment_indices, allow_empty=False
        )
        source = _readonly_indices(
            "source_train_indices", self.partition.train_indices, allow_empty=False
        )
        held_out = _readonly_indices(
            "held_out_indices", self.partition.test_indices, allow_empty=False
        )

        test_set = set(int(v) for v in held_out.tolist())
        source_set = set(int(v) for v in source.tolist())
        qualification_set = set(int(v) for v in qualification.tolist())
        final_set = set(int(v) for v in final_assessment.tolist())

        if not qualification_set.issubset(test_set):
            raise ValueError("qualification indices must come from the held-out partition")
        if not final_set.issubset(test_set):
            raise ValueError("final-assessment indices must come from the held-out partition")
        if qualification_set & final_set:
            raise ValueError("qualification and final-assessment indices must be disjoint")
        if qualification_set & source_set or final_set & source_set:
            raise ValueError("held-out authority overlaps source training data")

        y = np.asarray(self.partition.data.y).astype(str)
        held_labels = tuple(sorted(np.unique(y[held_out]).tolist()))
        if len(held_labels) < 2:
            raise ValueError("three-way calibration split requires at least two held-out classes")

        qualification_labels = set(y[qualification].tolist())
        final_labels = set(y[final_assessment].tolist())
        if qualification_labels != set(held_labels):
            raise ValueError(
                "qualification set must preserve complete held-out class support"
            )
        if final_labels != set(held_labels):
            raise ValueError(
                "final-assessment set must preserve complete held-out class support"
            )

        raw_mapping = {str(label): values for label, values in self.calibration_order_by_class.items()}
        if set(raw_mapping) != set(held_labels):
            missing = sorted(set(held_labels) - set(raw_mapping))
            extra = sorted(set(raw_mapping) - set(held_labels))
            raise ValueError(
                "calibration pools must contain every held-out class exactly once; "
                f"missing={missing}, extra={extra}"
            )

        normalized: dict[str, np.ndarray] = {}
        seen_calibration: set[int] = set()
        for label in held_labels:
            array = _readonly_indices(
                f"calibration_order_by_class[{label!r}]",
                raw_mapping[label],
                allow_empty=True,
            )
            array_set = set(int(v) for v in array.tolist())
            if not array_set.issubset(test_set):
                raise ValueError(f"calibration pool for {label!r} leaves held-out partition")
            if array_set & source_set:
                raise ValueError(f"calibration pool for {label!r} overlaps source training")
            if array_set & qualification_set:
                raise ValueError(f"calibration pool for {label!r} overlaps qualification")
            if array_set & final_set:
                raise ValueError(f"calibration pool for {label!r} overlaps final assessment")
            duplicate = seen_calibration & array_set
            if duplicate:
                raise ValueError(f"calibration pools overlap at indices {sorted(duplicate)}")
            if array.size and not np.all(y[array] == label):
                raise ValueError(
                    f"calibration pool for {label!r} contains samples from another class"
                )
            seen_calibration.update(array_set)
            normalized[label] = array

        covered = qualification_set | final_set | seen_calibration
        if covered != test_set:
            missing = sorted(test_set - covered)
            extra = sorted(covered - test_set)
            raise ValueError(
                "calibration/qualification/final split must cover held-out partition exactly; "
                f"missing={missing}, extra={extra}"
            )

        object.__setattr__(self, "qualification_indices", qualification)
        object.__setattr__(self, "final_assessment_indices", final_assessment)
        object.__setattr__(self, "calibration_order_by_class", MappingProxyType(normalized))
        object.__setattr__(self, "qualification_fraction", qualification_fraction)
        object.__setattr__(self, "final_assessment_fraction", final_fraction)
        object.__setattr__(self, "seed", seed)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.calibration_order_by_class))

    @property
    def source_train_indices(self) -> np.ndarray:
        values = np.ascontiguousarray(np.asarray(self.partition.train_indices, dtype=np.int64)).copy()
        values.setflags(write=False)
        return values

    @property
    def max_budget_per_class(self) -> int:
        """Largest balanced calibration budget supported by every held-out class."""
        return min(len(values) for values in self.calibration_order_by_class.values())

    def calibration_indices(self, per_class: int) -> np.ndarray:
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
            result = np.asarray([], dtype=np.int64)
        else:
            selected = [
                values[:budget]
                for _, values in sorted(self.calibration_order_by_class.items())
            ]
            result = np.sort(np.concatenate(selected).astype(np.int64, copy=False))
        result = np.ascontiguousarray(result)
        result.setflags(write=False)
        return result

    def train_indices_for_budget(self, per_class: int) -> np.ndarray:
        calibration = self.calibration_indices(per_class)
        if len(calibration) == 0:
            result = self.source_train_indices.copy()
        else:
            result = np.sort(
                np.concatenate([self.source_train_indices, calibration]).astype(
                    np.int64, copy=False
                )
            )
        result = np.ascontiguousarray(result)
        result.setflags(write=False)
        return result

    @property
    def fingerprint(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "partition_fingerprint": self.partition.fingerprint,
            "qualification_indices": self.qualification_indices.tolist(),
            "final_assessment_indices": self.final_assessment_indices.tolist(),
            "calibration_order_by_class": {
                label: values.tolist()
                for label, values in sorted(self.calibration_order_by_class.items())
            },
            "qualification_fraction": self.qualification_fraction,
            "final_assessment_fraction": self.final_assessment_fraction,
            "seed": self.seed,
        }
        return _canonical_sha256(payload)

    def manifest(self) -> dict[str, Any]:
        y = np.asarray(self.partition.data.y).astype(str)
        return {
            "schema_version": self.schema_version,
            "kind": "three_way_calibration_split",
            "dataset_id": self.partition.data.dataset_id,
            "split_unit": self.partition.split_unit,
            "held_out_values": list(self.partition.held_out_values),
            "source_train_samples": int(len(self.source_train_indices)),
            "calibration_pool_samples": int(
                sum(len(values) for values in self.calibration_order_by_class.values())
            ),
            "qualification_samples": int(len(self.qualification_indices)),
            "final_assessment_samples": int(len(self.final_assessment_indices)),
            "qualification_fraction_requested": self.qualification_fraction,
            "final_assessment_fraction_requested": self.final_assessment_fraction,
            "max_balanced_budget_per_class": self.max_budget_per_class,
            "calibration_pool_by_class": {
                label: int(len(values))
                for label, values in sorted(self.calibration_order_by_class.items())
            },
            "qualification_by_class": {
                label: int(np.sum(y[self.qualification_indices] == label))
                for label in self.labels
            },
            "final_assessment_by_class": {
                label: int(np.sum(y[self.final_assessment_indices] == label))
                for label in self.labels
            },
            "seed": self.seed,
            "partition_fingerprint": self.partition.fingerprint,
            "three_way_split_fingerprint": self.fingerprint,
            "invariants": [
                "source training contains no held-out deployment unit",
                "calibration budgets are balanced and nested within each class",
                "qualification indices are fixed across calibration budgets",
                "final-assessment indices are fixed across calibration budgets",
                "calibration, qualification, and final-assessment indices are pairwise disjoint",
                "qualification and final-assessment sets preserve complete held-out class support",
                "final-assessment rows are not part of state-selection authority",
            ],
        }


def make_three_way_calibration_split(
    partition: EvaluationPartition,
    *,
    qualification_fraction: float = 0.25,
    final_assessment_fraction: float = 0.25,
    seed: int = 0,
) -> ThreeWayCalibrationSplit:
    """Freeze calibration, qualification, and untouched final-assessment rows.

    Fractions are interpreted against each held-out class independently. At
    least one row per class is reserved for qualification and one for final
    assessment. Small classes may therefore leave a zero-sized calibration
    pool; the balanced maximum budget then becomes zero rather than silently
    changing class support.
    """

    q_fraction = _fraction("qualification_fraction", qualification_fraction)
    f_fraction = _fraction("final_assessment_fraction", final_assessment_fraction)
    if q_fraction + f_fraction >= 1.0:
        raise ValueError(
            "qualification_fraction + final_assessment_fraction must be strictly less than 1"
        )
    normalized_seed = _seed(seed)

    y = np.asarray(partition.data.y).astype(str)
    held_out = np.asarray(partition.test_indices, dtype=np.int64)
    held_labels = y[held_out]
    labels = tuple(sorted(np.unique(held_labels).tolist()))
    if len(labels) < 2:
        raise ValueError("three-way calibration split requires at least two held-out classes")

    rng = np.random.default_rng(normalized_seed)
    qualification: list[np.ndarray] = []
    final_assessment: list[np.ndarray] = []
    calibration: dict[str, np.ndarray] = {}

    for label in labels:
        class_indices = held_out[held_labels == label].copy()
        rng.shuffle(class_indices)
        n_class = int(len(class_indices))
        n_qualification = max(1, int(np.ceil(n_class * q_fraction)))
        n_final = max(1, int(np.ceil(n_class * f_fraction)))
        if n_qualification + n_final > n_class:
            raise ValueError(
                "held-out class does not contain enough rows for separate qualification "
                f"and final assessment: label={label!r}, n={n_class}, "
                f"qualification={n_qualification}, final={n_final}"
            )

        # Reserve the final-assessment subset first. It never becomes eligible
        # for calibration or retain/rollback qualification.
        final_assessment.append(class_indices[:n_final])
        qualification.append(class_indices[n_final : n_final + n_qualification])
        calibration[label] = class_indices[n_final + n_qualification :]

    return ThreeWayCalibrationSplit(
        partition=partition,
        qualification_indices=np.sort(np.concatenate(qualification)),
        final_assessment_indices=np.sort(np.concatenate(final_assessment)),
        calibration_order_by_class=calibration,
        qualification_fraction=q_fraction,
        final_assessment_fraction=f_fraction,
        seed=normalized_seed,
    )
