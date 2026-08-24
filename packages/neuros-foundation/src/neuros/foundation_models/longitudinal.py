"""Longitudinal calibration-budget contracts for real-world neural evaluation.

The core invariant is simple but important: every point on a calibration curve
must be evaluated on the *same held-out examples*. Calibration examples come
from a separately frozen pool inside the held-out deployment unit and budgets
are nested, so a larger budget is a strict superset of a smaller one.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .real_world import EvaluationPartition


@dataclass(frozen=True, slots=True)
class NestedCalibrationSplit:
    """Freeze one source-train / calibration-pool / evaluation partition.

    This contract is intended for classification-style longitudinal studies
    such as motor-imagery EEG. It is deliberately not used for FALCON-style
    continuous decoding, where calibration windows and chronology should follow
    the benchmark's native split semantics.
    """

    partition: EvaluationPartition
    evaluation_indices: np.ndarray
    calibration_order_by_class: Mapping[str, np.ndarray]
    evaluation_fraction: float
    seed: int

    def __post_init__(self) -> None:
        if not 0.0 < float(self.evaluation_fraction) < 1.0:
            raise ValueError("evaluation_fraction must lie strictly between 0 and 1")

        evaluation = np.asarray(self.evaluation_indices, dtype=np.int64).reshape(-1)
        if len(evaluation) == 0:
            raise ValueError("fixed evaluation set must be non-empty")

        test_set = set(self.partition.test_indices.tolist())
        if not set(evaluation.tolist()).issubset(test_set):
            raise ValueError("evaluation indices must come from the held-out partition")
        if np.intersect1d(evaluation, self.partition.train_indices).size:
            raise ValueError("evaluation indices overlap source training data")

        normalized: dict[str, np.ndarray] = {}
        seen_calibration: set[int] = set()
        for label, values in self.calibration_order_by_class.items():
            key = str(label)
            array = np.asarray(values, dtype=np.int64).reshape(-1)
            if not set(array.tolist()).issubset(test_set):
                raise ValueError(f"calibration pool for {key!r} leaves held-out partition")
            if np.intersect1d(array, evaluation).size:
                raise ValueError(f"calibration pool for {key!r} overlaps evaluation")
            if np.intersect1d(array, self.partition.train_indices).size:
                raise ValueError(f"calibration pool for {key!r} overlaps source training")
            duplicate = seen_calibration.intersection(array.tolist())
            if duplicate:
                raise ValueError(f"calibration pools overlap at indices {sorted(duplicate)}")
            seen_calibration.update(array.tolist())
            normalized[key] = array

        if not normalized:
            raise ValueError("at least one class calibration pool is required")

        covered = set(evaluation.tolist()).union(seen_calibration)
        if covered != test_set:
            missing = sorted(test_set - covered)
            extra = sorted(covered - test_set)
            raise ValueError(
                "calibration/evaluation split must cover the held-out partition exactly; "
                f"missing={missing}, extra={extra}"
            )

        object.__setattr__(self, "evaluation_indices", evaluation)
        object.__setattr__(self, "calibration_order_by_class", normalized)

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(sorted(self.calibration_order_by_class))

    @property
    def source_train_indices(self) -> np.ndarray:
        return self.partition.train_indices

    @property
    def max_budget_per_class(self) -> int:
        """Largest balanced per-class budget supported by every class."""
        return min(len(values) for values in self.calibration_order_by_class.values())

    def calibration_indices(self, per_class: int) -> np.ndarray:
        """Return a deterministic nested balanced calibration subset.

        Budgets larger than the smallest class pool fail closed instead of
        silently becoming class-imbalanced.
        """
        budget = int(per_class)
        if budget < 0:
            raise ValueError("calibration budget must be non-negative")
        if budget > self.max_budget_per_class:
            raise ValueError(
                f"budget {budget} exceeds balanced maximum {self.max_budget_per_class}"
            )
        if budget == 0:
            return np.asarray([], dtype=np.int64)
        selected = [
            values[:budget]
            for _, values in sorted(self.calibration_order_by_class.items())
        ]
        return np.sort(np.concatenate(selected).astype(np.int64, copy=False))

    def train_indices_for_budget(self, per_class: int) -> np.ndarray:
        """Historical source data plus the requested held-out calibration data."""
        calibration = self.calibration_indices(per_class)
        if len(calibration) == 0:
            return self.source_train_indices.copy()
        return np.sort(np.concatenate([self.source_train_indices, calibration]))

    @property
    def fingerprint(self) -> str:
        payload = {
            "partition_fingerprint": self.partition.fingerprint,
            "evaluation_indices": self.evaluation_indices.tolist(),
            "calibration_order_by_class": {
                label: values.tolist()
                for label, values in sorted(self.calibration_order_by_class.items())
            },
            "evaluation_fraction": float(self.evaluation_fraction),
            "seed": int(self.seed),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def manifest(self) -> dict[str, Any]:
        y = np.asarray(self.partition.data.y)
        return {
            "schema_version": 1,
            "kind": "nested_calibration_split",
            "dataset_id": self.partition.data.dataset_id,
            "split_unit": self.partition.split_unit,
            "held_out_values": list(self.partition.held_out_values),
            "source_train_samples": int(len(self.source_train_indices)),
            "calibration_pool_samples": int(
                sum(len(values) for values in self.calibration_order_by_class.values())
            ),
            "evaluation_samples": int(len(self.evaluation_indices)),
            "evaluation_fraction_requested": float(self.evaluation_fraction),
            "max_balanced_budget_per_class": int(self.max_budget_per_class),
            "calibration_pool_by_class": {
                label: int(len(values))
                for label, values in sorted(self.calibration_order_by_class.items())
            },
            "evaluation_by_class": {
                label: int(np.sum(y[self.evaluation_indices].astype(str) == label))
                for label in self.labels
            },
            "seed": int(self.seed),
            "partition_fingerprint": self.partition.fingerprint,
            "calibration_split_fingerprint": self.fingerprint,
            "invariants": [
                "evaluation indices are fixed across calibration budgets",
                "calibration budgets are nested within each class",
                "calibration/evaluation indices are disjoint",
                "source training data contains no held-out deployment unit",
            ],
        }


def make_nested_calibration_split(
    partition: EvaluationPartition,
    *,
    evaluation_fraction: float = 0.5,
    seed: int = 0,
) -> NestedCalibrationSplit:
    """Freeze a class-stratified calibration pool and fixed evaluation set.

    At least one held-out example per class is reserved for evaluation. If a
    class contains only one held-out example it contributes no calibration
    examples, making the balanced maximum budget zero. This is preferable to
    silently evaluating on different class support across budgets.
    """
    fraction = float(evaluation_fraction)
    if not 0.0 < fraction < 1.0:
        raise ValueError("evaluation_fraction must lie strictly between 0 and 1")

    y = np.asarray(partition.data.y)
    held_out = partition.test_indices
    held_labels = y[held_out].astype(str)
    labels = tuple(sorted(np.unique(held_labels).tolist()))
    if len(labels) < 2:
        raise ValueError("nested calibration split requires at least two held-out classes")

    rng = np.random.default_rng(int(seed))
    evaluation: list[np.ndarray] = []
    calibration: dict[str, np.ndarray] = {}

    for label in labels:
        class_indices = held_out[held_labels == label].copy()
        rng.shuffle(class_indices)
        n_class = len(class_indices)
        n_evaluation = max(1, int(np.ceil(n_class * fraction)))
        n_evaluation = min(n_class, n_evaluation)
        evaluation.append(class_indices[:n_evaluation])
        calibration[label] = class_indices[n_evaluation:]

    return NestedCalibrationSplit(
        partition=partition,
        evaluation_indices=np.sort(np.concatenate(evaluation)),
        calibration_order_by_class=calibration,
        evaluation_fraction=fraction,
        seed=int(seed),
    )
