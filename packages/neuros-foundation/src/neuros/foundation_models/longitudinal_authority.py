"""Replayable authority for longitudinal neural benchmark cases.

A protocol fingerprint is necessary but not sufficient when multiple methods are
run in separate processes. This module serializes the *actual* source,
calibration, and final-evaluation indices together with a SHA-256 fingerprint of
the processed neural array. A competing method must restore and validate this
authority before it can claim to share the same evidence case.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from .benchmark import SplitUnit
from .longitudinal import NestedCalibrationSplit, ordered_group_values
from .real_world import EvaluationPartition, GroupedEvaluationData

HistoryPolicy = Literal["prior", "all-other", "custom"]


def processed_data_sha256(data: GroupedEvaluationData) -> str:
    """Hash the exact processed neural array plus target/group sample identity.

    This hash is intentionally downstream of MOABB/MNE preprocessing. It does
    not replace the upstream raw-file/version checksum, but it guarantees that
    two method runs consumed byte-identical processed ``X`` values with the same
    row order, labels, and deployment-unit identities.
    """

    x = np.asarray(data.X)
    if x.dtype.hasobject:
        raise TypeError("processed neural arrays with object dtype cannot be fingerprinted")

    digest = hashlib.sha256()
    digest.update(b"neuros.processed-neural-data.v1\0")
    digest.update(str(data.dataset_id).encode("utf-8"))
    digest.update(b"\0")
    digest.update(x.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(json.dumps(list(x.shape), separators=(",", ":")).encode("ascii"))
    digest.update(b"\0")

    # Stream sample-by-sample to avoid materializing a second full dataset copy.
    for sample in x:
        contiguous = np.ascontiguousarray(sample)
        digest.update(memoryview(contiguous).cast("B"))

    identity = {
        "targets": np.asarray(data.y).astype(str).tolist(),
        "groups": {
            key: np.asarray(values).astype(str).tolist()
            for key, values in sorted(data.groups.items())
        },
    }
    digest.update(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return digest.hexdigest()


def _ordered_values(values: np.ndarray) -> tuple[str, ...]:
    return tuple(dict.fromkeys(np.asarray(values).astype(str).tolist()))


def _indices_tuple(values: Any) -> tuple[int, ...]:
    array = np.asarray(values, dtype=np.int64).reshape(-1)
    return tuple(int(value) for value in array.tolist())


@dataclass(frozen=True, slots=True)
class LongitudinalCaseAuthority:
    """Frozen sample authority for one target deployment unit.

    ``source_train_indices`` contains only historical/source examples.
    ``calibration_order_by_class`` contains the ordered target-session pool from
    which every labeled budget is sliced. ``evaluation_indices`` is immutable
    across budgets.
    """

    dataset_id: str
    case_id: str
    split_unit: SplitUnit
    held_out_values: tuple[str, ...]
    history_policy: HistoryPolicy
    observed_group_order: tuple[str, ...]
    source_group_values: tuple[str, ...]
    source_train_indices: tuple[int, ...]
    evaluation_indices: tuple[int, ...]
    calibration_order_by_class: Mapping[str, tuple[int, ...]]
    evaluation_fraction: float
    seed: int
    partition_fingerprint: str
    calibration_split_fingerprint: str
    processed_data_sha256: str
    n_samples: int
    input_shape: tuple[int, ...]
    case_metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.dataset_id or not self.case_id:
            raise ValueError("dataset_id and case_id must be non-empty")
        if self.split_unit == "sample":
            raise ValueError("longitudinal authority requires a deployment-unit split")
        if not self.held_out_values:
            raise ValueError("held_out_values must be non-empty")
        if self.history_policy not in {"prior", "all-other", "custom"}:
            raise ValueError(f"unsupported history_policy={self.history_policy!r}")
        if self.n_samples <= 0:
            raise ValueError("n_samples must be positive")
        if not self.input_shape or self.input_shape[0] != self.n_samples:
            raise ValueError("input_shape must begin with n_samples")
        if len(self.processed_data_sha256) != 64:
            raise ValueError("processed_data_sha256 must be a SHA-256 hex digest")
        if not 0.0 < float(self.evaluation_fraction) < 1.0:
            raise ValueError("evaluation_fraction must lie strictly between 0 and 1")

        calibration = {
            str(label): _indices_tuple(values)
            for label, values in self.calibration_order_by_class.items()
        }
        if not calibration:
            raise ValueError("calibration_order_by_class must be non-empty")
        object.__setattr__(self, "calibration_order_by_class", MappingProxyType(calibration))
        object.__setattr__(self, "case_metadata", MappingProxyType(dict(self.case_metadata)))

    @classmethod
    def from_split(
        cls,
        split: NestedCalibrationSplit,
        *,
        case_id: str,
        history_policy: HistoryPolicy,
        observed_group_order: tuple[str, ...] | None = None,
        case_metadata: Mapping[str, Any] | None = None,
    ) -> "LongitudinalCaseAuthority":
        data = split.partition.data
        unit = split.partition.split_unit
        if unit == "sample":
            raise ValueError("longitudinal authority requires a deployment-unit split")
        group = np.asarray(data.groups[unit]).astype(str)
        observed = observed_group_order or ordered_group_values(data, split_unit=unit)
        source_values = _ordered_values(group[split.source_train_indices])
        return cls(
            dataset_id=data.dataset_id,
            case_id=case_id,
            split_unit=unit,
            held_out_values=tuple(str(v) for v in split.partition.held_out_values),
            history_policy=history_policy,
            observed_group_order=tuple(str(v) for v in observed),
            source_group_values=source_values,
            source_train_indices=_indices_tuple(split.source_train_indices),
            evaluation_indices=_indices_tuple(split.evaluation_indices),
            calibration_order_by_class={
                label: _indices_tuple(values)
                for label, values in split.calibration_order_by_class.items()
            },
            evaluation_fraction=float(split.evaluation_fraction),
            seed=int(split.seed),
            partition_fingerprint=split.partition.fingerprint,
            calibration_split_fingerprint=split.fingerprint,
            processed_data_sha256=processed_data_sha256(data),
            n_samples=int(len(data.X)),
            input_shape=tuple(int(v) for v in np.asarray(data.X).shape),
            case_metadata={} if case_metadata is None else case_metadata,
        )

    @property
    def authority_fingerprint(self) -> str:
        payload = self.to_dict(include_fingerprint=False)
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "dataset_id": self.dataset_id,
            "case_id": self.case_id,
            "split_unit": self.split_unit,
            "held_out_values": list(self.held_out_values),
            "history_policy": self.history_policy,
            "observed_group_order": list(self.observed_group_order),
            "source_group_values": list(self.source_group_values),
            "source_train_indices": list(self.source_train_indices),
            "evaluation_indices": list(self.evaluation_indices),
            "calibration_order_by_class": {
                key: list(values)
                for key, values in sorted(self.calibration_order_by_class.items())
            },
            "evaluation_fraction": float(self.evaluation_fraction),
            "seed": int(self.seed),
            "partition_fingerprint": self.partition_fingerprint,
            "calibration_split_fingerprint": self.calibration_split_fingerprint,
            "processed_data_sha256": self.processed_data_sha256,
            "n_samples": int(self.n_samples),
            "input_shape": list(self.input_shape),
            "case_metadata": dict(self.case_metadata),
        }
        if include_fingerprint:
            payload["authority_fingerprint"] = self.authority_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LongitudinalCaseAuthority":
        value = cls(
            dataset_id=str(payload["dataset_id"]),
            case_id=str(payload["case_id"]),
            split_unit=str(payload["split_unit"]),  # type: ignore[arg-type]
            held_out_values=tuple(str(v) for v in payload["held_out_values"]),
            history_policy=str(payload["history_policy"]),  # type: ignore[arg-type]
            observed_group_order=tuple(str(v) for v in payload["observed_group_order"]),
            source_group_values=tuple(str(v) for v in payload["source_group_values"]),
            source_train_indices=tuple(int(v) for v in payload["source_train_indices"]),
            evaluation_indices=tuple(int(v) for v in payload["evaluation_indices"]),
            calibration_order_by_class={
                str(key): tuple(int(v) for v in values)
                for key, values in dict(payload["calibration_order_by_class"]).items()
            },
            evaluation_fraction=float(payload["evaluation_fraction"]),
            seed=int(payload["seed"]),
            partition_fingerprint=str(payload["partition_fingerprint"]),
            calibration_split_fingerprint=str(payload["calibration_split_fingerprint"]),
            processed_data_sha256=str(payload["processed_data_sha256"]),
            n_samples=int(payload["n_samples"]),
            input_shape=tuple(int(v) for v in payload["input_shape"]),
            case_metadata=dict(payload.get("case_metadata", {})),
            schema_version=int(payload.get("schema_version", 1)),
        )
        expected = payload.get("authority_fingerprint")
        if expected is not None and str(expected) != value.authority_fingerprint:
            raise ValueError("authority_fingerprint does not match serialized content")
        return value

    def restore(self, data: GroupedEvaluationData) -> NestedCalibrationSplit:
        """Validate a newly loaded dataset and reconstruct the frozen split."""
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
        evaluation = np.asarray(self.evaluation_indices, dtype=np.int64)
        calibration = {
            label: np.asarray(values, dtype=np.int64)
            for label, values in self.calibration_order_by_class.items()
        }
        all_indices = [source, evaluation, *calibration.values()]
        if any(np.any(values < 0) or np.any(values >= n) for values in all_indices):
            raise ValueError("authority contains out-of-range sample indices")

        calibration_flat = (
            np.concatenate(list(calibration.values()))
            if calibration
            else np.asarray([], dtype=np.int64)
        )
        test = np.sort(np.concatenate([evaluation, calibration_flat]))
        partition = EvaluationPartition(
            data=data,
            split_unit=self.split_unit,
            train_indices=source,
            test_indices=test,
            held_out_values=self.held_out_values,
        )
        if partition.fingerprint != self.partition_fingerprint:
            raise ValueError("partition fingerprint differs from authority")

        split = NestedCalibrationSplit(
            partition=partition,
            evaluation_indices=evaluation,
            calibration_order_by_class=calibration,
            evaluation_fraction=self.evaluation_fraction,
            seed=self.seed,
        )
        if split.fingerprint != self.calibration_split_fingerprint:
            raise ValueError("calibration split fingerprint differs from authority")

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
