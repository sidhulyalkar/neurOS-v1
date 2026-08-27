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
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, Mapping

import numpy as np

from .benchmark import SplitUnit
from .longitudinal import NestedCalibrationSplit, ordered_group_values
from .real_world import EvaluationPartition, GroupedEvaluationData

HistoryPolicy = Literal["prior", "all-other", "custom"]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("longitudinal authority cannot contain NaN or infinity")
        return value
    if isinstance(value, np.generic):
        return _canonical_json(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise TypeError("longitudinal authority cannot contain object arrays")
        return _canonical_json(value.tolist())
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            normalized_key = str(key)
            if not normalized_key.strip():
                raise ValueError("longitudinal authority mapping keys must be non-empty")
            if normalized_key in normalized:
                raise ValueError(
                    "longitudinal authority mapping keys collide after string normalization"
                )
            normalized[normalized_key] = _canonical_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered sets are not valid longitudinal authority values")
    raise TypeError(
        "longitudinal authority values must be deterministic JSON-compatible values; "
        f"got {type(value).__name__}"
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


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        _canonical_json(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_sha256(name: str, value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a lowercase SHA-256 string")
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 hex digest")
    return normalized


def _exact_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be an integer without coercion")
    number = int(value)
    if minimum is not None and number < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return number


def _finite_float(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{name} must be numeric without string coercion")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


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


def _indices_tuple(name: str, values: Any) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an iterable of integer indices")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of integer indices") from exc
    result = tuple(_exact_int(name, value, minimum=0) for value in raw)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate indices")
    return result


def _string_tuple(name: str, values: Any, *, allow_empty: bool = True) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if any(not isinstance(value, str) or not value.strip() for value in raw):
        raise ValueError(f"{name} must contain non-empty strings")
    result = tuple(value.strip() for value in raw)
    if not allow_empty and not result:
        raise ValueError(f"{name} must be non-empty")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicates")
    return result


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
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("LongitudinalCaseAuthority schema_version must be 1")
        if not isinstance(self.dataset_id, str) or not self.dataset_id.strip():
            raise ValueError("dataset_id must be a non-empty string")
        if not isinstance(self.case_id, str) or not self.case_id.strip():
            raise ValueError("case_id must be a non-empty string")
        if not isinstance(self.split_unit, str) or not self.split_unit.strip():
            raise ValueError("split_unit must be a non-empty string")
        if self.split_unit == "sample":
            raise ValueError("longitudinal authority requires a deployment-unit split")
        held_out = _string_tuple("held_out_values", self.held_out_values, allow_empty=False)
        observed = _string_tuple("observed_group_order", self.observed_group_order, allow_empty=False)
        sources = _string_tuple("source_group_values", self.source_group_values)
        if self.history_policy not in {"prior", "all-other", "custom"}:
            raise ValueError(f"unsupported history_policy={self.history_policy!r}")
        n_samples = _exact_int("n_samples", self.n_samples, minimum=1)
        seed = _exact_int("seed", self.seed, minimum=0)
        fraction = _finite_float("evaluation_fraction", self.evaluation_fraction)
        if not 0.0 < fraction < 1.0:
            raise ValueError("evaluation_fraction must lie strictly between 0 and 1")
        source_indices = _indices_tuple("source_train_indices", self.source_train_indices)
        evaluation_indices = _indices_tuple("evaluation_indices", self.evaluation_indices)
        if not evaluation_indices:
            raise ValueError("evaluation_indices must be non-empty")

        if not isinstance(self.calibration_order_by_class, Mapping):
            raise ValueError("calibration_order_by_class must be a mapping")
        calibration: dict[str, tuple[int, ...]] = {}
        for raw_label, values in self.calibration_order_by_class.items():
            if not isinstance(raw_label, str) or not raw_label.strip():
                raise ValueError("calibration class labels must be non-empty strings")
            label = raw_label.strip()
            if label in calibration:
                raise ValueError("calibration class labels cannot duplicate after normalization")
            calibration[label] = _indices_tuple(
                f"calibration_order_by_class[{label!r}]", values
            )
        if not calibration:
            raise ValueError("calibration_order_by_class must be non-empty")

        calibration_flat = [value for values in calibration.values() for value in values]
        if len(set(calibration_flat)) != len(calibration_flat):
            raise ValueError("calibration indices cannot be shared between classes")
        source_set = set(source_indices)
        evaluation_set = set(evaluation_indices)
        calibration_set = set(calibration_flat)
        if source_set & evaluation_set:
            raise ValueError("source and evaluation indices must be disjoint")
        if source_set & calibration_set:
            raise ValueError("source and calibration indices must be disjoint")
        if evaluation_set & calibration_set:
            raise ValueError("evaluation and calibration indices must be disjoint")
        if any(value >= n_samples for value in (*source_indices, *evaluation_indices, *calibration_flat)):
            raise ValueError("longitudinal authority contains out-of-range indices")

        shape = _indices_tuple("input_shape", self.input_shape)
        if not shape or shape[0] != n_samples or any(value < 1 for value in shape):
            raise ValueError("input_shape must contain positive dimensions and begin with n_samples")
        partition = self.partition_fingerprint
        calibration_fp = self.calibration_split_fingerprint
        if not isinstance(partition, str) or not partition.strip():
            raise ValueError("partition_fingerprint must be non-empty")
        if not isinstance(calibration_fp, str) or not calibration_fp.strip():
            raise ValueError("calibration_split_fingerprint must be non-empty")
        processed = _require_sha256("processed_data_sha256", self.processed_data_sha256)
        metadata = _freeze_json(self.case_metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("case_metadata must be a mapping")

        object.__setattr__(self, "dataset_id", self.dataset_id.strip())
        object.__setattr__(self, "case_id", self.case_id.strip())
        object.__setattr__(self, "held_out_values", held_out)
        object.__setattr__(self, "observed_group_order", observed)
        object.__setattr__(self, "source_group_values", sources)
        object.__setattr__(self, "source_train_indices", source_indices)
        object.__setattr__(self, "evaluation_indices", evaluation_indices)
        object.__setattr__(self, "calibration_order_by_class", MappingProxyType(calibration))
        object.__setattr__(self, "evaluation_fraction", fraction)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "partition_fingerprint", partition.strip())
        object.__setattr__(self, "calibration_split_fingerprint", calibration_fp.strip())
        object.__setattr__(self, "processed_data_sha256", processed)
        object.__setattr__(self, "n_samples", n_samples)
        object.__setattr__(self, "input_shape", shape)
        object.__setattr__(self, "case_metadata", metadata)

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
            source_train_indices=_indices_tuple("source_train_indices", split.source_train_indices),
            evaluation_indices=_indices_tuple("evaluation_indices", split.evaluation_indices),
            calibration_order_by_class={
                label: _indices_tuple(f"calibration_order_by_class[{label!r}]", values)
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
    def authority_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_fingerprint=False))

    @property
    def authority_fingerprint(self) -> str:
        """Display-only prefix retained for backward compatibility."""
        return self.authority_sha256[:16]

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
            "evaluation_fraction": self.evaluation_fraction,
            "seed": self.seed,
            "partition_fingerprint": self.partition_fingerprint,
            "calibration_split_fingerprint": self.calibration_split_fingerprint,
            "processed_data_sha256": self.processed_data_sha256,
            "n_samples": self.n_samples,
            "input_shape": list(self.input_shape),
            "case_metadata": _thaw_json(self.case_metadata),
        }
        if include_fingerprint:
            payload["authority_sha256"] = self.authority_sha256
            payload["authority_fingerprint"] = self.authority_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "LongitudinalCaseAuthority":
        if not isinstance(payload, Mapping):
            raise TypeError("serialized longitudinal authority must be a mapping")
        required_strings = (
            "dataset_id",
            "case_id",
            "split_unit",
            "history_policy",
            "partition_fingerprint",
            "calibration_split_fingerprint",
            "processed_data_sha256",
        )
        for name in required_strings:
            if not isinstance(payload.get(name), str):
                raise ValueError(f"serialized {name} must be a string without coercion")
        calibration_payload = payload.get("calibration_order_by_class")
        if not isinstance(calibration_payload, Mapping):
            raise ValueError("serialized calibration_order_by_class must be a mapping")
        metadata = payload.get("case_metadata", {})
        if not isinstance(metadata, Mapping):
            raise ValueError("serialized case_metadata must be a mapping")

        value = cls(
            dataset_id=payload["dataset_id"],
            case_id=payload["case_id"],
            split_unit=payload["split_unit"],  # type: ignore[arg-type]
            held_out_values=_string_tuple("held_out_values", payload["held_out_values"], allow_empty=False),
            history_policy=payload["history_policy"],  # type: ignore[arg-type]
            observed_group_order=_string_tuple(
                "observed_group_order", payload["observed_group_order"], allow_empty=False
            ),
            source_group_values=_string_tuple("source_group_values", payload["source_group_values"]),
            source_train_indices=_indices_tuple("source_train_indices", payload["source_train_indices"]),
            evaluation_indices=_indices_tuple("evaluation_indices", payload["evaluation_indices"]),
            calibration_order_by_class={
                key: _indices_tuple(f"calibration_order_by_class[{key!r}]", values)
                for key, values in calibration_payload.items()
            },
            evaluation_fraction=_finite_float("evaluation_fraction", payload["evaluation_fraction"]),
            seed=_exact_int("seed", payload["seed"], minimum=0),
            partition_fingerprint=payload["partition_fingerprint"],
            calibration_split_fingerprint=payload["calibration_split_fingerprint"],
            processed_data_sha256=payload["processed_data_sha256"],
            n_samples=_exact_int("n_samples", payload["n_samples"], minimum=1),
            input_shape=_indices_tuple("input_shape", payload["input_shape"]),
            case_metadata=metadata,
            schema_version=_exact_int("schema_version", payload.get("schema_version", 1), minimum=1),
        )
        expected_full = payload.get("authority_sha256")
        if expected_full is not None:
            if _require_sha256("authority_sha256", expected_full) != value.authority_sha256:
                raise ValueError("authority_sha256 does not match serialized content")
        expected_display = payload.get("authority_fingerprint")
        if expected_display is not None:
            if not isinstance(expected_display, str) or expected_display != value.authority_fingerprint:
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
