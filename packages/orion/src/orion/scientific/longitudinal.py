"""Bridge frozen longitudinal benchmark cases into ORION scientific authority."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .common import ObservationRole, canonical_sha256, require_sha256
from .lineage import DatasetLineage
from .observations import ObservationSetAuthority, TargetObservationBudget


def _strict_indices(name: str, values: Any) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of integer indices")
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of integer indices") from exc
    result: list[int] = []
    for value in raw:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must contain actual integers without coercion")
        if value < 0:
            raise ValueError(f"{name} cannot contain negative indices")
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate indices")
    return tuple(result)


def _strict_nonempty_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string without coercion")
    return value.strip()


def _source_case_sha256(case_payload: Mapping[str, Any]) -> tuple[str, str | None]:
    source = dict(case_payload)
    legacy_display = source.pop("authority_fingerprint", None)
    supplied_full = source.pop("authority_sha256", None)
    full = canonical_sha256(source)
    if supplied_full is not None and require_sha256("authority_sha256", supplied_full) != full:
        raise ValueError("serialized longitudinal authority_sha256 does not match its content")
    if legacy_display is not None:
        legacy_display = _strict_nonempty_string(
            "legacy authority_fingerprint", legacy_display
        )
    return full, legacy_display


def bind_longitudinal_case_authority(
    case_payload: Mapping[str, Any],
    *,
    dataset_lineage: DatasetLineage,
    calibration_per_class: int,
    unlabeled_target_observation_indices: Sequence[int] = (),
    unlabeled_target_seconds: float | None = None,
) -> tuple[tuple[ObservationSetAuthority, ...], TargetObservationBudget]:
    """Bind a serialized #26/#27 case to explicit ORION observation roles.

    The frozen longitudinal split remains authoritative. This function does not
    regenerate indices. It derives a full SHA-256 over the serialized case
    authority and adds information-role governance around that exact authority.
    """

    if not isinstance(case_payload, Mapping):
        raise TypeError("case_payload must be a mapping")
    if not isinstance(dataset_lineage, DatasetLineage):
        raise TypeError("dataset_lineage must be DatasetLineage")
    dataset_id = _strict_nonempty_string("dataset_id", case_payload.get("dataset_id"))
    if dataset_id != dataset_lineage.dataset_id:
        raise ValueError("longitudinal case dataset_id does not match dataset lineage")
    if (
        isinstance(calibration_per_class, bool)
        or not isinstance(calibration_per_class, int)
        or calibration_per_class < 0
    ):
        raise ValueError("calibration_per_class must be a non-negative integer")

    case_id = _strict_nonempty_string("case_id", case_payload.get("case_id"))

    source_authority_sha256, legacy_display = _source_case_sha256(case_payload)
    source_indices = _strict_indices("source_train_indices", case_payload["source_train_indices"])
    evaluation_indices = _strict_indices("evaluation_indices", case_payload["evaluation_indices"])

    raw_calibration = case_payload["calibration_order_by_class"]
    if not isinstance(raw_calibration, Mapping) or not raw_calibration:
        raise ValueError("calibration_order_by_class must be a non-empty mapping")
    calibration_indices: list[int] = []
    seen_labels: set[str] = set()
    for raw_label in sorted(raw_calibration, key=lambda value: str(value)):
        label = _strict_nonempty_string("calibration class label", raw_label)
        if label in seen_labels:
            raise ValueError("calibration class labels cannot collide after normalization")
        seen_labels.add(label)
        ordered = _strict_indices(
            f"calibration_order_by_class[{label!r}]", raw_calibration[raw_label]
        )
        if len(ordered) < calibration_per_class:
            raise ValueError(
                f"case {case_id!r} has fewer than {calibration_per_class} calibration rows "
                f"for class {label!r}"
            )
        calibration_indices.extend(ordered[:calibration_per_class])

    calibration = tuple(calibration_indices)
    if len(set(calibration)) != len(calibration):
        raise ValueError("calibration rows cannot be shared between declared classes")
    unlabeled = _strict_indices(
        "unlabeled_target_observation_indices", unlabeled_target_observation_indices
    )

    n_samples = case_payload.get("n_samples")
    if n_samples is not None:
        if isinstance(n_samples, bool) or not isinstance(n_samples, int) or n_samples < 1:
            raise ValueError("n_samples must be a positive integer when present")
        for name, values in (
            ("source", source_indices),
            ("evaluation", evaluation_indices),
            ("calibration", calibration),
            ("unlabeled target", unlabeled),
        ):
            if values and max(values) >= n_samples:
                raise ValueError(f"{name} indices exceed longitudinal authority n_samples")

    source_set = set(source_indices)
    evaluation_set = set(evaluation_indices)
    calibration_set = set(calibration)
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
    if unlabeled_set & source_set:
        raise ValueError("unlabeled target observations cannot borrow source-history rows")

    budget = TargetObservationBudget(
        labeled_examples=len(calibration),
        labeled_examples_per_class=calibration_per_class,
        unlabeled_examples=len(unlabeled),
        unlabeled_seconds=unlabeled_target_seconds,
    )

    processed_sha = case_payload.get("processed_data_sha256")
    if processed_sha is not None:
        processed_sha = require_sha256("processed_data_sha256", processed_sha)

    domain = f"{dataset_lineage.dataset_id}:{case_id}"
    common_metadata = {
        "source_authority_sha256": source_authority_sha256,
        "legacy_source_authority_fingerprint": legacy_display,
        "partition_fingerprint": case_payload.get("partition_fingerprint"),
        "calibration_split_fingerprint": case_payload.get("calibration_split_fingerprint"),
        "processed_data_sha256": processed_sha,
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
            observation_ids=tuple(str(value) for value in calibration),
            domain_id=domain,
            metadata={
                **common_metadata,
                "calibration_per_class": calibration_per_class,
            },
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
    if unlabeled or (budget.unlabeled_seconds or 0.0) > 0.0:
        observations.append(
            ObservationSetAuthority(
                authority_id=f"{case_id}:unlabeled-target",
                dataset_lineage_sha256=dataset_lineage.lineage_sha256,
                role=ObservationRole.UNLABELED_TARGET_OBSERVATION,
                observation_ids=tuple(str(value) for value in unlabeled),
                domain_id=domain,
                metadata={
                    **common_metadata,
                    "unlabeled_seconds": budget.unlabeled_seconds,
                },
            )
        )

    return tuple(observations), budget
