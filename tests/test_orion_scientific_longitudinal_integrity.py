from __future__ import annotations

import pytest

from orion.scientific_authority import (
    DatasetLineage,
    LineageCompleteness,
    ObservationRole,
    bind_longitudinal_case_authority,
)

SHA = "a" * 64


def _dataset() -> DatasetLineage:
    return DatasetLineage(
        dataset_id="kumar2024",
        upstream_source="MOABB:Kumar2024",
        lineage_completeness=LineageCompleteness.COMPLETE,
    )


def _payload():
    return {
        "dataset_id": "kumar2024",
        "case_id": "subject-1/session-2",
        "source_train_indices": [0, 1],
        "evaluation_indices": [6, 7],
        "calibration_order_by_class": {
            "left": [2, 4],
            "right": [3, 5],
        },
        "processed_data_sha256": SHA,
        "history_policy": "prior",
        "held_out_values": ["2"],
        "n_samples": 8,
    }


def test_continuous_unlabeled_target_time_creates_consumable_observation_authority():
    observations, budget = bind_longitudinal_case_authority(
        _payload(),
        dataset_lineage=_dataset(),
        calibration_per_class=0,
        unlabeled_target_seconds=2.5,
    )
    unlabeled = [
        item
        for item in observations
        if item.role is ObservationRole.UNLABELED_TARGET_OBSERVATION
    ]
    assert len(unlabeled) == 1
    assert unlabeled[0].observation_ids == ()
    assert unlabeled[0].metadata["unlabeled_seconds"] == 2.5
    assert budget.unlabeled_examples == 0
    assert budget.unlabeled_seconds == 2.5
    assert budget.has_target_information


def test_zero_unlabeled_time_does_not_invent_unlabeled_observation_authority():
    observations, budget = bind_longitudinal_case_authority(
        _payload(),
        dataset_lineage=_dataset(),
        calibration_per_class=0,
        unlabeled_target_seconds=0.0,
    )
    assert all(
        item.role is not ObservationRole.UNLABELED_TARGET_OBSERVATION
        for item in observations
    )
    assert not budget.has_target_information


def test_longitudinal_bridge_does_not_coerce_dataset_or_case_identity():
    bad_dataset = _payload()
    bad_dataset["dataset_id"] = 123
    with pytest.raises(ValueError, match="dataset_id.*without coercion"):
        bind_longitudinal_case_authority(
            bad_dataset,
            dataset_lineage=_dataset(),
            calibration_per_class=0,
        )

    bad_case = _payload()
    bad_case["case_id"] = 123
    with pytest.raises(ValueError, match="case_id.*without coercion"):
        bind_longitudinal_case_authority(
            bad_case,
            dataset_lineage=_dataset(),
            calibration_per_class=0,
        )


def test_longitudinal_bridge_does_not_coerce_class_labels():
    payload = _payload()
    payload["calibration_order_by_class"] = {
        1: [2, 4],
        "right": [3, 5],
    }
    with pytest.raises(ValueError, match="calibration class label.*without coercion"):
        bind_longitudinal_case_authority(
            payload,
            dataset_lineage=_dataset(),
            calibration_per_class=0,
        )
