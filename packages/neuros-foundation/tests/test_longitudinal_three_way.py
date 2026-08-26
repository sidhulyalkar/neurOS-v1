from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.longitudinal_three_way import (
    ThreeWayCalibrationSplit,
    make_three_way_calibration_split,
)
from neuros.foundation_models.real_world import GroupedEvaluationData, hold_out_groups


def _fixture_data() -> GroupedEvaluationData:
    rng = np.random.default_rng(71)
    X = []
    y = []
    metadata = []
    for session in ("0", "1", "2"):
        for label in ("left", "right"):
            for trial in range(12):
                X.append(rng.normal(size=(4, 16)))
                y.append(label)
                metadata.append(
                    {
                        "subject": "1",
                        "session": session,
                        "run": f"run-{trial // 6}",
                    }
                )
    return GroupedEvaluationData.from_moabb_result(
        (np.asarray(X), np.asarray(y), metadata), dataset_id="three-way-fixture"
    )


def _partition():
    return hold_out_groups(_fixture_data(), split_unit="session", held_out_values=["2"])


def test_three_way_split_freezes_distinct_qualification_and_final_sets():
    split = make_three_way_calibration_split(
        _partition(),
        qualification_fraction=0.25,
        final_assessment_fraction=0.25,
        seed=7,
    )

    assert len(split.source_train_indices) == 48
    assert len(split.qualification_indices) == 6
    assert len(split.final_assessment_indices) == 6
    assert split.max_budget_per_class == 6

    assert split.qualification_indices.flags.writeable is False
    assert split.final_assessment_indices.flags.writeable is False
    for values in split.calibration_order_by_class.values():
        assert values.flags.writeable is False

    q = set(split.qualification_indices.tolist())
    final = set(split.final_assessment_indices.tolist())
    calibration_pool = set(
        np.concatenate(list(split.calibration_order_by_class.values())).tolist()
    )
    source = set(split.source_train_indices.tolist())
    held = set(split.partition.test_indices.tolist())

    assert q.isdisjoint(final)
    assert q.isdisjoint(calibration_pool)
    assert final.isdisjoint(calibration_pool)
    assert source.isdisjoint(q | final | calibration_pool)
    assert q | final | calibration_pool == held

    y = split.partition.data.y.astype(str)
    assert set(y[split.qualification_indices]) == {"left", "right"}
    assert set(y[split.final_assessment_indices]) == {"left", "right"}
    assert np.sum(y[split.qualification_indices] == "left") == 3
    assert np.sum(y[split.final_assessment_indices] == "left") == 3


def test_calibration_budgets_remain_nested_without_touching_held_out_sets():
    split = make_three_way_calibration_split(_partition(), seed=11)
    qualification_identity = split.qualification_indices.copy()
    final_identity = split.final_assessment_indices.copy()

    zero = set(split.calibration_indices(0).tolist())
    one = set(split.calibration_indices(1).tolist())
    three = set(split.calibration_indices(3).tolist())
    six = set(split.calibration_indices(6).tolist())

    assert zero == set()
    assert one < three < six
    assert len(one) == 2
    assert len(three) == 6
    assert len(six) == 12

    for budget in (0, 1, 3, 6):
        calibration = split.calibration_indices(budget)
        training = split.train_indices_for_budget(budget)
        assert np.array_equal(split.qualification_indices, qualification_identity)
        assert np.array_equal(split.final_assessment_indices, final_identity)
        assert np.intersect1d(calibration, split.qualification_indices).size == 0
        assert np.intersect1d(calibration, split.final_assessment_indices).size == 0
        assert np.intersect1d(training, split.qualification_indices).size == 0
        assert np.intersect1d(training, split.final_assessment_indices).size == 0


def test_three_way_fingerprint_and_manifest_are_deterministic():
    partition = _partition()
    first = make_three_way_calibration_split(partition, seed=13)
    second = make_three_way_calibration_split(partition, seed=13)
    changed = make_three_way_calibration_split(partition, seed=14)

    assert first.fingerprint == second.fingerprint
    assert len(first.fingerprint) == 64
    assert first.fingerprint != changed.fingerprint
    assert np.array_equal(first.qualification_indices, second.qualification_indices)
    assert np.array_equal(first.final_assessment_indices, second.final_assessment_indices)

    manifest = first.manifest()
    assert manifest["schema_version"] == 2
    assert manifest["kind"] == "three_way_calibration_split"
    assert manifest["source_train_samples"] == 48
    assert manifest["calibration_pool_samples"] == 12
    assert manifest["qualification_samples"] == 6
    assert manifest["final_assessment_samples"] == 6
    assert manifest["calibration_pool_by_class"] == {"left": 6, "right": 6}
    assert manifest["qualification_by_class"] == {"left": 3, "right": 3}
    assert manifest["final_assessment_by_class"] == {"left": 3, "right": 3}
    assert manifest["three_way_split_fingerprint"] == first.fingerprint
    assert "final-assessment rows are not part of state-selection authority" in manifest[
        "invariants"
    ]


def test_small_classes_fail_closed_or_produce_zero_calibration_budget():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(8, 2, 4))
    y = np.asarray(["left", "right"] * 2 + ["left", "left", "right", "right"])
    metadata = [
        {"subject": "1", "session": "source", "run": f"s-{i}"}
        for i in range(4)
    ] + [
        {"subject": "1", "session": "held", "run": f"h-{i}"}
        for i in range(4)
    ]
    data = GroupedEvaluationData.from_moabb_result((X, y, metadata), dataset_id="tiny")
    partition = hold_out_groups(data, split_unit="session", held_out_values=["held"])
    split = make_three_way_calibration_split(partition, seed=1)
    assert split.max_budget_per_class == 0
    assert split.calibration_indices(0).size == 0
    with pytest.raises(ValueError, match="balanced maximum"):
        split.calibration_indices(1)

    X_single = rng.normal(size=(6, 2, 4))
    y_single = np.asarray(["left", "right", "left", "right", "left", "right"])
    metadata_single = [
        {"subject": "1", "session": "source", "run": "s0"},
        {"subject": "1", "session": "source", "run": "s1"},
        {"subject": "1", "session": "source", "run": "s2"},
        {"subject": "1", "session": "source", "run": "s3"},
        {"subject": "1", "session": "held", "run": "h0"},
        {"subject": "1", "session": "held", "run": "h1"},
    ]
    data_single = GroupedEvaluationData.from_moabb_result(
        (X_single, y_single, metadata_single), dataset_id="one-per-class"
    )
    held_single = hold_out_groups(
        data_single, split_unit="session", held_out_values=["held"]
    )
    with pytest.raises(ValueError, match="enough rows"):
        make_three_way_calibration_split(held_single, seed=1)


def test_invalid_fractions_and_budgets_fail_closed():
    partition = _partition()
    for invalid in (0.0, 1.0, -0.1, 1.1, np.nan):
        with pytest.raises(ValueError, match="strictly between"):
            make_three_way_calibration_split(
                partition,
                qualification_fraction=invalid,
            )
    with pytest.raises(ValueError, match="strictly less than 1"):
        make_three_way_calibration_split(
            partition,
            qualification_fraction=0.5,
            final_assessment_fraction=0.5,
        )

    split = make_three_way_calibration_split(partition)
    with pytest.raises(ValueError, match="non-negative"):
        split.calibration_indices(-1)
    with pytest.raises(ValueError, match="integer"):
        split.calibration_indices(1.5)  # type: ignore[arg-type]


def test_manual_overlap_and_wrong_class_calibration_fail_before_use():
    split = make_three_way_calibration_split(_partition(), seed=5)
    calibration = {
        label: np.array(values, copy=True)
        for label, values in split.calibration_order_by_class.items()
    }

    with pytest.raises(ValueError, match="disjoint"):
        ThreeWayCalibrationSplit(
            partition=split.partition,
            qualification_indices=split.qualification_indices,
            final_assessment_indices=split.qualification_indices,
            calibration_order_by_class=calibration,
            qualification_fraction=split.qualification_fraction,
            final_assessment_fraction=split.final_assessment_fraction,
            seed=split.seed,
        )

    left = calibration["left"].copy()
    right = calibration["right"].copy()
    left[0], right[0] = right[0], left[0]
    with pytest.raises(ValueError, match="another class"):
        ThreeWayCalibrationSplit(
            partition=split.partition,
            qualification_indices=split.qualification_indices,
            final_assessment_indices=split.final_assessment_indices,
            calibration_order_by_class={"left": left, "right": right},
            qualification_fraction=split.qualification_fraction,
            final_assessment_fraction=split.final_assessment_fraction,
            seed=split.seed,
        )
