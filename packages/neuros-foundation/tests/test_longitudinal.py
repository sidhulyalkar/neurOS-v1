from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models import (
    GroupedEvaluationData,
    chronological_partition,
    hold_out_groups,
    make_nested_calibration_split,
    ordered_group_values,
)


def _fixture_data():
    rng = np.random.default_rng(41)
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
        (np.asarray(X), np.asarray(y), metadata),
        dataset_id="fixture",
    )


def _fixture_partition():
    return hold_out_groups(_fixture_data(), split_unit="session", held_out_values=["2"])


def test_chronological_partition_uses_prior_only_and_excludes_future():
    data = _fixture_data()
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="1",
    )

    sessions = np.asarray(data.groups["session"])
    assert set(sessions[partition.train_indices]) == {"0"}
    assert set(sessions[partition.test_indices]) == {"1"}
    assert len(partition.train_indices) == 24
    assert len(partition.test_indices) == 24

    future = set(np.flatnonzero(sessions == "2").tolist())
    assert future.isdisjoint(partition.train_indices.tolist())
    assert future.isdisjoint(partition.test_indices.tolist())


def test_chronological_partition_fails_when_no_history_exists():
    data = _fixture_data()
    with pytest.raises(ValueError, match="no prior"):
        chronological_partition(data, split_unit="session", held_out_value="0")


def test_first_observed_order_is_not_lexicographically_resorted():
    rng = np.random.default_rng(2)
    sessions = ("session_1", "session_2", "session_10")
    X = []
    y = []
    metadata = []
    for session in sessions:
        for label in ("left", "right"):
            for trial in range(2):
                X.append(rng.normal(size=(2, 4)))
                y.append(label)
                metadata.append(
                    {"subject": "1", "session": session, "run": f"r-{trial}"}
                )
    data = GroupedEvaluationData.from_moabb_result(
        (np.asarray(X), np.asarray(y), metadata), dataset_id="ordered"
    )

    assert ordered_group_values(data, split_unit="session") == sessions
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="session_10",
    )
    assert set(data.groups["session"][partition.train_indices]) == {
        "session_1",
        "session_2",
    }


def test_explicit_chronology_must_cover_each_observed_group_once():
    data = _fixture_data()
    with pytest.raises(ValueError, match="every observed group"):
        chronological_partition(
            data,
            split_unit="session",
            held_out_value="2",
            order=("0", "2"),
        )
    with pytest.raises(ValueError, match="duplicate"):
        chronological_partition(
            data,
            split_unit="session",
            held_out_value="2",
            order=("0", "1", "1", "2"),
        )


def test_nested_calibration_uses_fixed_evaluation_examples():
    partition = _fixture_partition()
    split = make_nested_calibration_split(
        partition,
        evaluation_fraction=0.5,
        seed=7,
    )

    assert len(split.source_train_indices) == 48
    assert len(split.evaluation_indices) == 12
    assert split.max_budget_per_class == 6

    y = partition.data.y.astype(str)
    assert set(y[split.evaluation_indices]) == {"left", "right"}
    assert np.sum(y[split.evaluation_indices] == "left") == 6
    assert np.sum(y[split.evaluation_indices] == "right") == 6

    evaluation_identity = split.evaluation_indices.copy()
    for budget in (0, 1, 3, 6):
        calibration = split.calibration_indices(budget)
        assert len(calibration) == budget * 2
        assert np.array_equal(split.evaluation_indices, evaluation_identity)
        assert np.intersect1d(calibration, split.evaluation_indices).size == 0
        assert np.intersect1d(calibration, split.source_train_indices).size == 0


def test_calibration_budgets_are_strictly_nested_per_class():
    split = make_nested_calibration_split(_fixture_partition(), seed=9)
    one = set(split.calibration_indices(1).tolist())
    three = set(split.calibration_indices(3).tolist())
    six = set(split.calibration_indices(6).tolist())

    assert one < three < six
    assert len(one) == 2
    assert len(three) == 6
    assert len(six) == 12


def test_split_fingerprint_and_manifest_are_deterministic():
    partition = _fixture_partition()
    first = make_nested_calibration_split(partition, seed=17)
    second = make_nested_calibration_split(partition, seed=17)
    changed = make_nested_calibration_split(partition, seed=18)

    assert first.fingerprint == second.fingerprint
    assert first.fingerprint != changed.fingerprint
    assert np.array_equal(first.evaluation_indices, second.evaluation_indices)

    manifest = first.manifest()
    assert manifest["source_train_samples"] == 48
    assert manifest["calibration_pool_samples"] == 12
    assert manifest["evaluation_samples"] == 12
    assert manifest["max_balanced_budget_per_class"] == 6
    assert manifest["calibration_pool_by_class"] == {"left": 6, "right": 6}
    assert manifest["evaluation_by_class"] == {"left": 6, "right": 6}
    assert manifest["calibration_split_fingerprint"] == first.fingerprint


def test_budget_above_balanced_pool_fails_closed():
    split = make_nested_calibration_split(_fixture_partition(), seed=3)
    with pytest.raises(ValueError, match="balanced maximum"):
        split.calibration_indices(split.max_budget_per_class + 1)
    with pytest.raises(ValueError, match="non-negative"):
        split.calibration_indices(-1)


def test_smallest_class_controls_balanced_budget():
    rng = np.random.default_rng(5)
    X = rng.normal(size=(16, 3, 8))
    y = np.asarray(["left"] * 12 + ["right"] * 4)
    metadata = [
        {"subject": "1", "session": "held", "run": f"r-{i}"}
        for i in range(16)
    ]
    X = np.concatenate([rng.normal(size=(8, 3, 8)), X], axis=0)
    y = np.concatenate([np.asarray(["left", "right"] * 4), y])
    metadata = [
        {"subject": "1", "session": "source", "run": f"s-{i}"}
        for i in range(8)
    ] + metadata
    data = GroupedEvaluationData.from_moabb_result(
        (X, y, metadata),
        dataset_id="imbalanced",
    )
    partition = hold_out_groups(data, split_unit="session", held_out_values=["held"])
    split = make_nested_calibration_split(partition, evaluation_fraction=0.5, seed=1)

    assert split.max_budget_per_class == 2


def test_invalid_fraction_and_single_class_fail_closed():
    partition = _fixture_partition()
    for invalid in (0.0, 1.0, -0.1, 1.1):
        with pytest.raises(ValueError, match="strictly between"):
            make_nested_calibration_split(partition, evaluation_fraction=invalid)

    data = partition.data
    single_class = GroupedEvaluationData(
        dataset_id="single",
        X=data.X,
        y=np.asarray(["left"] * len(data.X)),
        groups=data.groups,
        metadata=data.metadata,
    )
    held = hold_out_groups(single_class, split_unit="session", held_out_values=["2"])
    with pytest.raises(ValueError, match="at least two"):
        make_nested_calibration_split(held)
