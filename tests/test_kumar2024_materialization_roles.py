from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from neuros.evidence.kumar2024_materialization import (
    build_case_result_observation_roles,
    build_processed_subject_shard,
)
from neuros.foundation_models.longitudinal_authority import (
    LongitudinalCaseAuthority,
    processed_data_sha256,
)
from neuros.foundation_models.real_world import GroupedEvaluationData


def _runner_observation_hash(role: str, processed_sha256: str, indices) -> str:
    payload = {
        "schema": "neuros.qualification_observation_set.v1",
        "payload": {
            "role": role,
            "processed_data_sha256": processed_sha256,
            "indices": [int(value) for value in indices],
        },
    }
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _data() -> GroupedEvaluationData:
    X = np.arange(10 * 2 * 4, dtype=np.float32).reshape(10, 2, 4)
    y = np.asarray([
        "left", "right", "left", "right", "left",
        "right", "left", "right", "left", "right",
    ])
    return GroupedEvaluationData(
        dataset_id="moabb-kumar2024",
        X=X,
        y=y,
        groups={
            "subject": np.asarray(["1"] * 10),
            "session": np.asarray(["0", "0", "0", "0", "1", "1", "1", "1", "1", "1"]),
            "run": np.asarray(["0", "0", "1", "1", "0", "0", "1", "1", "2", "2"]),
        },
        metadata=tuple({"row": index} for index in range(10)),
    )


def _authority(data: GroupedEvaluationData) -> LongitudinalCaseAuthority:
    return LongitudinalCaseAuthority(
        dataset_id=data.dataset_id,
        case_id="kumar2024/subject-1/session-1/split-2026",
        split_unit="session",
        held_out_values=("1",),
        history_policy="prior",
        observed_group_order=("0", "1"),
        source_group_values=("0",),
        source_train_indices=(0, 1, 2, 3),
        evaluation_indices=(8, 9),
        calibration_order_by_class={
            "left": (4, 6),
            "right": (5, 7),
        },
        evaluation_fraction=0.5,
        seed=2026,
        partition_fingerprint="partition-fixture",
        calibration_split_fingerprint="calibration-fixture",
        processed_data_sha256=processed_data_sha256(data),
        n_samples=len(data.X),
        input_shape=tuple(int(value) for value in data.X.shape),
        case_metadata={"subject": 1},
    )


def _row(authority, *, budget: int, calibration, fit, state=None, source_sha=None):
    processed = authority.processed_data_sha256
    source = authority.source_train_indices
    evaluation = authority.evaluation_indices
    return SimpleNamespace(
        calibration_per_class=budget,
        method_id="fixture-eegnet",
        source_train_indices_sha256=(
            source_sha
            if source_sha is not None
            else _runner_observation_hash("supervised_source_history", processed, source)
        ),
        labeled_target_indices_sha256=_runner_observation_hash(
            "labeled_target_calibration", processed, calibration
        ),
        fit_indices_sha256=_runner_observation_hash(
            "supervised_fit", processed, fit
        ),
        evaluation_indices_sha256=_runner_observation_hash(
            "untouched_final_assessment", processed, evaluation
        ),
        qualification_model_state=state,
        sha256=f"fixture-row-{budget}",
    )


def test_roles_reconcile_with_runner_hashes_and_zero_budget_is_signed():
    data = _data()
    authority = _authority(data)
    shard = build_processed_subject_shard(
        data,
        subject=1,
        preprocessing_authority_sha256="b" * 64,
    )

    zero_fit = np.asarray(authority.source_train_indices, dtype=np.int64)
    one_calibration = np.asarray([4, 5], dtype=np.int64)
    one_fit = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    neural_state = SimpleNamespace(
        learned_state=SimpleNamespace(
            metadata={"validation_relative_indices": [1, 4]}
        )
    )
    result = SimpleNamespace(
        case_authority_sha256=authority.authority_sha256,
        rows=(
            _row(
                authority,
                budget=0,
                calibration=np.asarray([], dtype=np.int64),
                fit=zero_fit,
            ),
            _row(
                authority,
                budget=1,
                calibration=one_calibration,
                fit=one_fit,
                state=neural_state,
            ),
        ),
    )

    rendered = build_case_result_observation_roles(
        authority=authority,
        shard=shard,
        result=result,
    )

    zero_roles = rendered[0]["roles"]
    assert zero_roles["labeled_target_calibration"]["row_indices"] == []
    assert zero_roles["labeled_target_calibration"]["display_ids"] == []
    assert zero_roles["labeled_target_calibration"]["observation_sha256s"] == []
    assert zero_roles["labeled_target_calibration"]["nsq_observation_set_sha256"] == (
        _runner_observation_hash(
            "labeled_target_calibration",
            authority.processed_data_sha256,
            [],
        )
    )

    one_roles = rendered[1]["roles"]
    expected = {
        "supervised_source_history": [0, 1, 2, 3],
        "labeled_target_calibration": [4, 5],
        "supervised_fit": [0, 1, 2, 3, 4, 5],
        "untouched_final_assessment": [8, 9],
    }
    for role, indices in expected.items():
        assert one_roles[role]["row_indices"] == indices
        assert one_roles[role]["nsq_observation_set_sha256"] == (
            _runner_observation_hash(
                role,
                authority.processed_data_sha256,
                indices,
            )
        )

    assert one_roles["internal_model_validation"]["row_indices"] == [1, 4]
    assert one_roles["internal_model_training"]["row_indices"] == [0, 2, 3, 5]
    assert not set(one_roles["internal_model_validation"]["row_indices"]) & {8, 9}
    serialized = json.dumps(rendered, sort_keys=True)
    assert '"left"' not in serialized
    assert '"right"' not in serialized


def test_role_reconciliation_fails_closed_on_runner_hash_mismatch():
    data = _data()
    authority = _authority(data)
    shard = build_processed_subject_shard(
        data,
        subject=1,
        preprocessing_authority_sha256="b" * 64,
    )
    result = SimpleNamespace(
        case_authority_sha256=authority.authority_sha256,
        rows=(
            _row(
                authority,
                budget=0,
                calibration=[],
                fit=authority.source_train_indices,
                source_sha="0" * 64,
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="does not reconcile"):
        build_case_result_observation_roles(
            authority=authority,
            shard=shard,
            result=result,
        )


def test_internal_validation_membership_cannot_escape_authorized_fit_set():
    data = _data()
    authority = _authority(data)
    shard = build_processed_subject_shard(
        data,
        subject=1,
        preprocessing_authority_sha256="b" * 64,
    )
    calibration = np.asarray([4, 5], dtype=np.int64)
    fit = np.asarray([0, 1, 2, 3, 4, 5], dtype=np.int64)
    invalid_state = SimpleNamespace(
        learned_state=SimpleNamespace(
            metadata={"validation_relative_indices": [0, len(fit)]}
        )
    )
    result = SimpleNamespace(
        case_authority_sha256=authority.authority_sha256,
        rows=(
            _row(
                authority,
                budget=1,
                calibration=calibration,
                fit=fit,
                state=invalid_state,
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="escapes authorized fit set"):
        build_case_result_observation_roles(
            authority=authority,
            shard=shard,
            result=result,
        )
