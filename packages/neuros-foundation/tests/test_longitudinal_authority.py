from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
)
from neuros.foundation_models.longitudinal_authority import (
    LongitudinalCaseAuthority,
    processed_data_sha256,
)
from neuros.foundation_models.real_world import GroupedEvaluationData


def _fixture() -> GroupedEvaluationData:
    rng = np.random.default_rng(123)
    X = []
    y = []
    metadata = []
    for session_index, session in enumerate(("0", "1", "2")):
        for label_index, label in enumerate(("left", "right")):
            for trial in range(8):
                signal = rng.normal(size=(4, 32))
                signal[label_index, 8:24] += 0.1 * session_index + (1 if label_index else -1)
                X.append(signal)
                y.append(label)
                metadata.append(
                    {
                        "subject": "1",
                        "session": session,
                        "run": f"r-{trial // 4}",
                    }
                )
    return GroupedEvaluationData.from_moabb_result(
        (np.asarray(X, dtype=np.float32), np.asarray(y), metadata),
        dataset_id="fixture",
    )


def _authority(data: GroupedEvaluationData) -> LongitudinalCaseAuthority:
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="2",
        order=("0", "1", "2"),
    )
    split = make_nested_calibration_split(
        partition,
        evaluation_fraction=0.5,
        seed=2026,
    )
    return LongitudinalCaseAuthority.from_split(
        split,
        case_id="subject-1/session-2",
        history_policy="prior",
        observed_group_order=("0", "1", "2"),
        case_metadata={"subject": 1, "original_protocol": "GR"},
    )


def test_processed_fingerprint_is_stable_and_value_sensitive():
    data = _fixture()
    first = processed_data_sha256(data)
    second = processed_data_sha256(data)
    assert first == second
    assert len(first) == 64

    changed_x = data.X.copy()
    changed_x[0, 0, 0] += 1e-3
    changed = GroupedEvaluationData(
        dataset_id=data.dataset_id,
        X=changed_x,
        y=data.y,
        groups=data.groups,
        metadata=data.metadata,
    )
    assert processed_data_sha256(changed) != first


def test_authority_roundtrip_restores_exact_split():
    data = _fixture()
    authority = _authority(data)
    serialized = authority.to_dict()
    json.dumps(serialized, sort_keys=True)

    assert len(authority.authority_sha256) == 64
    assert authority.authority_fingerprint == authority.authority_sha256[:16]
    assert serialized["authority_sha256"] == authority.authority_sha256
    assert serialized["authority_fingerprint"] == authority.authority_fingerprint

    restored_authority = LongitudinalCaseAuthority.from_dict(serialized)
    split = restored_authority.restore(data)

    assert restored_authority.authority_sha256 == authority.authority_sha256
    assert restored_authority.authority_fingerprint == authority.authority_fingerprint
    assert split.partition.fingerprint == authority.partition_fingerprint
    assert split.fingerprint == authority.calibration_split_fingerprint
    assert tuple(split.source_train_indices) == authority.source_train_indices
    assert tuple(split.evaluation_indices) == authority.evaluation_indices
    assert restored_authority.case_metadata["original_protocol"] == "GR"


def test_authority_rejects_changed_processed_neural_values():
    data = _fixture()
    authority = _authority(data)
    changed_x = data.X.copy()
    changed_x[-1, -1, -1] += 0.25
    changed = GroupedEvaluationData(
        dataset_id=data.dataset_id,
        X=changed_x,
        y=data.y,
        groups=data.groups,
        metadata=data.metadata,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        authority.restore(changed)


def test_authority_rejects_sample_order_change_even_when_shape_matches():
    data = _fixture()
    authority = _authority(data)
    order = np.arange(len(data.X))
    order[[0, 1]] = order[[1, 0]]
    reordered = GroupedEvaluationData(
        dataset_id=data.dataset_id,
        X=data.X[order],
        y=data.y[order],
        groups={key: np.asarray(values)[order] for key, values in data.groups.items()},
        metadata=tuple(data.metadata[index] for index in order),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        authority.restore(reordered)


def test_serialized_authority_sha256_detects_structurally_valid_index_tampering():
    authority = _authority(_fixture())
    payload = copy.deepcopy(authority.to_dict())
    payload["evaluation_indices"][0], payload["evaluation_indices"][1] = (
        payload["evaluation_indices"][1],
        payload["evaluation_indices"][0],
    )
    with pytest.raises(ValueError, match="authority_sha256"):
        LongitudinalCaseAuthority.from_dict(payload)


def test_serialized_authority_rejects_lossy_index_coercion():
    authority = _authority(_fixture())
    payload = copy.deepcopy(authority.to_dict())
    payload["evaluation_indices"][0] = float(payload["evaluation_indices"][0])
    with pytest.raises(ValueError, match="without coercion"):
        LongitudinalCaseAuthority.from_dict(payload)


def test_prior_authority_rejects_future_session_in_source_history():
    data = _fixture()
    authority = _authority(data)
    payload = authority.to_dict(include_fingerprint=False)

    # Make a structurally self-consistent but scientifically invalid authority:
    # target session 1 with source sessions 0 and 2 (future leakage). The split
    # fingerprints intentionally no longer match, so restore must fail closed
    # before such an authority can be used by a method runner.
    group = np.asarray(data.groups["session"])
    payload["held_out_values"] = ["1"]
    payload["source_group_values"] = ["0", "2"]
    payload["source_train_indices"] = np.flatnonzero(np.isin(group, ["0", "2"])).tolist()
    target = np.flatnonzero(group == "1")
    payload["evaluation_indices"] = target[:8].tolist()
    payload["calibration_order_by_class"] = {
        "left": target[8:12].tolist(),
        "right": target[12:16].tolist(),
    }

    invalid = LongitudinalCaseAuthority.from_dict(payload)
    with pytest.raises(ValueError):
        invalid.restore(data)
