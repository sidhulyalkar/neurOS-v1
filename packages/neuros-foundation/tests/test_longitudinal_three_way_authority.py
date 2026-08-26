from __future__ import annotations

import copy

import numpy as np
import pytest

from neuros.foundation_models.longitudinal import chronological_partition
from neuros.foundation_models.longitudinal_three_way import make_three_way_calibration_split
from neuros.foundation_models.longitudinal_three_way_authority import (
    ThreeWayLongitudinalCaseAuthority,
)
from neuros.foundation_models.real_world import GroupedEvaluationData


def _fixture_data() -> GroupedEvaluationData:
    rng = np.random.default_rng(83)
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
        (np.asarray(X), np.asarray(y), metadata), dataset_id="three-way-authority-fixture"
    )


def _authority():
    data = _fixture_data()
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="2",
    )
    split = make_three_way_calibration_split(
        partition,
        qualification_fraction=0.25,
        final_assessment_fraction=0.25,
        seed=19,
    )
    authority = ThreeWayLongitudinalCaseAuthority.from_split(
        split,
        case_id="subject-1/session-2/v2",
        history_policy="prior",
        case_metadata={"protocol": "synthetic-contract"},
    )
    return data, split, authority


def test_v2_authority_freezes_chronology_and_three_distinct_target_roles():
    data, split, authority = _authority()

    assert authority.schema_version == 2
    assert authority.dataset_id == data.dataset_id
    assert authority.split_unit == "session"
    assert authority.held_out_values == ("2",)
    assert authority.observed_group_order == ("0", "1", "2")
    assert authority.source_group_values == ("0", "1")
    assert len(authority.source_train_indices) == 48
    assert len(authority.qualification_indices) == 6
    assert len(authority.final_assessment_indices) == 6
    assert authority.max_budget_per_class == 6
    assert len(authority.authority_fingerprint) == 64
    assert len(authority.qualification_set_sha256) == 64
    assert len(authority.final_assessment_set_sha256) == 64
    assert authority.qualification_set_sha256 != authority.final_assessment_set_sha256
    assert authority.three_way_split_fingerprint == split.fingerprint

    calibration = set(authority.calibration_indices(6))
    qualification = set(authority.qualification_indices)
    final_assessment = set(authority.final_assessment_indices)
    source = set(authority.source_train_indices)
    assert calibration.isdisjoint(qualification)
    assert calibration.isdisjoint(final_assessment)
    assert qualification.isdisjoint(final_assessment)
    assert source.isdisjoint(calibration | qualification | final_assessment)


def test_budget_identities_are_nested_and_final_authority_is_budget_invariant():
    _, _, authority = _authority()
    final_identity = authority.final_assessment_set_sha256
    qualification_identity = authority.qualification_set_sha256

    one = set(authority.calibration_indices(1))
    three = set(authority.calibration_indices(3))
    six = set(authority.calibration_indices(6))
    assert one < three < six
    assert authority.calibration_indices(0) == ()

    hashes = [authority.calibration_budget_sha256(budget) for budget in (0, 1, 3, 6)]
    assert len(set(hashes)) == 4
    assert all(len(value) == 64 for value in hashes)
    assert authority.final_assessment_set_sha256 == final_identity
    assert authority.qualification_set_sha256 == qualification_identity

    with pytest.raises(ValueError, match="integer"):
        authority.calibration_budget_sha256(1.5)  # type: ignore[arg-type]


def test_exact_set_guards_reject_subsets_supersets_and_reordering():
    _, _, authority = _authority()
    qualification = authority.qualification_indices
    final_assessment = authority.final_assessment_indices
    calibration = authority.calibration_indices(3)

    assert authority.require_qualification_indices(qualification) == qualification
    assert authority.require_final_assessment_indices(final_assessment) == final_assessment
    assert authority.require_calibration_indices(3, calibration) == calibration
    assert authority.require_calibration_indices(0, ()) == ()

    with pytest.raises(ValueError, match="complete frozen qualification"):
        authority.require_qualification_indices(qualification[:-1])
    with pytest.raises(ValueError, match="exact order"):
        authority.require_qualification_indices(tuple(reversed(qualification)))
    with pytest.raises(ValueError, match="complete frozen final-assessment"):
        authority.require_final_assessment_indices(final_assessment[:-1])
    with pytest.raises(ValueError, match="canonical order"):
        authority.require_calibration_indices(3, tuple(reversed(calibration)))


def test_serialization_roundtrip_and_restore_are_exact_and_deterministic():
    data, split, authority = _authority()
    payload = authority.to_dict()
    restored_authority = ThreeWayLongitudinalCaseAuthority.from_dict(payload)
    restored_split = restored_authority.restore(data)

    assert restored_authority.authority_fingerprint == authority.authority_fingerprint
    assert restored_authority.to_dict() == payload
    assert restored_split.fingerprint == split.fingerprint
    assert np.array_equal(restored_split.qualification_indices, split.qualification_indices)
    assert np.array_equal(
        restored_split.final_assessment_indices, split.final_assessment_indices
    )
    for budget in (0, 1, 3, 6):
        assert np.array_equal(
            restored_split.calibration_indices(budget), split.calibration_indices(budget)
        )


def test_tampered_serialized_final_set_fails_its_independent_sha_identity():
    _, _, authority = _authority()
    payload = copy.deepcopy(authority.to_dict())
    payload.pop("authority_fingerprint")
    values = list(payload["final_assessment_indices"])
    values[0], values[1] = values[1], values[0]
    payload["final_assessment_indices"] = values

    with pytest.raises(ValueError, match="final_assessment_set_sha256"):
        ThreeWayLongitudinalCaseAuthority.from_dict(payload)


def test_pairwise_overlap_fails_before_dataset_restore():
    _, _, authority = _authority()
    payload = copy.deepcopy(authority.to_dict())
    payload.pop("authority_fingerprint")
    payload.pop("qualification_set_sha256")
    payload.pop("final_assessment_set_sha256")
    payload["final_assessment_indices"][0] = payload["qualification_indices"][0]

    with pytest.raises(ValueError, match="pairwise disjoint"):
        ThreeWayLongitudinalCaseAuthority.from_dict(payload)


def test_processed_data_change_fails_before_reconstructed_split_is_trusted():
    data, _, authority = _authority()
    changed_x = np.array(data.X, copy=True)
    changed_x[0, 0, 0] += 0.125
    changed = GroupedEvaluationData(
        dataset_id=data.dataset_id,
        X=changed_x,
        y=np.array(data.y, copy=True),
        groups={key: np.array(values, copy=True) for key, values in data.groups.items()},
        metadata=data.metadata,
    )

    with pytest.raises(ValueError, match="processed neural data SHA-256"):
        authority.restore(changed)


def test_prior_chronology_tampering_fails_at_authority_construction():
    _, _, authority = _authority()
    payload = copy.deepcopy(authority.to_dict())
    payload.pop("authority_fingerprint")
    payload["observed_group_order"] = ["0", "2", "1"]

    with pytest.raises(ValueError, match="complete chronological prefix"):
        ThreeWayLongitudinalCaseAuthority.from_dict(payload)


def test_metadata_must_be_deterministic_and_finite():
    _, split, _ = _authority()

    with pytest.raises(ValueError, match="NaN or infinity"):
        ThreeWayLongitudinalCaseAuthority.from_split(
            split,
            case_id="nan-metadata",
            history_policy="prior",
            case_metadata={"score": float("nan")},
        )

    with pytest.raises(TypeError, match="JSON-compatible"):
        ThreeWayLongitudinalCaseAuthority.from_split(
            split,
            case_id="object-metadata",
            history_policy="prior",
            case_metadata={"opaque": object()},
        )


def test_declared_group_identity_invariants_fail_before_restore():
    _, _, authority = _authority()

    payload = copy.deepcopy(authority.to_dict())
    payload.pop("authority_fingerprint")
    payload["observed_group_order"] = ["0", "1", "1", "2"]
    with pytest.raises(ValueError, match="duplicate"):
        ThreeWayLongitudinalCaseAuthority.from_dict(payload)

    payload = copy.deepcopy(authority.to_dict())
    payload.pop("authority_fingerprint")
    payload["source_group_values"] = ["0", "2"]
    with pytest.raises(ValueError, match="disjoint"):
        ThreeWayLongitudinalCaseAuthority.from_dict(payload)


def test_schema_and_history_policy_fail_closed():
    _, _, authority = _authority()
    payload = copy.deepcopy(authority.to_dict())
    payload["schema_version"] = 1
    with pytest.raises(ValueError, match="schema_version=2"):
        ThreeWayLongitudinalCaseAuthority.from_dict(payload)

    payload = copy.deepcopy(authority.to_dict())
    payload.pop("authority_fingerprint")
    payload["history_policy"] = "future-data"
    with pytest.raises(ValueError, match="unsupported history_policy"):
        ThreeWayLongitudinalCaseAuthority.from_dict(payload)
