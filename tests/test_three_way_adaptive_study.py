from __future__ import annotations

import copy

import pytest

from orion import AdaptiveStudyAuthority, FinalAssessmentAuthority
from scripts.evidence.verify_three_way_adaptive_study import (
    build_source_authority,
    derive_adaptive_study,
    run_contract,
)


def test_longitudinal_v2_derives_exact_orion_roles_without_repartitioning():
    source = build_source_authority()
    study = derive_adaptive_study(source, budget_per_class=2)

    assert study.source_authority_fingerprint == source.authority_fingerprint
    assert study.adaptation_authority.source_authority_fingerprint == source.authority_fingerprint
    assert (
        study.final_assessment_authority.source_authority_fingerprint
        == source.authority_fingerprint
    )
    assert study.adaptation_authority.adaptation_indices == source.calibration_indices(2)
    assert study.adaptation_authority.evaluation_indices == source.qualification_indices
    assert study.final_assessment_authority.assessment_indices == source.final_assessment_indices
    assert study.adaptation_authority.processed_data_sha256 == source.processed_data_sha256
    assert (
        study.final_assessment_authority.processed_data_sha256
        == source.processed_data_sha256
    )
    assert study.adaptation_authority.protocol_fingerprint == source.three_way_split_fingerprint
    assert (
        study.final_assessment_authority.protocol_fingerprint
        == source.three_way_split_fingerprint
    )


def test_final_authority_is_budget_invariant_while_calibration_authority_changes():
    source = build_source_authority()
    one = derive_adaptive_study(source, budget_per_class=1)
    three = derive_adaptive_study(source, budget_per_class=3)

    assert one.adaptation_authority.authority_fingerprint != three.adaptation_authority.authority_fingerprint
    assert one.adaptation_authority.adaptation_indices != three.adaptation_authority.adaptation_indices
    assert one.adaptation_authority.evaluation_indices == three.adaptation_authority.evaluation_indices
    assert (
        one.final_assessment_authority.authority_fingerprint
        == three.final_assessment_authority.authority_fingerprint
    )
    assert (
        one.final_assessment_authority.assessment_indices
        == three.final_assessment_authority.assessment_indices
    )


def test_budget_zero_fails_as_adaptation_and_is_reserved_for_frozen_state():
    source = build_source_authority()
    with pytest.raises(ValueError, match="budget 0 as a frozen SelectedState"):
        derive_adaptive_study(source, budget_per_class=0)


def test_cross_authority_overlap_or_identity_drift_fails_closed():
    source = build_source_authority()
    study = derive_adaptive_study(source, budget_per_class=1)
    final = study.final_assessment_authority

    overlapping = FinalAssessmentAuthority(
        authority_id="overlap",
        dataset_id=final.dataset_id,
        split_unit=final.split_unit,
        assessment_indices=(
            study.adaptation_authority.adaptation_indices[0],
            *final.assessment_indices[:-1],
        ),
        processed_data_sha256=final.processed_data_sha256,
        n_samples=final.n_samples,
        source_authority_fingerprint=final.source_authority_fingerprint,
        metric_names=final.metric_names,
        protocol_fingerprint=final.protocol_fingerprint,
        seed=final.seed,
    )
    with pytest.raises(ValueError, match="overlap adaptation/calibration"):
        AdaptiveStudyAuthority(
            study_id="overlap",
            source_authority_fingerprint=source.authority_fingerprint,
            adaptation_authority=study.adaptation_authority,
            final_assessment_authority=overlapping,
        )

    wrong_sha = FinalAssessmentAuthority(
        authority_id="wrong-data",
        dataset_id=final.dataset_id,
        split_unit=final.split_unit,
        assessment_indices=final.assessment_indices,
        processed_data_sha256="0" * 64,
        n_samples=final.n_samples,
        source_authority_fingerprint=final.source_authority_fingerprint,
        metric_names=final.metric_names,
        protocol_fingerprint=final.protocol_fingerprint,
        seed=final.seed,
    )
    with pytest.raises(ValueError, match="processed-data identities differ"):
        AdaptiveStudyAuthority(
            study_id="wrong-data",
            source_authority_fingerprint=source.authority_fingerprint,
            adaptation_authority=study.adaptation_authority,
            final_assessment_authority=wrong_sha,
        )


def test_worker_proves_selection_precedes_final_assessment_and_replays_deterministically():
    first = run_contract()
    second = run_contract()

    assert first == second
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert first["invariants"]["final_assessment_after_state_selection"] is True
    assert first["invariants"]["budget_zero_is_frozen_not_adapted"] is True
    assert first["claim_boundary"]["real_neural_data"] is False
    assert first["claim_boundary"]["adaptation_efficacy"] is False

    frozen = first["frozen_baseline"]
    adaptive = first["adaptive_path"]
    assert frozen["selected_state"]["kind"] == "frozen"
    assert adaptive["selected_state"]["kind"] == "adapted"
    assert (
        frozen["final_assessment"]["authority_fingerprint"]
        == adaptive["final_assessment"]["authority_fingerprint"]
    )
    assert (
        frozen["final_assessment"]["assessment_indices"]
        == adaptive["final_assessment"]["assessment_indices"]
    )


def test_final_authority_metadata_copy_cannot_change_cross_package_identity():
    source = build_source_authority()
    study = derive_adaptive_study(source, budget_per_class=1)
    payload = study.final_assessment_authority.to_dict()
    fingerprint = study.final_assessment_authority.authority_fingerprint

    mutated = copy.deepcopy(payload)
    mutated["metadata"]["role"] = "changed-outside-object"
    assert study.final_assessment_authority.authority_fingerprint == fingerprint
    assert study.final_assessment_authority.to_dict()["metadata"]["role"] == "untouched-final-assessment"
