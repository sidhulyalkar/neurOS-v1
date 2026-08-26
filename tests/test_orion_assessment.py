from __future__ import annotations

import hashlib

import pytest

from orion import AdaptationProposal
from orion.adaptation import (
    AdaptationApplication,
    AdaptationAuthority,
    AdaptationDecision,
    AdaptationEvaluation,
    AdaptationOutcome,
    ArtifactIdentity,
    GovernedAdaptationProposal,
)
from orion.assessment import (
    AdaptiveStudyAuthority,
    FinalAssessmentAuthority,
    FinalAssessmentRecord,
    SelectedState,
    SelectionKind,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _artifact(label: str) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=f"decoder/{label}",
        artifact_type="decoder-state",
        sha256=_sha(label),
        metadata={"backend": "fixture"},
    )


def _adaptation_authority(
    *,
    adaptation_indices: tuple[int, ...] = (0, 2),
    qualification_indices: tuple[int, ...] = (1, 3),
    source: str = "three-way-source",
) -> AdaptationAuthority:
    return AdaptationAuthority(
        authority_id="case/budget-1",
        dataset_id="fixture-eeg",
        split_unit="session",
        adaptation_indices=adaptation_indices,
        evaluation_indices=qualification_indices,
        processed_data_sha256=_sha("processed-data"),
        n_samples=8,
        protocol_fingerprint="three-way-protocol",
        source_authority_fingerprint=source,
        seed=7,
    )


def _final_authority(
    *,
    assessment_indices: tuple[int, ...] = (4, 5, 6, 7),
    source: str = "three-way-source",
) -> FinalAssessmentAuthority:
    return FinalAssessmentAuthority(
        authority_id="case/final-v1",
        dataset_id="fixture-eeg",
        split_unit="session",
        assessment_indices=assessment_indices,
        processed_data_sha256=_sha("processed-data"),
        n_samples=8,
        source_authority_fingerprint=source,
        metric_names=("balanced_accuracy", "ece"),
        protocol_fingerprint="three-way-protocol",
        seed=7,
        metadata={"scorecard": {"primary": "balanced_accuracy", "version": 1}},
    )


def _outcome() -> tuple[AdaptationAuthority, AdaptationOutcome]:
    authority = _adaptation_authority()
    proposal = AdaptationProposal(
        reason="target calibration",
        changes={"learning_rate": 0.01},
        evidence={"calibration_loss": 0.5},
        requires_approval=True,
    )
    governed = GovernedAdaptationProposal.bind(
        proposal,
        authority=authority,
        before_artifact=_artifact("before"),
        adaptation_indices=authority.adaptation_indices,
    )
    decision = AdaptationDecision.approve(
        governed,
        actor="policy/fixture",
        reason="authorized fixture update",
    )
    application = AdaptationApplication.record(
        governed,
        decision,
        authority=authority,
        after_artifact=_artifact("after"),
        adaptation_indices=authority.adaptation_indices,
        update_evidence={"updates": 2},
    )
    evaluation = AdaptationEvaluation.record(
        application,
        authority=authority,
        evaluation_indices=authority.evaluation_indices,
        metrics_before={"balanced_accuracy": 0.60},
        metrics_after={"balanced_accuracy": 0.70},
    )
    outcome = AdaptationOutcome.retain(
        application,
        evaluation,
        actor="policy/fixture",
        reason="qualification improved",
    )
    return authority, outcome


def test_final_authority_rejects_bad_indices_scorecard_and_range():
    with pytest.raises(ValueError, match="duplicate"):
        _final_authority(assessment_indices=(4, 4, 6))

    with pytest.raises(ValueError, match="out-of-range"):
        _final_authority(assessment_indices=(4, 5, 8))

    with pytest.raises(ValueError, match="duplicates"):
        FinalAssessmentAuthority(
            authority_id="bad",
            dataset_id="fixture-eeg",
            split_unit="session",
            assessment_indices=(4, 5),
            processed_data_sha256=_sha("processed-data"),
            n_samples=8,
            source_authority_fingerprint="source",
            metric_names=("ece", "ece"),
        )


def test_final_authority_requires_complete_exact_rows_and_metric_scorecard():
    authority = _final_authority()
    assert authority.require_assessment_indices((4, 5, 6, 7)) == (4, 5, 6, 7)

    with pytest.raises(ValueError, match="complete frozen assessment indices"):
        authority.require_assessment_indices((4, 5, 6))
    with pytest.raises(ValueError, match="exact order"):
        authority.require_assessment_indices((7, 6, 5, 4))

    metrics = authority.require_metrics(
        {"balanced_accuracy": 0.72, "ece": 0.08}
    )
    assert metrics["balanced_accuracy"] == pytest.approx(0.72)

    with pytest.raises(ValueError, match="exact predeclared metric scorecard"):
        authority.require_metrics({"balanced_accuracy": 0.72})
    with pytest.raises(ValueError, match="exact predeclared metric scorecard"):
        authority.require_metrics(
            {"balanced_accuracy": 0.72, "ece": 0.08, "accuracy": 0.74}
        )
    with pytest.raises(ValueError, match="must be finite"):
        authority.require_metrics(
            {"balanced_accuracy": float("nan"), "ece": 0.08}
        )


def test_frozen_baseline_is_a_real_selected_state_without_fake_adaptation():
    selected = SelectedState.frozen(
        selection_id="baseline/predeclared",
        source_authority_fingerprint="three-way-source",
        artifact=_artifact("frozen"),
        metadata={"calibration_examples": 0},
    )

    assert selected.kind is SelectionKind.FROZEN
    assert selected.adaptation_authority_fingerprint is None
    assert selected.selection_evidence_fingerprint is None
    assert selected.artifact.sha256 == _sha("frozen")

    authority = _final_authority()
    record = FinalAssessmentRecord.record(
        selected,
        authority=authority,
        assessment_indices=authority.assessment_indices,
        metrics={"balanced_accuracy": 0.62, "ece": 0.12},
    )
    assert record.selected_artifact.sha256 == selected.artifact.sha256
    assert record.source_authority_fingerprint == "three-way-source"


def test_adapted_selected_state_binds_exact_outcome_artifact():
    authority, outcome = _outcome()
    selected = SelectedState.from_adaptation_outcome(
        outcome,
        selection_id="hebbian/retained",
        source_authority_fingerprint=authority.source_authority_fingerprint or "",
    )

    assert selected.kind is SelectionKind.ADAPTED
    assert selected.artifact.sha256 == outcome.active_artifact.sha256
    assert selected.adaptation_authority_fingerprint == outcome.authority_fingerprint
    assert selected.selection_evidence_fingerprint == outcome.outcome_fingerprint


def test_study_authority_requires_same_source_dataset_protocol_and_disjoint_final_rows():
    adaptation = _adaptation_authority()
    final = _final_authority()
    study = AdaptiveStudyAuthority(
        study_id="fixture/budget-1",
        source_authority_fingerprint="three-way-source",
        adaptation_authority=adaptation,
        final_assessment_authority=final,
    )
    assert study.study_fingerprint

    with pytest.raises(ValueError, match="same source study authority"):
        AdaptiveStudyAuthority(
            study_id="bad-source",
            source_authority_fingerprint="other-source",
            adaptation_authority=adaptation,
            final_assessment_authority=final,
        )

    overlapping = _final_authority(assessment_indices=(2, 4, 5, 6))
    with pytest.raises(ValueError, match="overlap adaptation/calibration"):
        AdaptiveStudyAuthority(
            study_id="bad-overlap",
            source_authority_fingerprint="three-way-source",
            adaptation_authority=adaptation,
            final_assessment_authority=overlapping,
        )

    qualification_overlap = _final_authority(assessment_indices=(3, 4, 5, 6))
    with pytest.raises(ValueError, match="overlap qualification"):
        AdaptiveStudyAuthority(
            study_id="bad-qualification-overlap",
            source_authority_fingerprint="three-way-source",
            adaptation_authority=adaptation,
            final_assessment_authority=qualification_overlap,
        )


def test_study_can_only_select_an_outcome_from_its_adaptation_authority():
    adaptation, outcome = _outcome()
    study = AdaptiveStudyAuthority(
        study_id="fixture/budget-1",
        source_authority_fingerprint="three-way-source",
        adaptation_authority=adaptation,
        final_assessment_authority=_final_authority(),
    )
    selected = study.select_outcome(outcome, selection_id="selected/hebbian")
    assert selected.artifact.sha256 == outcome.active_artifact.sha256

    other_authority = _adaptation_authority(
        adaptation_indices=(0, 1),
        qualification_indices=(2, 3),
    )
    proposal = AdaptationProposal(reason="other", changes={"x": 1})
    governed = GovernedAdaptationProposal.bind(
        proposal,
        authority=other_authority,
        before_artifact=_artifact("other-before"),
        adaptation_indices=other_authority.adaptation_indices,
    )
    decision = AdaptationDecision.approve(
        governed, actor="fixture", reason="fixture"
    )
    application = AdaptationApplication.record(
        governed,
        decision,
        authority=other_authority,
        after_artifact=_artifact("other-after"),
        adaptation_indices=other_authority.adaptation_indices,
    )
    evaluation = AdaptationEvaluation.record(
        application,
        authority=other_authority,
        evaluation_indices=other_authority.evaluation_indices,
        metrics_before={"balanced_accuracy": 0.5},
        metrics_after={"balanced_accuracy": 0.6},
    )
    other_outcome = AdaptationOutcome.retain(
        application,
        evaluation,
        actor="fixture",
        reason="fixture",
    )
    with pytest.raises(ValueError, match="does not belong to this study authority"):
        study.select_outcome(other_outcome, selection_id="wrong")


def test_final_record_rejects_wrong_source_rows_or_scorecard():
    selected = SelectedState.frozen(
        selection_id="baseline",
        source_authority_fingerprint="three-way-source",
        artifact=_artifact("baseline"),
    )
    authority = _final_authority()

    with pytest.raises(ValueError, match="complete frozen assessment indices"):
        FinalAssessmentRecord.record(
            selected,
            authority=authority,
            assessment_indices=(4, 5, 6),
            metrics={"balanced_accuracy": 0.6, "ece": 0.1},
        )

    with pytest.raises(ValueError, match="exact predeclared metric scorecard"):
        FinalAssessmentRecord.record(
            selected,
            authority=authority,
            assessment_indices=authority.assessment_indices,
            metrics={"balanced_accuracy": 0.6},
        )

    wrong_source = SelectedState.frozen(
        selection_id="wrong-source",
        source_authority_fingerprint="different-source",
        artifact=_artifact("baseline"),
    )
    with pytest.raises(ValueError, match="same source authority"):
        FinalAssessmentRecord.record(
            wrong_source,
            authority=authority,
            assessment_indices=authority.assessment_indices,
            metrics={"balanced_accuracy": 0.6, "ece": 0.1},
        )


def test_assessment_metadata_is_deeply_immutable_and_fingerprint_stable():
    metadata = {"scorecard": {"thresholds": [0.5, 0.7]}}
    authority = FinalAssessmentAuthority(
        authority_id="immutable",
        dataset_id="fixture-eeg",
        split_unit="session",
        assessment_indices=(4, 5),
        processed_data_sha256=_sha("processed-data"),
        n_samples=8,
        source_authority_fingerprint="three-way-source",
        metric_names=("balanced_accuracy",),
        metadata=metadata,
    )
    fingerprint = authority.authority_fingerprint

    metadata["scorecard"]["thresholds"].append(0.9)
    assert authority.authority_fingerprint == fingerprint
    assert authority.to_dict()["metadata"]["scorecard"]["thresholds"] == [0.5, 0.7]

    with pytest.raises(TypeError):
        authority.metadata["scorecard"]["thresholds"] += (0.9,)


def test_assessment_metadata_rejects_key_collisions_and_nonfinite_values():
    with pytest.raises(ValueError, match="collide after string normalization"):
        FinalAssessmentAuthority(
            authority_id="collision",
            dataset_id="fixture-eeg",
            split_unit="session",
            assessment_indices=(4, 5),
            processed_data_sha256=_sha("processed-data"),
            n_samples=8,
            source_authority_fingerprint="three-way-source",
            metric_names=("balanced_accuracy",),
            metadata={1: "integer", "1": "string"},
        )

    with pytest.raises(ValueError, match="NaN or infinity"):
        FinalAssessmentAuthority(
            authority_id="nan",
            dataset_id="fixture-eeg",
            split_unit="session",
            assessment_indices=(4, 5),
            processed_data_sha256=_sha("processed-data"),
            n_samples=8,
            source_authority_fingerprint="three-way-source",
            metric_names=("balanced_accuracy",),
            metadata={"threshold": float("nan")},
        )


def test_identical_assessment_replay_has_identical_fingerprints():
    def run():
        final = _final_authority()
        selected = SelectedState.frozen(
            selection_id="baseline/predeclared",
            source_authority_fingerprint="three-way-source",
            artifact=_artifact("frozen"),
        )
        record = FinalAssessmentRecord.record(
            selected,
            authority=final,
            assessment_indices=final.assessment_indices,
            metrics={"balanced_accuracy": 0.62, "ece": 0.12},
        )
        return final, selected, record

    first = run()
    second = run()
    assert first[0].authority_fingerprint == second[0].authority_fingerprint
    assert first[1].selection_fingerprint == second[1].selection_fingerprint
    assert first[2].assessment_fingerprint == second[2].assessment_fingerprint
