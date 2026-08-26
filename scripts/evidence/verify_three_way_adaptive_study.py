#!/usr/bin/env python3
"""Verify end-to-end three-way authority binding across foundation and ORION.

This worker is intentionally synthetic. It proves that a dataset-specific
``ThreeWayLongitudinalCaseAuthority`` can derive method-level ORION calibration /
qualification authority plus a separate final-assessment authority without
reversing package dependencies or re-partitioning data inside the method.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from neuros.foundation_models import (
    GroupedEvaluationData,
    ThreeWayLongitudinalCaseAuthority,
    chronological_partition,
    make_three_way_calibration_split,
)
from orion import (
    AdaptationApplication,
    AdaptationAuthority,
    AdaptationDecision,
    AdaptationEvaluation,
    AdaptationOutcome,
    AdaptationProposal,
    AdaptiveStudyAuthority,
    ArtifactIdentity,
    FinalAssessmentAuthority,
    FinalAssessmentRecord,
    GovernedAdaptationProposal,
    SelectedState,
)

METRIC_NAMES = ("balanced_accuracy", "ece", "brier")


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _fixture_data() -> GroupedEvaluationData:
    rng = np.random.default_rng(211)
    X: list[np.ndarray] = []
    y: list[str] = []
    metadata: list[dict[str, str]] = []
    for session in ("0", "1", "2"):
        for label in ("left", "right"):
            for trial in range(16):
                X.append(rng.normal(size=(4, 24)))
                y.append(label)
                metadata.append(
                    {
                        "subject": "1",
                        "session": session,
                        "run": f"run-{trial // 8}",
                    }
                )
    return GroupedEvaluationData.from_moabb_result(
        (np.asarray(X), np.asarray(y), metadata),
        dataset_id="synthetic-three-way-adaptive-study",
    )


def build_source_authority() -> ThreeWayLongitudinalCaseAuthority:
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
        seed=23,
    )
    return ThreeWayLongitudinalCaseAuthority.from_split(
        split,
        case_id="subject-1/session-2/three-way-v2",
        history_policy="prior",
        case_metadata={"purpose": "cross-package authority contract"},
    )


def derive_adaptive_study(
    source: ThreeWayLongitudinalCaseAuthority,
    *,
    budget_per_class: int,
) -> AdaptiveStudyAuthority:
    """Derive ORION authority from a frozen v2 case without choosing new rows."""

    calibration = source.calibration_indices(budget_per_class)
    if not calibration:
        raise ValueError(
            "adaptive study authority requires a positive calibration budget; "
            "represent budget 0 as a frozen SelectedState instead"
        )

    adaptation = AdaptationAuthority(
        authority_id=f"{source.case_id}/budget-{budget_per_class}/adaptation",
        dataset_id=source.dataset_id,
        split_unit=source.split_unit,
        adaptation_indices=calibration,
        evaluation_indices=source.qualification_indices,
        processed_data_sha256=source.processed_data_sha256,
        n_samples=source.n_samples,
        protocol_fingerprint=source.three_way_split_fingerprint,
        source_authority_fingerprint=source.authority_fingerprint,
        seed=source.seed,
        metadata={
            "budget_per_class": budget_per_class,
            "role": "calibration-plus-qualification",
        },
    )
    final = FinalAssessmentAuthority(
        authority_id=f"{source.case_id}/final-assessment",
        dataset_id=source.dataset_id,
        split_unit=source.split_unit,
        assessment_indices=source.final_assessment_indices,
        processed_data_sha256=source.processed_data_sha256,
        n_samples=source.n_samples,
        source_authority_fingerprint=source.authority_fingerprint,
        metric_names=METRIC_NAMES,
        protocol_fingerprint=source.three_way_split_fingerprint,
        seed=source.seed,
        metadata={
            "role": "untouched-final-assessment",
            "metric_policy": "predeclared synthetic contract scorecard",
        },
    )
    return AdaptiveStudyAuthority(
        study_id=f"{source.case_id}/budget-{budget_per_class}",
        source_authority_fingerprint=source.authority_fingerprint,
        adaptation_authority=adaptation,
        final_assessment_authority=final,
    )


def _artifact(label: str) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=f"synthetic-decoder/{label}",
        artifact_type="decoder-state",
        sha256=_sha(label),
        metadata={"backend": "synthetic-contract"},
    )


def run_contract() -> dict[str, Any]:
    source = build_source_authority()
    budget_one = derive_adaptive_study(source, budget_per_class=1)
    budget_three = derive_adaptive_study(source, budget_per_class=3)

    # Calibration identity may change with budget. Qualification and final
    # assessment are inherited unchanged from the source v2 authority.
    assert budget_one.adaptation_authority.adaptation_indices == source.calibration_indices(1)
    assert budget_three.adaptation_authority.adaptation_indices == source.calibration_indices(3)
    assert (
        budget_one.adaptation_authority.evaluation_indices
        == budget_three.adaptation_authority.evaluation_indices
        == source.qualification_indices
    )
    assert (
        budget_one.final_assessment_authority.assessment_indices
        == budget_three.final_assessment_authority.assessment_indices
        == source.final_assessment_indices
    )
    assert (
        budget_one.final_assessment_authority.authority_fingerprint
        == budget_three.final_assessment_authority.authority_fingerprint
    )

    # Budget 0 is intentionally a frozen baseline, not an empty adaptation.
    frozen = SelectedState.frozen(
        selection_id=f"{source.case_id}/baseline/budget-0",
        source_authority_fingerprint=source.authority_fingerprint,
        artifact=_artifact("frozen-baseline"),
        metadata={"calibration_examples": 0, "selection": "predeclared"},
    )
    frozen_final = FinalAssessmentRecord.record(
        frozen,
        authority=budget_one.final_assessment_authority,
        assessment_indices=source.final_assessment_indices,
        metrics={"balanced_accuracy": 0.61, "ece": 0.13, "brier": 0.24},
    )

    # Positive-budget adaptive path: calibration -> qualification -> selection
    # -> final assessment. Numbers are synthetic and have no efficacy meaning.
    authority = budget_one.adaptation_authority
    proposal = AdaptationProposal(
        reason="synthetic positive-budget adaptation",
        changes={"learning_rate": 0.01, "epochs": 1},
        evidence={"calibration_samples": len(authority.adaptation_indices)},
        requires_approval=True,
    )
    governed = GovernedAdaptationProposal.bind(
        proposal,
        authority=authority,
        before_artifact=_artifact("adaptive-before"),
        adaptation_indices=authority.adaptation_indices,
    )
    decision = AdaptationDecision.approve(
        governed,
        actor="synthetic-study-policy",
        reason="positive calibration budget authorized",
    )
    application = AdaptationApplication.record(
        governed,
        decision,
        authority=authority,
        after_artifact=_artifact("adaptive-after"),
        adaptation_indices=authority.adaptation_indices,
        update_evidence={"updates": len(authority.adaptation_indices)},
    )
    qualification = AdaptationEvaluation.record(
        application,
        authority=authority,
        evaluation_indices=source.qualification_indices,
        metrics_before={"balanced_accuracy": 0.58, "ece": 0.16},
        metrics_after={"balanced_accuracy": 0.66, "ece": 0.11},
    )
    outcome = AdaptationOutcome.retain(
        application,
        qualification,
        actor="synthetic-study-policy",
        reason="predeclared qualification policy retained the update",
    )
    selected = budget_one.select_outcome(
        outcome,
        selection_id=f"{source.case_id}/adaptive/budget-1",
        metadata={"state_frozen_before_final_assessment": True},
    )
    adapted_final = FinalAssessmentRecord.record(
        selected,
        authority=budget_one.final_assessment_authority,
        assessment_indices=source.final_assessment_indices,
        metrics={"balanced_accuracy": 0.64, "ece": 0.12, "brier": 0.22},
    )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "three_way_adaptive_study_contract",
        "source_authority": source.to_dict(),
        "budget_one_study": budget_one.to_dict(),
        "budget_three_study": budget_three.to_dict(),
        "frozen_baseline": {
            "selected_state": frozen.to_dict(),
            "final_assessment": frozen_final.to_dict(),
        },
        "adaptive_path": {
            "proposal": governed.to_dict(),
            "decision": decision.to_dict(),
            "application": application.to_dict(),
            "qualification": qualification.to_dict(),
            "outcome": outcome.to_dict(),
            "selected_state": selected.to_dict(),
            "final_assessment": adapted_final.to_dict(),
        },
        "invariants": {
            "source_authority_shared": True,
            "qualification_fixed_across_budgets": True,
            "final_assessment_fixed_across_budgets": True,
            "budget_zero_is_frozen_not_adapted": True,
            "final_scorecard_predeclared": list(METRIC_NAMES),
            "final_assessment_after_state_selection": True,
        },
        "claim_boundary": {
            "synthetic_contract_only": True,
            "real_neural_data": False,
            "adaptation_efficacy": False,
            "calibration_reduction": False,
            "orion_superiority": False,
            "hardware": False,
            "closed_loop": False,
            "clinical": False,
        },
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    print(json.dumps(run_contract(), sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
