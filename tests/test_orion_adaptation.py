from __future__ import annotations

import hashlib

import numpy as np
import pytest

from orion import AdaptationProposal
from orion.adaptation import (
    AdaptationApplication,
    AdaptationAuthority,
    AdaptationDecision,
    AdaptationEvaluation,
    AdaptationOutcome,
    AdaptationPhase,
    ArtifactIdentity,
    GovernedAdaptationProposal,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _authority() -> AdaptationAuthority:
    return AdaptationAuthority(
        authority_id="subject-07/session-02/calibration-v1",
        dataset_id="fixture-eeg",
        split_unit="session",
        adaptation_indices=(0, 2, 4, 6),
        evaluation_indices=(1, 3, 5, 7),
        processed_data_sha256=_sha("processed-data"),
        n_samples=8,
        protocol_fingerprint="protocol-1234",
        source_authority_fingerprint="longitudinal-5678",
        seed=9,
    )


def _artifact(label: str) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=f"decoder/{label}",
        artifact_type="decoder-state",
        sha256=_sha(label),
        metadata={"backend": "fixture", "shape": [2, 2]},
    )


def _governed() -> tuple[AdaptationAuthority, GovernedAdaptationProposal]:
    authority = _authority()
    proposal = AdaptationProposal(
        reason="target-session calibration",
        changes={"learning_rate": 0.01, "steps": 4},
        evidence={"calibration_loss": 0.42},
        requires_approval=True,
    )
    governed = GovernedAdaptationProposal.bind(
        proposal,
        authority=authority,
        before_artifact=_artifact("before"),
        adaptation_indices=(0, 2, 4, 6),
    )
    return authority, governed


def test_authority_rejects_adaptation_evaluation_leakage():
    with pytest.raises(ValueError, match="must be disjoint"):
        AdaptationAuthority(
            authority_id="bad",
            dataset_id="fixture",
            split_unit="session",
            adaptation_indices=(0, 1, 2),
            evaluation_indices=(2, 3),
            processed_data_sha256=_sha("data"),
            n_samples=4,
        )


def test_authority_requires_exact_frozen_indices_and_order():
    authority = _authority()

    assert authority.require_adaptation_indices((0, 2, 4, 6)) == (0, 2, 4, 6)
    assert authority.require_evaluation_indices((1, 3, 5, 7)) == (1, 3, 5, 7)

    with pytest.raises(ValueError, match="exact authority adaptation indices"):
        authority.require_adaptation_indices((0, 2))
    with pytest.raises(ValueError, match="exact frozen evaluation indices"):
        authority.require_evaluation_indices((1, 3, 5))
    with pytest.raises(ValueError, match="exact frozen evaluation indices"):
        authority.require_evaluation_indices((7, 5, 3, 1))


def test_rejected_proposal_cannot_mutate_state():
    authority, governed = _governed()
    rejected = AdaptationDecision.reject(
        governed,
        actor="policy/calibration-budget",
        reason="calibration error did not cross update threshold",
    )

    assert rejected.phase is AdaptationPhase.REJECTED
    with pytest.raises(ValueError, match="rejected adaptation cannot be applied"):
        AdaptationApplication.record(
            governed,
            rejected,
            authority=authority,
            after_artifact=_artifact("after"),
            adaptation_indices=authority.adaptation_indices,
        )


def test_approved_application_binds_before_after_and_authorized_samples():
    authority, governed = _governed()
    approved = AdaptationDecision.approve(
        governed,
        actor="operator/sid",
        reason="approved for frozen calibration partition",
    )
    application = AdaptationApplication.record(
        governed,
        approved,
        authority=authority,
        after_artifact=_artifact("after"),
        adaptation_indices=authority.adaptation_indices,
        update_evidence={"updates": 4, "weight_delta_l2": 0.125},
    )

    assert application.before_artifact.sha256 == _sha("before")
    assert application.after_artifact.sha256 == _sha("after")
    assert application.adaptation_indices == authority.adaptation_indices
    assert application.update_evidence["updates"] == pytest.approx(4.0)

    with pytest.raises(ValueError, match="exact authority adaptation indices"):
        AdaptationApplication.record(
            governed,
            approved,
            authority=authority,
            after_artifact=_artifact("different-after"),
            adaptation_indices=(0, 2, 4, 7),
        )


def test_application_cannot_claim_a_noop_as_an_applied_update():
    authority, governed = _governed()
    approved = AdaptationDecision.approve(
        governed,
        actor="operator/sid",
        reason="apply fixture",
    )
    same_state = ArtifactIdentity(
        artifact_id="decoder/copied-name",
        artifact_type="decoder-state",
        sha256=governed.before_artifact.sha256,
    )

    with pytest.raises(ValueError, match="must change the artifact SHA-256"):
        AdaptationApplication.record(
            governed,
            approved,
            authority=authority,
            after_artifact=same_state,
            adaptation_indices=authority.adaptation_indices,
        )


def test_evaluation_cannot_use_adaptation_rows_or_cherry_pick_subset():
    authority, governed = _governed()
    approved = AdaptationDecision.approve(
        governed,
        actor="operator/sid",
        reason="apply fixture",
    )
    application = AdaptationApplication.record(
        governed,
        approved,
        authority=authority,
        after_artifact=_artifact("after"),
        adaptation_indices=authority.adaptation_indices,
    )

    with pytest.raises(ValueError, match="exact frozen evaluation indices"):
        AdaptationEvaluation.record(
            application,
            authority=authority,
            evaluation_indices=(0, 1, 3, 5, 7),
            metrics_before={"accuracy": 0.60},
            metrics_after={"accuracy": 0.70},
        )

    with pytest.raises(ValueError, match="exact frozen evaluation indices"):
        AdaptationEvaluation.record(
            application,
            authority=authority,
            evaluation_indices=(1, 3, 5),
            metrics_before={"accuracy": 0.60},
            metrics_after={"accuracy": 0.70},
        )


def test_rollback_restores_exact_pre_adaptation_identity():
    authority, governed = _governed()
    approved = AdaptationDecision.approve(
        governed,
        actor="operator/sid",
        reason="apply fixture",
    )
    application = AdaptationApplication.record(
        governed,
        approved,
        authority=authority,
        after_artifact=_artifact("after"),
        adaptation_indices=authority.adaptation_indices,
    )
    evaluation = AdaptationEvaluation.record(
        application,
        authority=authority,
        evaluation_indices=authority.evaluation_indices,
        metrics_before={"accuracy": 0.72, "ece": 0.08},
        metrics_after={"accuracy": 0.69, "ece": 0.11},
    )

    outcome = AdaptationOutcome.rollback(
        application,
        evaluation,
        restored_artifact=governed.before_artifact,
        actor="policy/held-out-regression",
        reason="held-out accuracy and calibration regressed",
    )
    assert outcome.phase is AdaptationPhase.ROLLED_BACK
    assert outcome.active_artifact.sha256 == governed.before_artifact.sha256

    wrong_restore = _artifact("not-the-before-state")
    with pytest.raises(ValueError, match="exact pre-adaptation artifact"):
        AdaptationOutcome.rollback(
            application,
            evaluation,
            restored_artifact=wrong_restore,
            actor="policy/held-out-regression",
            reason="attempt invalid rollback",
        )


def test_retain_keeps_exact_post_adaptation_identity():
    authority, governed = _governed()
    approved = AdaptationDecision.approve(
        governed,
        actor="operator/sid",
        reason="apply fixture",
    )
    application = AdaptationApplication.record(
        governed,
        approved,
        authority=authority,
        after_artifact=_artifact("after"),
        adaptation_indices=authority.adaptation_indices,
    )
    evaluation = AdaptationEvaluation.record(
        application,
        authority=authority,
        evaluation_indices=authority.evaluation_indices,
        metrics_before={"accuracy": 0.61, "ece": 0.12},
        metrics_after={"accuracy": 0.74, "ece": 0.07},
    )
    outcome = AdaptationOutcome.retain(
        application,
        evaluation,
        actor="policy/held-out-improvement",
        reason="held-out utility improved without calibration regression",
    )

    assert outcome.phase is AdaptationPhase.RETAINED
    assert outcome.active_artifact.sha256 == application.after_artifact.sha256


def test_identical_replay_produces_identical_authority_and_evidence_fingerprints():
    first_authority, first_governed = _governed()
    second_authority, second_governed = _governed()

    assert first_authority.authority_fingerprint == second_authority.authority_fingerprint
    assert first_governed.proposal_fingerprint == second_governed.proposal_fingerprint

    def run(authority: AdaptationAuthority, governed: GovernedAdaptationProposal):
        decision = AdaptationDecision.approve(
            governed,
            actor="policy/replay",
            reason="deterministic fixture approval",
        )
        application = AdaptationApplication.record(
            governed,
            decision,
            authority=authority,
            after_artifact=_artifact("after"),
            adaptation_indices=authority.adaptation_indices,
            update_evidence={"updates": 4, "weight_delta_l2": 0.125},
        )
        evaluation = AdaptationEvaluation.record(
            application,
            authority=authority,
            evaluation_indices=authority.evaluation_indices,
            metrics_before={"accuracy": 0.61},
            metrics_after={"accuracy": 0.74},
        )
        outcome = AdaptationOutcome.retain(
            application,
            evaluation,
            actor="policy/replay",
            reason="deterministic held-out result",
        )
        return decision, application, evaluation, outcome

    first = run(first_authority, first_governed)
    second = run(second_authority, second_governed)

    assert first[0].decision_fingerprint == second[0].decision_fingerprint
    assert first[1].application_fingerprint == second[1].application_fingerprint
    assert first[2].evaluation_fingerprint == second[2].evaluation_fingerprint
    assert first[3].outcome_fingerprint == second[3].outcome_fingerprint
