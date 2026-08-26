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
    AdaptationPhase,
    ArtifactIdentity,
    GovernedAdaptationProposal,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _authority() -> AdaptationAuthority:
    return AdaptationAuthority(
        authority_id="integrity-fixture",
        dataset_id="fixture",
        split_unit="session",
        adaptation_indices=(0, 2),
        evaluation_indices=(1, 3),
        processed_data_sha256=_sha("data"),
        n_samples=4,
    )


def _artifact(label: str) -> ArtifactIdentity:
    return ArtifactIdentity(artifact_id=label, sha256=_sha(label))


def test_directly_forged_governed_authority_cannot_be_applied():
    authority = _authority()
    proposal = AdaptationProposal(reason="fixture", changes={"step": 1})
    forged = GovernedAdaptationProposal(
        proposal=proposal,
        authority_fingerprint="forged-authority",
        before_artifact=_artifact("before"),
        adaptation_indices=authority.adaptation_indices,
    )
    # Even a manually constructed decision that points at the real authority
    # cannot launder the forged governed record into an application.
    decision = AdaptationDecision(
        proposal_fingerprint=forged.proposal_fingerprint,
        authority_fingerprint=authority.authority_fingerprint,
        phase=AdaptationPhase.APPROVED,
        actor="adversarial-test",
        reason="attempt forged transition",
    )

    with pytest.raises(ValueError, match="governed proposal authority"):
        AdaptationApplication.record(
            forged,
            decision,
            authority=authority,
            after_artifact=_artifact("after"),
            adaptation_indices=authority.adaptation_indices,
        )


def test_directly_forged_application_indices_cannot_reach_evaluation():
    authority = _authority()
    forged = AdaptationApplication(
        proposal_fingerprint="proposal",
        authority_fingerprint=authority.authority_fingerprint,
        decision_fingerprint="decision",
        before_artifact=_artifact("before"),
        after_artifact=_artifact("after"),
        adaptation_indices=(0,),
    )

    with pytest.raises(ValueError, match="exact authority adaptation indices"):
        AdaptationEvaluation.record(
            forged,
            authority=authority,
            evaluation_indices=authority.evaluation_indices,
            metrics_before={"accuracy": 0.5},
            metrics_after={"accuracy": 0.6},
        )


def test_forged_evaluation_authority_cannot_finalize_outcome():
    authority = _authority()
    application = AdaptationApplication(
        proposal_fingerprint="proposal",
        authority_fingerprint=authority.authority_fingerprint,
        decision_fingerprint="decision",
        before_artifact=_artifact("before"),
        after_artifact=_artifact("after"),
        adaptation_indices=authority.adaptation_indices,
    )
    forged_evaluation = AdaptationEvaluation(
        application_fingerprint=application.application_fingerprint,
        authority_fingerprint="forged-authority",
        evaluation_indices=authority.evaluation_indices,
        metrics_before={"accuracy": 0.5},
        metrics_after={"accuracy": 0.6},
    )

    with pytest.raises(ValueError, match="evaluation authority"):
        AdaptationOutcome.retain(
            application,
            forged_evaluation,
            actor="adversarial-test",
            reason="attempt forged finalization",
        )
