"""Dependency-light demonstration of ORION adaptation authority.

This example simulates artifact identities rather than training a real decoder.
Its purpose is to show the governed state-transition and evidence surface that
real ORION personalization or external local-learning rules must use.
"""

from __future__ import annotations

import hashlib
import json

from orion import (
    AdaptationApplication,
    AdaptationAuthority,
    AdaptationDecision,
    AdaptationEvaluation,
    AdaptationOutcome,
    AdaptationProposal,
    ArtifactIdentity,
    GovernedAdaptationProposal,
)


def sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def main() -> None:
    authority = AdaptationAuthority(
        authority_id="demo-subject/session-2/calibration-4",
        dataset_id="orion-adaptation-fixture",
        split_unit="session",
        adaptation_indices=(0, 2, 4, 6),
        evaluation_indices=(1, 3, 5, 7),
        processed_data_sha256=sha("fixture-data"),
        n_samples=8,
        protocol_fingerprint="demo-session-disjoint-v1",
        source_authority_fingerprint="fixture-longitudinal-authority",
        seed=7,
    )
    before = ArtifactIdentity(
        artifact_id="decoder/pre-adaptation",
        artifact_type="decoder-state",
        sha256=sha("decoder-before"),
    )
    proposal = GovernedAdaptationProposal.bind(
        AdaptationProposal(
            reason="target-session calibration",
            changes={"learning_rate": 0.001, "steps": 4},
            evidence={"calibration_error": 0.31},
            requires_approval=True,
        ),
        authority=authority,
        before_artifact=before,
        adaptation_indices=authority.adaptation_indices,
    )
    decision = AdaptationDecision.approve(
        proposal,
        actor="policy/demo",
        reason="fixture calibration budget authorized",
    )
    after = ArtifactIdentity(
        artifact_id="decoder/post-adaptation",
        artifact_type="decoder-state",
        sha256=sha("decoder-after"),
    )
    application = AdaptationApplication.record(
        proposal,
        decision,
        authority=authority,
        after_artifact=after,
        adaptation_indices=authority.adaptation_indices,
        update_evidence={"updates": 4, "parameter_delta_l2": 0.14},
    )
    evaluation = AdaptationEvaluation.record(
        application,
        authority=authority,
        evaluation_indices=authority.evaluation_indices,
        metrics_before={"accuracy": 0.625, "ece": 0.12},
        metrics_after={"accuracy": 0.750, "ece": 0.08},
    )
    outcome = AdaptationOutcome.retain(
        application,
        evaluation,
        actor="policy/demo",
        reason="held-out accuracy improved and ECE decreased",
    )

    payload = {
        "authority": authority.to_dict(),
        "proposal": proposal.to_dict(),
        "decision": decision.to_dict(),
        "application": application.to_dict(),
        "evaluation": evaluation.to_dict(),
        "outcome": outcome.to_dict(),
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
