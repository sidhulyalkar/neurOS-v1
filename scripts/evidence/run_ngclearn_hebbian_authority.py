"""Compose ngc-learn Hebbian learning with ORION adaptation authority.

This evidence worker deliberately lives outside both packages. ``neuros-foundation``
owns the optional ngc-learn mechanism while ORION owns mutation authority. The worker
is the seam where the two independent contracts are bound and audited.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

import numpy as np

from orion import (
    AdaptationApplication,
    AdaptationAuthority,
    AdaptationDecision,
    AdaptationEvaluation,
    AdaptationOutcome,
    AdaptationPhase,
    AdaptationProposal,
    ArtifactIdentity,
    GovernedAdaptationProposal,
)
from neuros.foundation_models.ngclearn_bridge import _array_sha256, _matrix
from neuros.foundation_models.ngclearn_hebbian import NgcLearnHebbianPredictiveCoding


def _canonical_sha256(payload: dict[str, Any]) -> str:
    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _artifact(
    *,
    authority: AdaptationAuthority,
    phase: str,
    state: Any,
    model: NgcLearnHebbianPredictiveCoding,
) -> ArtifactIdentity:
    return ArtifactIdentity(
        artifact_id=f"ngclearn-hebbian/{authority.authority_id}/{phase}",
        artifact_type="ngclearn-hebbian-learning-state",
        sha256=state.state_sha256,
        version="1",
        metadata={
            "method_id": "neuros-ngclearn-hebbian-predictive-reconstruction-v1",
            "weights_sha256": state.weights_sha256,
            "optimizer_sha256": state.optimizer_sha256,
            "optimizer": model.optimizer,
            "latent_dim": model.latent_dim,
        },
    )


def run_governed_hebbian_adaptation(
    samples: Any,
    *,
    authority: AdaptationAuthority,
    loaded_processed_data_sha256: str,
    model: NgcLearnHebbianPredictiveCoding,
    sample_rate_hz: float,
    epochs: int = 1,
    minimum_mse_improvement: float = 0.0,
) -> dict[str, Any]:
    """Run one authority-bound Hebbian adaptation and held-out decision.

    ``loaded_processed_data_sha256`` is supplied by the data-authority layer. It
    can therefore carry the richer ``LongitudinalCaseAuthority`` hash semantics
    rather than forcing this worker to redefine processed-dataset identity.

    The learner sees only ``authority.adaptation_indices``. Held-out rows are
    used for read-only before/after evaluation and may determine retain versus
    rollback only after mutation has finished.
    """

    matrix = _matrix(samples)
    if matrix.shape[0] != authority.n_samples:
        raise ValueError(
            "sample count differs from adaptation authority: "
            f"authority={authority.n_samples}, loaded={matrix.shape[0]}"
        )
    if loaded_processed_data_sha256 != authority.processed_data_sha256:
        raise ValueError("loaded processed-data SHA-256 differs from adaptation authority")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("epochs must be a positive integer")
    if isinstance(minimum_mse_improvement, bool) or not isinstance(
        minimum_mse_improvement, (int, float)
    ):
        raise ValueError("minimum_mse_improvement must be numeric")
    minimum_mse_improvement = float(minimum_mse_improvement)
    if not math.isfinite(minimum_mse_improvement) or minimum_mse_improvement < 0.0:
        raise ValueError("minimum_mse_improvement must be finite and non-negative")

    adaptation_indices = authority.require_adaptation_indices(authority.adaptation_indices)
    evaluation_indices = authority.require_evaluation_indices(authority.evaluation_indices)
    adaptation_matrix = np.ascontiguousarray(
        matrix[np.asarray(adaptation_indices, dtype=np.int64)]
    )
    evaluation_matrix = np.ascontiguousarray(
        matrix[np.asarray(evaluation_indices, dtype=np.int64)]
    )
    adaptation_input_sha256 = _array_sha256(adaptation_matrix)
    evaluation_input_sha256 = _array_sha256(evaluation_matrix)

    # Calibration evidence may authorize a proposal. Held-out evidence is kept
    # out of proposal/approval semantics and is used only for final evaluation.
    calibration_before = model.infer(adaptation_matrix, sample_rate_hz=sample_rate_hz)
    evaluation_before = model.infer(evaluation_matrix, sample_rate_hz=sample_rate_hz)
    before_state = model.snapshot_state()
    if calibration_before.state_sha256 != before_state.state_sha256:
        raise RuntimeError("calibration inference mutated learning state")
    if evaluation_before.state_sha256 != before_state.state_sha256:
        raise RuntimeError("held-out baseline inference mutated learning state")

    before_artifact = _artifact(
        authority=authority,
        phase="before",
        state=before_state,
        model=model,
    )
    proposal = AdaptationProposal(
        reason="authorized Hebbian calibration under frozen adaptation authority",
        changes={
            "method": "ngclearn-hebbian-predictive-reconstruction-v1",
            "epochs": epochs,
            "learning_rate": model.learning_rate,
            "optimizer": model.optimizer,
            "settling_steps": model.settling_steps,
        },
        evidence={"calibration_reconstruction_mse": calibration_before.mean_squared_error},
        requires_approval=True,
    )
    governed = GovernedAdaptationProposal.bind(
        proposal,
        authority=authority,
        before_artifact=before_artifact,
        adaptation_indices=adaptation_indices,
    )
    decision = AdaptationDecision.approve(
        governed,
        actor="evidence-worker/frozen-authority",
        reason="proposal is restricted to the complete frozen calibration partition",
    )

    adaptation = model.adapt(
        adaptation_matrix,
        sample_rate_hz=sample_rate_hz,
        epochs=epochs,
    )
    if adaptation.evidence.adaptation_input_sha256 != adaptation_input_sha256:
        raise RuntimeError(
            "learner adaptation-input SHA-256 differs from authority-selected calibration rows"
        )
    if adaptation.evidence.state_before_sha256 != before_state.state_sha256:
        raise RuntimeError("learner pre-update state differs from governed proposal artifact")

    after_artifact = _artifact(
        authority=authority,
        phase="after",
        state=adaptation.state_after,
        model=model,
    )
    application = AdaptationApplication.record(
        governed,
        decision,
        authority=authority,
        after_artifact=after_artifact,
        adaptation_indices=adaptation_indices,
        update_evidence={
            "update_count": float(adaptation.evidence.update_count),
            "weight_delta_l2": adaptation.evidence.weight_delta_l2,
        },
    )

    evaluation_after = model.infer(evaluation_matrix, sample_rate_hz=sample_rate_hz)
    if evaluation_after.state_sha256 != adaptation.state_after.state_sha256:
        raise RuntimeError("held-out evaluation mutated the post-adaptation learning state")
    evaluation = AdaptationEvaluation.record(
        application,
        authority=authority,
        evaluation_indices=evaluation_indices,
        metrics_before={"reconstruction_mse": evaluation_before.mean_squared_error},
        metrics_after={"reconstruction_mse": evaluation_after.mean_squared_error},
    )

    improvement = evaluation_before.mean_squared_error - evaluation_after.mean_squared_error
    if improvement >= minimum_mse_improvement:
        outcome = AdaptationOutcome.retain(
            application,
            evaluation,
            actor="policy/held-out-reconstruction",
            reason=(
                "held-out reconstruction MSE met the predeclared minimum improvement "
                f"threshold ({minimum_mse_improvement:.12g})"
            ),
        )
    else:
        model.restore_state(adaptation.state_before)
        restored_state = model.snapshot_state()
        if restored_state.state_sha256 != before_state.state_sha256:
            raise RuntimeError("rollback did not restore the exact governed pre-update state")
        restored_artifact = _artifact(
            authority=authority,
            phase="rollback",
            state=restored_state,
            model=model,
        )
        outcome = AdaptationOutcome.rollback(
            application,
            evaluation,
            restored_artifact=restored_artifact,
            actor="policy/held-out-reconstruction",
            reason=(
                "held-out reconstruction MSE failed the predeclared minimum improvement "
                f"threshold ({minimum_mse_improvement:.12g})"
            ),
        )

    active_state = model.snapshot_state()
    if active_state.state_sha256 != outcome.active_artifact.sha256:
        raise RuntimeError("live learner state differs from finalized adaptation outcome")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "authority": authority.to_dict(),
        "inputs": {
            "loaded_matrix_sha256": _array_sha256(matrix),
            "loaded_processed_data_sha256": loaded_processed_data_sha256,
            "adaptation_input_sha256": adaptation_input_sha256,
            "evaluation_input_sha256": evaluation_input_sha256,
            "adaptation_indices": list(adaptation_indices),
            "evaluation_indices": list(evaluation_indices),
        },
        "proposal": governed.to_dict(),
        "decision": decision.to_dict(),
        "learner_adaptation": adaptation.evidence.to_dict(),
        "application": application.to_dict(),
        "evaluation": evaluation.to_dict(),
        "held_out_mse_improvement": improvement,
        "minimum_mse_improvement": minimum_mse_improvement,
        "outcome": outcome.to_dict(),
        "claim_boundary": {
            "orion_adaptation_authority_enforced": True,
            "exact_adaptation_rows_verified": True,
            "exact_evaluation_rows_verified": True,
            "held_out_evaluation_read_only": True,
            "retain_or_exact_rollback_exercised": True,
            "real_dataset_qualified": False,
            "calibration_reduction_qualified": False,
            "cross_subject_transfer_qualified": False,
            "online_adaptation_qualified": False,
            "stdp_learning_qualified": False,
            "hardware_qualified": False,
            "closed_loop_qualified": False,
            "clinical_qualified": False,
        },
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    return payload


def _fixture() -> tuple[np.ndarray, AdaptationAuthority, NgcLearnHebbianPredictiveCoding]:
    samples = np.asarray(
        [
            [0.40, -0.20],
            [0.15, 0.35],
            [-0.30, 0.10],
            [0.38, -0.18],
            [0.18, 0.32],
            [-0.28, 0.12],
        ],
        dtype=np.float32,
    )
    authority = AdaptationAuthority(
        authority_id="fixture/session-02/budget-3",
        dataset_id="ngclearn-hebbian-authority-fixture",
        split_unit="session",
        adaptation_indices=(0, 1, 2),
        evaluation_indices=(3, 4, 5),
        # The fixture treats the canonical float64 bridge matrix as its processed
        # data authority. Real longitudinal studies may carry a richer hash that
        # also includes labels/group identities and pass that verified identity in.
        processed_data_sha256=_array_sha256(_matrix(samples)),
        n_samples=len(samples),
        protocol_fingerprint="fixture-protocol-v1",
        source_authority_fingerprint="fixture-longitudinal-authority-v1",
        seed=17,
    )
    model = NgcLearnHebbianPredictiveCoding(
        latent_dim=2,
        settling_steps=16,
        settling_dt_ms=1.0,
        tau_m_ms=12.0,
        learning_rate=0.005,
        optimizer="adam",
        sign_value=-1.0,
        weight_bound=0.0,
        seed=17,
        weights=np.asarray([[0.30, 0.00], [0.00, 0.30]], dtype=np.float32),
    )
    return samples, authority, model


def main() -> None:
    samples, authority, model = _fixture()
    payload = run_governed_hebbian_adaptation(
        samples,
        authority=authority,
        loaded_processed_data_sha256=authority.processed_data_sha256,
        model=model,
        sample_rate_hz=250.0,
        epochs=2,
        minimum_mse_improvement=0.0,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
