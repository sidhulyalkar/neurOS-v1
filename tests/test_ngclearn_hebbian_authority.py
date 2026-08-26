from __future__ import annotations

import pytest

from neuros.foundation_models.ngclearn_bridge import _array_sha256, _matrix
from scripts.evidence.run_ngclearn_hebbian_authority import (
    _fixture,
    run_governed_hebbian_adaptation,
)


def _run(*, minimum_mse_improvement: float = 0.0):
    samples, authority, model = _fixture()
    payload = run_governed_hebbian_adaptation(
        samples,
        authority=authority,
        loaded_processed_data_sha256=authority.processed_data_sha256,
        model=model,
        sample_rate_hz=250.0,
        epochs=2,
        minimum_mse_improvement=minimum_mse_improvement,
    )
    return samples, authority, model, payload


def test_real_upstream_authority_worker_binds_exact_calibration_and_evaluation_rows():
    pytest.importorskip("ngclearn")
    samples, authority, model, payload = _run()

    # Match the exact canonical matrix representation consumed by the bridge and
    # sliced by the authority worker rather than the caller's source dtype.
    canonical = _matrix(samples)
    adaptation = canonical[list(authority.adaptation_indices)]
    evaluation = canonical[list(authority.evaluation_indices)]
    assert payload["inputs"]["adaptation_input_sha256"] == _array_sha256(adaptation)
    assert payload["inputs"]["evaluation_input_sha256"] == _array_sha256(evaluation)
    assert (
        payload["learner_adaptation"]["adaptation_input_sha256"]
        == payload["inputs"]["adaptation_input_sha256"]
    )
    assert payload["application"]["adaptation_indices"] == list(authority.adaptation_indices)
    assert payload["evaluation"]["evaluation_indices"] == list(authority.evaluation_indices)
    assert payload["learner_adaptation"]["update_count"] == 6

    fingerprint = authority.authority_fingerprint
    assert payload["proposal"]["authority_fingerprint"] == fingerprint
    assert payload["decision"]["authority_fingerprint"] == fingerprint
    assert payload["application"]["authority_fingerprint"] == fingerprint
    assert payload["evaluation"]["authority_fingerprint"] == fingerprint
    assert payload["outcome"]["authority_fingerprint"] == fingerprint

    boundary = payload["claim_boundary"]
    assert boundary["orion_adaptation_authority_enforced"] is True
    assert boundary["exact_adaptation_rows_verified"] is True
    assert boundary["exact_evaluation_rows_verified"] is True
    assert boundary["held_out_evaluation_read_only"] is True
    assert boundary["real_dataset_qualified"] is False
    assert boundary["calibration_reduction_qualified"] is False

    # Approval evidence is calibration-only. Held-out performance cannot silently
    # become an input to the pre-adaptation approval decision.
    proposal_evidence = payload["proposal"]["proposal"]["evidence"]
    assert set(proposal_evidence) == {"calibration_reconstruction_mse"}

    active = model.snapshot_state()
    assert active.state_sha256 == payload["outcome"]["active_artifact"]["sha256"]


def test_authority_worker_is_deterministic_across_independent_real_upstream_runs():
    pytest.importorskip("ngclearn")
    _, first_authority, _, first = _run()
    _, second_authority, _, second = _run()

    assert first_authority.authority_fingerprint == second_authority.authority_fingerprint
    assert first["evidence_sha256"] == second["evidence_sha256"]
    assert (
        first["learner_adaptation"]["evidence_sha256"]
        == second["learner_adaptation"]["evidence_sha256"]
    )
    assert (
        first["outcome"]["outcome_fingerprint"]
        == second["outcome"]["outcome_fingerprint"]
    )
    assert first == second


def test_predeclared_failed_held_out_gate_rolls_back_exact_complete_learning_state():
    pytest.importorskip("ngclearn")
    _, _, model, payload = _run(minimum_mse_improvement=1_000_000.0)

    assert payload["outcome"]["phase"] == "rolled-back"
    assert (
        payload["outcome"]["active_artifact"]["sha256"]
        == payload["application"]["before_artifact"]["sha256"]
    )
    assert (
        payload["outcome"]["active_artifact"]["sha256"]
        != payload["application"]["after_artifact"]["sha256"]
    )
    assert (
        model.snapshot_state().state_sha256
        == payload["outcome"]["active_artifact"]["sha256"]
    )


def test_processed_data_identity_mismatch_fails_before_learner_runtime_is_initialized():
    pytest.importorskip("ngclearn")
    samples, authority, model = _fixture()

    with pytest.raises(ValueError, match="processed-data SHA-256"):
        run_governed_hebbian_adaptation(
            samples,
            authority=authority,
            loaded_processed_data_sha256="0" * 64,
            model=model,
            sample_rate_hz=250.0,
            epochs=1,
        )

    with pytest.raises(RuntimeError, match="before the first infer/adapt"):
        model.snapshot_state()


def test_invalid_retention_threshold_fails_closed():
    pytest.importorskip("ngclearn")
    samples, authority, model = _fixture()
    with pytest.raises(ValueError, match="minimum_mse_improvement"):
        run_governed_hebbian_adaptation(
            samples,
            authority=authority,
            loaded_processed_data_sha256=authority.processed_data_sha256,
            model=model,
            sample_rate_hz=250.0,
            minimum_mse_improvement=-1.0,
        )
