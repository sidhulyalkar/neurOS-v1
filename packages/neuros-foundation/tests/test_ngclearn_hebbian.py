from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.ngclearn_bridge import _array_sha256, _matrix
from neuros.foundation_models.ngclearn_hebbian import (
    HEBBIAN_PC_METHOD_ID,
    NgcLearnHebbianPredictiveCoding,
    NgcLearnHebbianState,
    _state_sha256,
    _tree_sha256,
)


def _samples() -> np.ndarray:
    return np.asarray(
        [
            [0.40, -0.20],
            [0.15, 0.35],
            [-0.30, 0.10],
            [0.25, 0.45],
        ],
        dtype=np.float32,
    )


def _weights() -> np.ndarray:
    return np.asarray([[0.30, 0.00], [0.00, 0.30]], dtype=np.float32)


def _model(*, optimizer: str = "sgd") -> NgcLearnHebbianPredictiveCoding:
    return NgcLearnHebbianPredictiveCoding(
        latent_dim=2,
        settling_steps=16,
        settling_dt_ms=1.0,
        tau_m_ms=12.0,
        prior_gamma=0.0,
        activation="identity",
        learning_rate=0.005,
        optimizer=optimizer,
        sign_value=-1.0,
        weight_bound=0.0,
        seed=17,
        weights=_weights(),
    )


def test_hebbian_configuration_fails_closed_before_optional_runtime_import():
    with pytest.raises(ValueError, match="latent_dim"):
        NgcLearnHebbianPredictiveCoding(latent_dim=0)
    with pytest.raises(ValueError, match="settling_steps"):
        NgcLearnHebbianPredictiveCoding(settling_steps=0)
    with pytest.raises(ValueError, match="optimizer"):
        NgcLearnHebbianPredictiveCoding(optimizer="mystery")
    with pytest.raises(ValueError, match="learning_rate"):
        NgcLearnHebbianPredictiveCoding(learning_rate=0.0)
    with pytest.raises(ValueError, match="weight_bound"):
        NgcLearnHebbianPredictiveCoding(weight_bound=-1.0)
    with pytest.raises(ValueError, match="weights"):
        NgcLearnHebbianPredictiveCoding(latent_dim=2, weights=np.ones((3, 2)))


def test_real_upstream_hebbian_adaptation_changes_complete_state():
    pytest.importorskip("ngclearn")
    model = _model(optimizer="sgd")
    samples = _samples()

    inference = model.infer(samples[:2], sample_rate_hz=250.0)
    before = model.snapshot_state()
    assert inference.state_sha256 == before.state_sha256
    assert before.weights.dtype == np.dtype(np.float32)
    assert before.weights.flags.writeable is False

    result = model.adapt(samples[2:], sample_rate_hz=250.0, epochs=2)
    after = model.snapshot_state()

    assert result.evidence.method_id == HEBBIAN_PC_METHOD_ID
    assert result.evidence.ngclearn_version.startswith("3.2.")
    assert result.evidence.update_count == 4
    assert result.evidence.n_observations == 2
    assert result.evidence.epochs == 2
    assert "HebbianSynapse" in result.evidence.generative_synapse_component
    assert "GaussianErrorCell" in result.evidence.error_component
    assert result.evidence.row_normalization_after_update is False
    assert result.evidence.sign_value == pytest.approx(-1.0)
    assert result.evidence.state_before_sha256 == before.state_sha256
    assert result.evidence.state_after_sha256 == after.state_sha256
    assert before.state_sha256 != after.state_sha256
    assert before.weights_sha256 != after.weights_sha256
    assert before.optimizer_sha256 != after.optimizer_sha256
    assert result.evidence.weight_delta_l2 > 0.0
    assert result.evidence.adaptation_input_sha256 == _array_sha256(_matrix(samples[2:]))

    boundary = result.evidence.to_dict()["claim_boundary"]
    assert boundary["hebbian_synapse_executed"] is True
    assert boundary["state_identity_includes_optimizer"] is True
    assert boundary["transactional_checkpoint_validation"] is True
    assert boundary["optimizer_schema_validated_before_rollback"] is True
    assert boundary["orion_authority_applied_here"] is False
    assert boundary["real_dataset_qualified"] is False
    assert boundary["calibration_reduction_qualified"] is False
    assert boundary["stdp_learning_qualified"] is False


def test_inference_after_adaptation_is_strictly_read_only():
    pytest.importorskip("ngclearn")
    model = _model(optimizer="adam")
    samples = _samples()

    model.adapt(samples[:2], sample_rate_hz=250.0, epochs=1)
    before = model.snapshot_state()
    first = model.infer(samples[2:], sample_rate_hz=250.0)
    middle = model.snapshot_state()
    second = model.infer(samples[2:], sample_rate_hz=250.0)
    after = model.snapshot_state()

    assert before.state_sha256 == first.state_sha256 == middle.state_sha256
    assert middle.state_sha256 == second.state_sha256 == after.state_sha256
    assert np.array_equal(first.values, second.values)
    assert np.array_equal(first.reconstruction, second.reconstruction)
    assert first.mean_squared_error == pytest.approx(second.mean_squared_error, rel=0, abs=0)


def test_identical_models_and_data_produce_identical_learning_state():
    pytest.importorskip("ngclearn")
    samples = _samples()
    first = _model(optimizer="adam")
    second = _model(optimizer="adam")

    first_result = first.adapt(samples, sample_rate_hz=250.0, epochs=2)
    second_result = second.adapt(samples, sample_rate_hz=250.0, epochs=2)

    assert first_result.state_before.state_sha256 == second_result.state_before.state_sha256
    assert first_result.state_after.state_sha256 == second_result.state_after.state_sha256
    assert first_result.evidence.evidence_sha256 == second_result.evidence.evidence_sha256
    assert np.array_equal(first_result.state_after.weights, second_result.state_after.weights)


def test_adam_rollback_restores_weights_optimizer_and_future_trajectory():
    pytest.importorskip("ngclearn")
    samples = _samples()
    model = _model(optimizer="adam")
    reference = _model(optimizer="adam")

    model.infer(samples[:1], sample_rate_hz=250.0)
    reference.infer(samples[:1], sample_rate_hz=250.0)
    checkpoint = model.snapshot_state()
    reference_checkpoint = reference.snapshot_state()
    assert checkpoint.state_sha256 == reference_checkpoint.state_sha256

    first_adaptation = model.adapt(samples[:2], sample_rate_hz=250.0, epochs=2)
    assert first_adaptation.state_after.state_sha256 != checkpoint.state_sha256
    assert first_adaptation.state_after.optimizer_sha256 != checkpoint.optimizer_sha256

    model.restore_state(checkpoint)
    restored = model.snapshot_state()
    assert restored.state_sha256 == checkpoint.state_sha256
    assert restored.weights_sha256 == checkpoint.weights_sha256
    assert restored.optimizer_sha256 == checkpoint.optimizer_sha256
    assert np.array_equal(restored.weights, checkpoint.weights)

    replay = model.adapt(samples[:2], sample_rate_hz=250.0, epochs=2)
    reference_replay = reference.adapt(samples[:2], sample_rate_hz=250.0, epochs=2)
    assert replay.state_after.state_sha256 == first_adaptation.state_after.state_sha256
    assert replay.state_after.state_sha256 == reference_replay.state_after.state_sha256
    assert replay.evidence.evidence_sha256 == reference_replay.evidence.evidence_sha256


def test_corrupt_checkpoint_fails_before_live_state_is_modified():
    pytest.importorskip("ngclearn")
    model = _model(optimizer="adam")
    samples = _samples()
    model.adapt(samples[:2], sample_rate_hz=250.0, epochs=1)
    live_before = model.snapshot_state()

    corrupted_weights = np.array(live_before.weights, copy=True)
    corrupted_weights[0, 0] += np.asarray(0.125, dtype=corrupted_weights.dtype)
    corrupted = NgcLearnHebbianState(
        weights=corrupted_weights,
        optimizer_state=live_before.optimizer_state,
        weights_sha256=live_before.weights_sha256,
        optimizer_sha256=live_before.optimizer_sha256,
        state_sha256=live_before.state_sha256,
    )

    with pytest.raises(ValueError, match="weight SHA-256"):
        model.restore_state(corrupted)

    live_after = model.snapshot_state()
    assert live_after.state_sha256 == live_before.state_sha256
    assert np.array_equal(live_after.weights, live_before.weights)


def test_self_consistent_incompatible_optimizer_checkpoint_fails_transactionally():
    pytest.importorskip("ngclearn")
    jax = pytest.importorskip("jax")
    model = _model(optimizer="adam")
    samples = _samples()
    model.adapt(samples[:2], sample_rate_hz=250.0, epochs=1)
    live_before = model.snapshot_state()

    # Forge a self-consistent SGD-like scalar optimizer state. Its hashes all
    # agree with its contents, but it is structurally invalid for this Adam learner.
    forged_optimizer = np.asarray(1.0, dtype=np.float32)
    forged_optimizer_sha = _tree_sha256(jax, forged_optimizer)
    forged = NgcLearnHebbianState(
        weights=live_before.weights,
        optimizer_state=forged_optimizer,
        weights_sha256=live_before.weights_sha256,
        optimizer_sha256=forged_optimizer_sha,
        state_sha256=_state_sha256(live_before.weights_sha256, forged_optimizer_sha),
    )

    with pytest.raises(ValueError, match="optimizer pytree structure"):
        model.restore_state(forged)

    live_after = model.snapshot_state()
    assert live_after.state_sha256 == live_before.state_sha256
    assert np.array_equal(live_after.weights, live_before.weights)


def test_bad_samples_geometry_and_epochs_fail_closed():
    pytest.importorskip("ngclearn")
    model = _model()
    with pytest.raises(ValueError, match="2D time x channel"):
        model.adapt(np.ones(4), sample_rate_hz=250.0)
    with pytest.raises(ValueError, match="NaN or infinite"):
        model.adapt(np.asarray([[0.0, np.nan]]), sample_rate_hz=250.0)
    with pytest.raises(ValueError, match="epochs"):
        model.adapt(_samples(), sample_rate_hz=250.0, epochs=0)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        model.infer(_samples(), sample_rate_hz=0.0)
