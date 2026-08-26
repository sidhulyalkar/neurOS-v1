from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.ngclearn_predictive_coding import (
    PC_METHOD_ID,
    NgcLearnPredictiveCodingTransform,
)


def test_predictive_coding_configuration_fails_before_optional_import():
    with pytest.raises(ValueError, match="latent_dim"):
        NgcLearnPredictiveCodingTransform(latent_dim=0)
    with pytest.raises(ValueError, match="settling_steps"):
        NgcLearnPredictiveCodingTransform(settling_steps=0)
    with pytest.raises(ValueError, match="settling_dt_ms"):
        NgcLearnPredictiveCodingTransform(settling_dt_ms=0.0)
    with pytest.raises(ValueError, match="prior_gamma"):
        NgcLearnPredictiveCodingTransform(prior_gamma=-0.1)
    with pytest.raises(ValueError, match="weights first dimension"):
        NgcLearnPredictiveCodingTransform(latent_dim=2, weights=np.ones((3, 4)))


def test_real_ngclearn_predictive_coding_reduces_identity_reconstruction_error():
    pytest.importorskip("ngclearn")
    samples = np.asarray(
        [
            [0.50, -0.25],
            [1.00, 0.50],
            [-0.75, 0.25],
            [0.20, 0.80],
        ],
        dtype=np.float64,
    )
    transform = NgcLearnPredictiveCodingTransform(
        latent_dim=2,
        settling_steps=80,
        settling_dt_ms=1.0,
        tau_m_ms=20.0,
        prior_gamma=0.0,
        activation="identity",
        integration_type="euler",
        output="linear",
        weights=np.eye(2, dtype=np.float64),
        seed=11,
    )

    first = transform.transform(samples, sample_rate_hz=250.0)
    second = transform.transform(samples, sample_rate_hz=250.0)

    assert first.values.shape == samples.shape
    assert first.reconstruction.shape == samples.shape
    assert first.mean_squared_error_by_step.shape == (81,)
    assert np.isfinite(first.values).all()
    assert np.isfinite(first.reconstruction).all()
    assert np.isfinite(first.mean_squared_error_by_step).all()

    # This known-ground-truth identity dictionary establishes that the real
    # upstream error-feedback dynamics actually correct a prediction rather
    # than merely executing a graph with the right class names.
    assert first.evidence.initial_mse > 0.0
    assert first.evidence.final_mse < first.evidence.initial_mse
    assert first.evidence.error_reduction_fraction is not None
    assert first.evidence.error_reduction_fraction > 0.90
    assert first.evidence.samples_improved_fraction == 1.0
    assert first.mean_squared_error_by_step[-1] < first.mean_squared_error_by_step[1]

    # Reset-per-sample and fixed weights make repeated execution exact on the
    # same installed upstream/runtime surface.
    assert np.allclose(first.values, second.values, rtol=0.0, atol=0.0)
    assert np.allclose(first.reconstruction, second.reconstruction, rtol=0.0, atol=0.0)
    assert np.allclose(
        first.mean_squared_error_by_step,
        second.mean_squared_error_by_step,
        rtol=0.0,
        atol=0.0,
    )
    assert first.evidence.evidence_sha256 == second.evidence.evidence_sha256
    assert first.evidence.method_id == PC_METHOD_ID

    boundary = first.evidence.to_dict()["claim_boundary"]
    assert boundary["upstream_package_executed"] is True
    assert boundary["predictive_coding_circuit_qualified"] is True
    assert boundary["iterative_error_feedback_exercised"] is True
    assert boundary["fixed_weight_inference_only"] is True
    assert boundary["hebbian_learning_qualified"] is False
    assert boundary["stdp_learning_qualified"] is False
    assert boundary["online_learning_qualified"] is False
    assert boundary["real_dataset_qualified"] is False
    assert boundary["hardware_qualified"] is False
    assert boundary["closed_loop_qualified"] is False
    assert boundary["clinical_qualified"] is False


def test_predictive_coding_hashes_and_geometry_are_explicit():
    pytest.importorskip("ngclearn")
    weights = np.asarray(
        [
            [0.7, 0.0, 0.2],
            [0.0, 0.6, -0.1],
        ],
        dtype=np.float64,
    )
    samples = np.asarray([[0.2, 0.4, -0.1], [0.1, -0.3, 0.5]], dtype=np.float64)
    transform = NgcLearnPredictiveCodingTransform(
        latent_dim=2,
        settling_steps=10,
        weights=weights,
        seed=3,
    )
    result = transform.transform(samples, sample_rate_hz=500.0)

    assert result.evidence.input_shape == (2, 3)
    assert result.evidence.latent_shape == (2, 2)
    assert result.evidence.reconstruction_shape == (2, 3)
    assert len(result.evidence.input_sha256) == 64
    assert len(result.evidence.weights_sha256) == 64
    assert len(result.evidence.latent_sha256) == 64
    assert len(result.evidence.reconstruction_sha256) == 64
    assert len(result.evidence.error_trajectory_sha256) == 64
    assert len(result.evidence.evidence_sha256) == 64
    assert result.evidence.tied_transpose_feedback is True
    assert result.evidence.reset_per_sample is True
    assert result.evidence.learning_enabled is False


def test_predictive_coding_rejects_input_or_weight_geometry_drift():
    pytest.importorskip("ngclearn")
    transform = NgcLearnPredictiveCodingTransform(
        latent_dim=2,
        weights=np.eye(2, dtype=np.float64),
    )
    transform.transform(np.ones((2, 2)), sample_rate_hz=250.0)

    with pytest.raises(ValueError, match="input channel count is fixed"):
        transform.transform(np.ones((2, 3)), sample_rate_hz=250.0)

    mismatched = NgcLearnPredictiveCodingTransform(
        latent_dim=2,
        weights=np.ones((2, 3), dtype=np.float64),
    )
    with pytest.raises(ValueError, match="weights second dimension"):
        mismatched.transform(np.ones((2, 2)), sample_rate_hz=250.0)
