from __future__ import annotations

import numpy as np
import pytest

import neuros.foundation_models.ngclearn_bridge as bridge
from neuros.foundation_models.ngclearn_bridge import (
    NgcLearnRateCellTransform,
    NgcLearnVersionError,
    ngclearn_runtime_identity,
)


def test_ngclearn_bridge_rejects_unqualified_minor_line(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(bridge, "_version", lambda: "3.3.0")
    with pytest.raises(NgcLearnVersionError, match="only the ngc-learn 3.2 line"):
        bridge._load_upstream()


def test_rate_cell_configuration_is_fail_closed_before_optional_import():
    with pytest.raises(ValueError, match="tau_m_ms"):
        NgcLearnRateCellTransform(tau_m_ms=0.0)
    with pytest.raises(ValueError, match="output"):
        NgcLearnRateCellTransform(output="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="seed"):
        NgcLearnRateCellTransform(seed=-1)


def test_real_ngclearn_32_surface_and_rate_cell_execution():
    pytest.importorskip("ngclearn")
    identity = ngclearn_runtime_identity()

    assert identity["integration"] == "ngc-learn"
    assert identity["qualified_series"] == "3.2.x"
    assert identity["ngclearn_version"].startswith("3.2.")
    assert len(identity["qualified_symbols"]) == 6

    samples = np.asarray(
        [
            [0.0, 0.25],
            [0.5, -0.25],
            [1.0, 0.0],
            [0.0, 0.5],
        ],
        dtype=np.float64,
    )
    transform = NgcLearnRateCellTransform(
        tau_m_ms=10.0,
        gamma=1.0,
        activation="identity",
        integration_type="euler",
        output="linear",
        seed=7,
    )
    first = transform.transform(samples, sample_rate_hz=1000.0)
    second = transform.transform(samples, sample_rate_hz=1000.0)

    assert first.values.shape == samples.shape
    assert np.isfinite(first.values).all()
    assert np.allclose(first.values, second.values, rtol=0.0, atol=0.0)
    assert first.evidence.input_shape == samples.shape
    assert first.evidence.output_shape == samples.shape
    assert first.evidence.input_sha256 == second.evidence.input_sha256
    assert first.evidence.output_sha256 == second.evidence.output_sha256
    assert first.evidence.evidence_sha256 == second.evidence.evidence_sha256
    boundary = first.evidence.to_dict()["claim_boundary"]
    assert boundary["upstream_package_executed"] is True
    assert boundary["rate_cell_contract_exercised"] is True
    assert boundary["predictive_coding_circuit_qualified"] is False
    assert boundary["spiking_network_qualified"] is False
    assert boundary["real_dataset_qualified"] is False
    assert boundary["hardware_qualified"] is False


def test_rate_cell_preserves_declared_geometry_and_rejects_bad_samples():
    pytest.importorskip("ngclearn")
    transform = NgcLearnRateCellTransform()

    with pytest.raises(ValueError, match="2D time x channel"):
        transform.transform(np.ones(8), sample_rate_hz=250.0)
    with pytest.raises(ValueError, match="NaN or infinite"):
        transform.transform(np.asarray([[0.0, np.nan]]), sample_rate_hz=250.0)
    with pytest.raises(ValueError, match="sample_rate_hz"):
        transform.transform(np.ones((2, 2)), sample_rate_hz=0.0)
