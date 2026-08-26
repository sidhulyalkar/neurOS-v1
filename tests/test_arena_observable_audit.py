from __future__ import annotations

import numpy as np

from neuros.arena.audit import eeg_observable_audit, flatten_audit_metrics


def test_observable_audit_separates_spectral_temporal_spatial_dimensions():
    fs = 250.0
    t = np.arange(1000, dtype=float) / fs
    rng = np.random.default_rng(11)
    source = 8.0 * np.sin(2 * np.pi * 10.0 * t)
    data = np.vstack([
        source + rng.normal(0, 1.0, size=t.size),
        0.8 * source + rng.normal(0, 1.2, size=t.size),
        rng.normal(0, 3.0, size=t.size),
    ])
    audit = eeg_observable_audit(data, fs)
    assert audit["schema"] == "neuros.synthetic_bci_arena.eeg_observable_audit.v1"
    assert 9.0 <= audit["spectrum"]["alpha_peak_hz"] <= 11.0
    assert audit["spectrum"]["alpha_8_13_fraction"] > audit["spectrum"]["gamma_30_45_fraction"]
    assert audit["spatial"]["mean_abs_channel_correlation"] > 0
    assert audit["spatial"]["covariance_effective_rank"] > 1.0
    assert 0 <= audit["spectrum"]["normalized_spectral_entropy"] <= 1.0
    assert "physiological equivalence" in audit["evidence_boundary"]
    flat = flatten_audit_metrics(audit)
    assert "amplitude.median_channel_rms_uv" in flat
    assert "temporal.median_autocorrelation_100ms" in flat


def test_observable_audit_rejects_nonfinite_input():
    data = np.zeros((2, 100), dtype=float)
    data[0, 4] = np.nan
    try:
        eeg_observable_audit(data, 250.0)
    except ValueError as exc:
        assert "finite" in str(exc)
    else:
        raise AssertionError("non-finite EEG should not receive an audit")
