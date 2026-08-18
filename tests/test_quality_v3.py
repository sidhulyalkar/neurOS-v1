import json

import numpy as np

from neuros.contracts import ClockDomain, QualityFlag, SignalFrame
from neuros.quality import (
    BenchmarkManifest,
    FaultProfile,
    QualityThresholds,
    evaluate_runtime_snapshot,
    frequency_selectivity_probe,
    perturb_frame,
)


def _snapshot(*, dropped=0, p99=2.0, processed=10):
    return {
        "state": "stopped",
        "failure": None,
        "runtime_seconds": 0.1,
        "nodes": {
            "transform:eeg:0": {
                "processed": processed,
                "failed": 0,
                "p99_latency_ms": 1.0,
            },
            "decoder:primary": {
                "processed": processed,
                "failed": 0,
                "p99_latency_ms": p99,
            },
        },
        "edges": {
            "source:eeg->transform:eeg:0": {"accepted": processed, "dropped": dropped},
            "transform:eeg:0->decoder:primary": {"accepted": processed, "dropped": 0},
        },
    }


def test_quality_gate_passes_clean_snapshot_and_rejects_loss():
    thresholds = QualityThresholds(
        min_decoder_samples=5,
        max_drop_fraction=0.0,
        max_decoder_p99_ms=10.0,
        max_transform_p99_ms=10.0,
    )
    assert evaluate_runtime_snapshot(_snapshot(), thresholds).passed
    failed = evaluate_runtime_snapshot(_snapshot(dropped=1), thresholds)
    assert not failed.passed
    assert "edge_loss" in failed.failures


def test_fault_injection_is_deterministic_and_marks_quality():
    frame = SignalFrame(
        stream_id="eeg",
        sequence_id=3,
        data=np.ones((4, 8), dtype=np.float32),
        sample_rate_hz=250.0,
        host_receive_time_ns=100,
        device_time_ns=1_000_000_000,
        synchronized_time_ns=1_000_000_000,
        clock_domain=ClockDomain.SYNCHRONIZED,
    )
    profile = FaultProfile(
        seed=7,
        timestamp_jitter_std_ms=1.0,
        channel_dropout_probability=0.5,
        additive_noise_std=0.01,
        clock_drift_ppm=100.0,
    )
    a = perturb_frame(frame, profile, np.random.default_rng(7), origin_device_time_ns=0)
    b = perturb_frame(frame, profile, np.random.default_rng(7), origin_device_time_ns=0)
    np.testing.assert_array_equal(a.data, b.data)
    assert a.device_time_ns == b.device_time_ns
    assert a.synchronized_time_ns == b.synchronized_time_ns
    assert a.quality & QualityFlag.CLOCK_UNCERTAIN
    assert a.quality & QualityFlag.DISCONNECTED_CHANNEL


def test_known_frequency_probes_recover_expected_eeg_bands():
    for frequency, band in [(6.0, "theta"), (10.0, "alpha"), (20.0, "beta"), (40.0, "gamma")]:
        result = frequency_selectivity_probe(frequency, seed=42)
        assert result.expected_band == band
        assert result.winning_band == band
        assert result.selectivity_ratio > 3.0


def test_benchmark_manifest_is_json_serializable_and_hashed():
    manifest = BenchmarkManifest.capture(
        "unit-test",
        config={"a": 1},
        data_fingerprint={"dataset": "synthetic", "n": 10},
        seed=4,
        artifact_ids=("decoder:v1",),
    )
    payload = manifest.to_dict()
    json.dumps(payload)
    assert payload["config_hash"]
    assert payload["data_hash"]
    assert payload["seed"] == 4
