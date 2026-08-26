from __future__ import annotations

import numpy as np

from neuros.arena import (
    ArenaScenario,
    DeviceProfile,
    DisplayProfile,
    ParticipantProfile,
    StageSpec,
    TransportProfile,
    run_scenario,
)


def make_timing_run(device: DeviceProfile, transport: TransportProfile):
    scenario = ArenaScenario(
        "timing",
        (
            StageSpec("rest", 1.0, None, 0.0),
            StageSpec("target", 2.0, 10.0, 0.8),
        ),
        seed=71,
    )
    return run_scenario(
        scenario,
        ParticipantProfile(seed=5),
        device,
        DisplayProfile(),
        transport,
    )


def test_ideal_clock_correction_recovers_causal_time_from_bad_source_clock():
    run = make_timing_run(
        DeviceProfile(
            clock_offset_ms=37.0,
            clock_drift_ppm=125.0,
            timestamp_jitter_ms=0.0,
            line_noise_uv=0.0,
        ),
        TransportProfile(),
    )
    metrics = run.report["metrics"]["transport"]
    assert abs(metrics["source_clock_offset_ms_estimated"] - 37.0) < 0.1
    assert abs(metrics["source_clock_drift_ppm_estimated"] - 125.0) < 0.1
    assert metrics["corrected_timestamp_rmse_ms"] < 1e-8
    packet = run.packets[len(run.packets) // 2]
    source_error = np.mean(np.abs(packet.source_timestamps_s - packet.ground_truth_timestamps_s))
    corrected_error = np.mean(np.abs(packet.timestamps_s - packet.ground_truth_timestamps_s))
    assert corrected_error < source_error


def test_residual_sync_error_is_reported_independently_from_delivery_latency():
    run = make_timing_run(
        DeviceProfile(
            clock_offset_ms=-18.0,
            clock_drift_ppm=75.0,
            timestamp_jitter_ms=0.15,
            line_noise_uv=0.0,
        ),
        TransportProfile(
            jitter_ms=4.0,
            clock_correction_offset_error_ms=2.5,
            clock_correction_drift_error_ppm=30.0,
            clock_correction_noise_ms=0.25,
        ),
    )
    metrics = run.report["metrics"]["transport"]
    assert metrics["delivery_delay_p95_ms"] > 0
    assert metrics["source_timestamp_jitter_rms_ms"] > 0
    assert metrics["corrected_timestamp_p95_abs_ms"] > 1.0
    assert metrics["corrected_timestamp_max_abs_ms"] >= metrics["corrected_timestamp_p95_abs_ms"]
    assert abs(metrics["source_clock_drift_ppm_estimated"] - 75.0) < 20.0
