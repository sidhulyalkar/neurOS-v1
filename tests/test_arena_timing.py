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


def test_non_grid_stage_durations_resolve_on_one_continuous_source_sample_clock():
    fs = 250.0
    scenario = ArenaScenario(
        "non-grid-stage-clock",
        (
            StageSpec("first", 0.75, None, 0.0),
            StageSpec("second", 0.50, 10.0, 0.8),
        ),
        seed=73,
    )
    run = run_scenario(
        scenario,
        ParticipantProfile(seed=73),
        DeviceProfile(
            sampling_rate_hz=fs,
            chunk_samples=37,
            sensor_noise_uv=0.0,
            line_noise_uv=0.0,
            timestamp_jitter_ms=0.0,
        ),
        DisplayProfile(frame_drop_probability=0.0, frame_jitter_ms=0.0),
        TransportProfile(),
    )

    # round(0.75 * 250) = 188 samples, so the first stage resolves to
    # 0.752 seconds. The second stage must start on sample 188 rather than the
    # requested floating boundary at 0.750 seconds.
    assert run.stages[0].start_s == 0.0
    assert np.isclose(run.stages[0].end_s, 188 / fs)
    assert np.isclose(run.stages[1].start_s, 188 / fs)
    assert np.isclose(run.stages[1].end_s, 313 / fs)
    assert np.count_nonzero(run.stage_index == 0) == 188
    assert np.count_nonzero(run.stage_index == 1) == 125

    timestamps = run.device_output.ground_truth_timestamps_s
    np.testing.assert_allclose(np.diff(timestamps), 1.0 / fs, rtol=0.0, atol=1e-12)
    assert np.isclose(timestamps[187], 187 / fs)
    assert np.isclose(timestamps[188], 188 / fs)

    timing = run.report["metrics"]["stage_timing"]
    assert timing[0]["start_sample"] == 0
    assert timing[0]["end_sample"] == 188
    assert timing[0]["requested_duration_s"] == 0.75
    assert np.isclose(timing[0]["resolved_duration_s"], 0.752)
    assert np.isclose(timing[0]["duration_error_ms"], 2.0)
    assert timing[1]["start_sample"] == 188
    assert timing[1]["end_sample"] == 313
    assert np.isclose(timing[1]["duration_error_ms"], 0.0)
