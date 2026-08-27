from __future__ import annotations

import numpy as np

import neuros.arena.runner as arena_runner
from neuros.arena import (
    ArenaScenario,
    DeviceProfile,
    DisplayProfile,
    ParticipantProfile,
    StageSpec,
    TransportProfile,
    WorldModelProfile,
    run_scenario,
)
from neuros.arena.leadfield import save_leadfield_bundle
from neuros.arena.participant import compile_participant_state_trace


CHANNELS = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")


def _participant(**overrides) -> ParticipantProfile:
    values = dict(
        seed=91,
        response_delay_s=0.04,
        switch_time_constant_s=0.20,
        gaze_duty_cycle=0.80,
        response_attenuation_per_minute=0.05,
    )
    values.update(overrides)
    return ParticipantProfile(**values)


def _scenario() -> ArenaScenario:
    return ArenaScenario(
        "participant-chunk-probe",
        (
            StageSpec("rest", 0.25, None, 0.0),
            StageSpec("sight-a", 0.75, 10.0, 0.9),
            StageSpec("sight-b", 0.75, 10.0, 0.9),
            StageSpec("guard", 0.75, 12.0, 0.85),
        ),
        seed=97,
    )


def _device() -> DeviceProfile:
    return DeviceProfile(
        chunk_samples=31,
        sensor_noise_uv=0.0,
        line_noise_uv=0.0,
        timestamp_jitter_ms=0.0,
    )


def test_same_target_stage_segmentation_does_not_restart_participant_response():
    whole = ArenaScenario(
        "whole",
        (StageSpec("target", 2.0, 10.0, 0.9),),
        seed=91,
    )
    split = ArenaScenario(
        "split",
        (
            StageSpec("target-a", 0.75, 10.0, 0.9),
            StageSpec("target-b", 1.25, 10.0, 0.9),
        ),
        seed=91,
    )

    whole_trace = compile_participant_state_trace(whole, _participant(), 250.0)
    split_trace = compile_participant_state_trace(split, _participant(), 250.0)

    np.testing.assert_array_equal(whole_trace.attention_gain, split_trace.attention_gain)
    np.testing.assert_array_equal(
        whole_trace.requested_attention_gain,
        split_trace.requested_attention_gain,
    )
    np.testing.assert_array_equal(whole_trace.target_frequency_hz, split_trace.target_frequency_hz)
    assert np.flatnonzero(whole_trace.target_switch).tolist() == [0]
    assert np.flatnonzero(split_trace.target_switch).tolist() == [0]


def test_response_delay_restarts_on_target_identity_switch_not_stage_label():
    scenario = ArenaScenario(
        "switches",
        (
            StageSpec("sight-a", 0.5, 10.0, 1.0),
            StageSpec("sight-b", 0.5, 10.0, 1.0),
            StageSpec("guard", 0.5, 12.0, 1.0),
        ),
        seed=93,
    )
    trace = compile_participant_state_trace(scenario, _participant(), 250.0)

    # 40 ms at 250 Hz is exactly ten samples. A label-only boundary at sample
    # 125 does not restart it; the 10 -> 12 Hz identity switch at 250 does.
    assert np.flatnonzero(trace.target_switch).tolist() == [0, 250]
    assert np.all(trace.attention_gain[:10] == 0.0)
    assert trace.attention_gain[10] > 0.0
    assert trace.attention_gain[124] > 0.0
    assert trace.attention_gain[125] > 0.0
    assert np.all(trace.attention_gain[250:260] == 0.0)
    assert trace.attention_gain[260] > 0.0


def test_non_grid_response_delay_never_starts_early():
    scenario = ArenaScenario(
        "non-grid-delay",
        (StageSpec("target", 0.25, 10.0, 1.0),),
        seed=94,
    )
    trace = compile_participant_state_trace(
        scenario,
        _participant(response_delay_s=0.041),
        250.0,
    )

    # 41 ms lies between source samples 10 (40 ms) and 11 (44 ms). Causal delay
    # is a lower bound, so the first non-zero response is sample 11, never 10.
    assert np.all(trace.attention_gain[:11] == 0.0)
    assert trace.attention_gain[11] > 0.0


def test_target_transition_semantics_include_rest_boundaries_but_not_initial_rest():
    scenario = ArenaScenario(
        "rest-transitions",
        (
            StageSpec("rest-a", 0.20, None, 0.0),
            StageSpec("sight", 0.20, 10.0, 1.0),
            StageSpec("rest-b", 0.20, None, 0.0),
            StageSpec("guard", 0.20, 12.0, 1.0),
        ),
        seed=95,
    )
    trace = compile_participant_state_trace(scenario, _participant(), 250.0)
    summary = trace.to_summary()

    assert np.flatnonzero(trace.target_switch).tolist() == [50, 100, 150]
    assert summary["target_transition_samples"] == [50, 100, 150]
    assert summary["target_switch_samples"] == [50, 100, 150]


def _run_world(profile: WorldModelProfile):
    return run_scenario(
        _scenario(),
        _participant(),
        _device(),
        DisplayProfile(frame_drop_probability=0.0, frame_jitter_ms=0.0),
        TransportProfile(),
        profile,
    )


def _assert_partition_invariant(monkeypatch, profile: WorldModelProfile):
    monkeypatch.setattr(arena_runner, "WORLD_RENDER_CHUNK_SAMPLES", 1)
    fine = _run_world(profile)
    monkeypatch.setattr(arena_runner, "WORLD_RENDER_CHUNK_SAMPLES", 37)
    coarse = _run_world(profile)

    np.testing.assert_array_equal(fine.device_output.data_uv, coarse.device_output.data_uv)
    np.testing.assert_array_equal(
        fine.device_output.ground_truth_timestamps_s,
        coarse.device_output.ground_truth_timestamps_s,
    )
    assert fine.report["metrics"]["participant_state"] == coarse.report["metrics"]["participant_state"]
    assert fine.report["metrics"]["world_model"]["stage_end_latent"] == coarse.report["metrics"]["world_model"]["stage_end_latent"]
    assert fine.report["metrics"]["world_model"]["render_chunk_samples"] == 1
    assert coarse.report["metrics"]["world_model"]["render_chunk_samples"] == 37
    assert fine.report["metrics"]["world_model"]["stage_end_latent"][-1]["participant_stream_coupling"] == 1.0


def test_w1_default_world_is_invariant_to_internal_render_chunking(monkeypatch):
    _assert_partition_invariant(monkeypatch, WorldModelProfile("driven_state_space"))


def _write_baseline(path) -> None:
    fs = 250.0
    time = np.arange(int(8 * fs), dtype=float) / fs
    rng = np.random.default_rng(707)
    data = rng.normal(0.0, 3.5, size=(8, time.size))
    data += 1.8 * np.sin(2 * np.pi * 9.3 * time)[None, :]
    np.savez(
        path,
        data_uv=data.astype(np.float32),
        sampling_rate_hz=np.asarray(fs),
        channel_names=np.asarray(CHANNELS),
    )


def test_w2_semi_synthetic_world_is_invariant_to_internal_render_chunking(monkeypatch, tmp_path):
    baseline = tmp_path / "participant-background.npz"
    _write_baseline(baseline)
    _assert_partition_invariant(
        monkeypatch,
        WorldModelProfile(
            "semi_synthetic_replay",
            {"path": str(baseline), "random_offset": False, "response_scale": 0.8},
        ),
    )


def _write_leadfield(path) -> None:
    visual = np.asarray([0.05, 0.08, 0.10, 0.08, 0.42, 0.86, 1.0, 0.81])
    nuisance = np.asarray([
        [1.0, 0.3, 0.4, 0.3, 0.2, 0.1, 0.05, 0.1],
        [0.1, 0.8, 1.0, 0.8, 0.3, 0.1, 0.1, 0.1],
    ])
    save_leadfield_bundle(
        path,
        channel_names=CHANNELS,
        visual_topography=visual,
        nuisance_topographies=nuisance,
        metadata={"test": "participant-stream-partition"},
    )


def test_w3_leadfield_world_is_invariant_to_internal_render_chunking(monkeypatch, tmp_path):
    bundle = tmp_path / "participant-leadfield.npz"
    _write_leadfield(bundle)
    _assert_partition_invariant(
        monkeypatch,
        WorldModelProfile("leadfield_driven", {"path": str(bundle)}),
    )
