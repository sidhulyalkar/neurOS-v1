from __future__ import annotations

import numpy as np
import pytest

from neuros.arena import (
    ArenaScenario,
    DeviceProfile,
    DisplayProfile,
    ParticipantProfile,
    PRESENTATION_EPOCH_MODEL,
    StageSpec,
    TransportProfile,
    WorldModelProfile,
    compile_participant_state_trace,
    compile_presentation_plan,
    run_scenario,
)


def _participant() -> ParticipantProfile:
    return ParticipantProfile(
        seed=141,
        response_delay_s=0.04,
        switch_time_constant_s=0.20,
        gaze_duty_cycle=0.85,
    )


def _display() -> DisplayProfile:
    return DisplayProfile(
        response_lag_ms=6.0,
        frame_jitter_ms=0.9,
        frame_drop_probability=0.18,
    )


def _run(scenario: ArenaScenario):
    return run_scenario(
        scenario,
        _participant(),
        DeviceProfile(
            chunk_samples=31,
            sensor_noise_uv=0.0,
            line_noise_uv=0.0,
            timestamp_jitter_ms=0.0,
        ),
        _display(),
        TransportProfile(),
        WorldModelProfile("driven_state_space"),
    )


def test_label_only_stage_split_preserves_physical_presentation_and_eeg():
    whole = ArenaScenario(
        "presentation-whole",
        (
            StageSpec(
                "sight",
                2.0,
                10.0,
                0.9,
                stimulus_id="sight-orb",
            ),
        ),
        seed=151,
    )
    split = ArenaScenario(
        "presentation-split",
        (
            StageSpec(
                "sight-a",
                0.8,
                10.0,
                0.9,
                stimulus_id="sight-orb",
            ),
            StageSpec(
                "sight-b",
                1.2,
                10.0,
                0.9,
                stimulus_id="sight-orb",
            ),
        ),
        seed=151,
    )

    whole_run = _run(whole)
    split_run = _run(split)

    np.testing.assert_array_equal(whole_run.device_output.data_uv, split_run.device_output.data_uv)
    np.testing.assert_array_equal(
        whole_run.device_output.ground_truth_timestamps_s,
        split_run.device_output.ground_truth_timestamps_s,
    )
    whole_presentation = whole_run.report["metrics"]["presentation"]
    split_presentation = split_run.report["metrics"]["presentation"]
    assert whole_presentation["model"] == PRESENTATION_EPOCH_MODEL
    assert whole_presentation["epoch_count"] == 1
    assert split_presentation["epoch_count"] == 1
    assert split_presentation["stage_epoch_index"] == [0, 0]
    assert split_presentation["epochs"][0]["stages"] == ["sight-a", "sight-b"]
    assert split_run.report["metrics"]["participant_state"]["target_transition_samples"] == [0]


def test_response_lag_delays_emission_clock_without_advancing_code_phase():
    scenario = ArenaScenario(
        "display-lag-phase",
        (StageSpec("sight", 0.6, 10.0, 0.9, stimulus_id="sight"),),
        seed=153,
    )
    zero_lag = DisplayProfile(
        refresh_hz=120.0,
        response_lag_ms=0.0,
        frame_jitter_ms=0.0,
        frame_drop_probability=0.0,
    )
    delayed = DisplayProfile(
        refresh_hz=120.0,
        response_lag_ms=17.0,
        frame_jitter_ms=0.0,
        frame_drop_probability=0.0,
    )

    zero_trace = compile_presentation_plan(scenario, zero_lag, 250.0).epochs[0].trace
    delayed_trace = compile_presentation_plan(scenario, delayed, 250.0).epochs[0].trace

    # Response lag is command -> emission latency. It must not be fed back into
    # the code oscillator and thereby alter which luminance each command frame
    # requested. The emitted waveform is the same command sequence shifted later.
    np.testing.assert_array_equal(
        zero_trace.command_frame_times_s,
        delayed_trace.command_frame_times_s,
    )
    np.testing.assert_array_equal(zero_trace.luminance, delayed_trace.luminance)
    np.testing.assert_allclose(
        delayed_trace.frame_times_s - delayed_trace.command_frame_times_s,
        0.017,
        rtol=0.0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        delayed_trace.frame_times_s - zero_trace.frame_times_s,
        0.017,
        rtol=0.0,
        atol=1e-12,
    )
    assert delayed_trace.model == "neuros.arena.display_trace.v2"


def test_explicit_retrigger_creates_new_display_epoch_without_inventing_attention_switch():
    continuation = ArenaScenario(
        "continuation",
        (
            StageSpec("a", 0.5, 10.0, 0.9, stimulus_id="sight"),
            StageSpec("b", 0.5, 10.0, 0.9, stimulus_id="sight"),
        ),
        seed=157,
    )
    retrigger = ArenaScenario(
        "retrigger",
        (
            StageSpec("a", 0.5, 10.0, 0.9, stimulus_id="sight"),
            StageSpec(
                "b",
                0.5,
                10.0,
                0.9,
                stimulus_id="sight",
                stimulus_retrigger=True,
            ),
        ),
        seed=157,
    )

    continued = _run(continuation)
    restarted = _run(retrigger)

    assert continued.report["metrics"]["presentation"]["epoch_count"] == 1
    assert restarted.report["metrics"]["presentation"]["epoch_count"] == 2
    assert restarted.report["metrics"]["presentation"]["stage_epoch_index"] == [0, 1]
    assert restarted.report["metrics"]["participant_state"]["target_transition_samples"] == [0]
    assert not np.array_equal(continued.device_output.data_uv, restarted.device_output.data_uv)


def test_same_frequency_different_stimulus_is_real_target_switch():
    scenario = ArenaScenario(
        "same-frequency-different-object",
        (
            StageSpec("left", 0.5, 10.0, 1.0, stimulus_id="left-orb"),
            StageSpec("right", 0.5, 10.0, 1.0, stimulus_id="right-orb"),
        ),
        seed=163,
    )

    plan = compile_presentation_plan(scenario, _display(), 250.0)
    trace = compile_participant_state_trace(scenario, _participant(), 250.0)

    assert plan.stage_epoch_index == (0, 1)
    assert len(plan.epochs) == 2
    assert np.flatnonzero(trace.target_switch).tolist() == [0, 125]
    # 40 ms at 250 Hz is 10 samples. Changing physical target identity restarts
    # the declared participant response delay even when both objects use 10 Hz.
    assert np.all(trace.attention_gain[125:135] == 0.0)
    assert trace.attention_gain[135] > 0.0


def test_frequency_change_creates_epoch_without_explicit_retrigger():
    scenario = ArenaScenario(
        "frequency-switch",
        (
            StageSpec("sight", 0.5, 10.0, 0.9, stimulus_id="wisp"),
            StageSpec("guard", 0.5, 12.0, 0.9, stimulus_id="wisp"),
        ),
        seed=167,
    )
    plan = compile_presentation_plan(scenario, _display(), 250.0)
    assert plan.stage_epoch_index == (0, 1)
    assert [epoch.target_frequency_hz for epoch in plan.epochs] == [10.0, 12.0]


def test_stage_round_trip_preserves_presentation_controls():
    scenario = ArenaScenario(
        "presentation-round-trip",
        (
            StageSpec(
                "target",
                0.5,
                10.0,
                0.8,
                stimulus_id="sight-orb",
                stimulus_retrigger=True,
            ),
        ),
        seed=173,
    )
    loaded = ArenaScenario.from_dict(scenario.to_dict())
    assert loaded.stages[0].stimulus_id == "sight-orb"
    assert loaded.stages[0].stimulus_retrigger is True


def test_presentation_controls_fail_closed_on_ambiguous_types():
    with pytest.raises(ValueError, match="stimulus_id"):
        StageSpec("bad-id", 0.5, 10.0, stimulus_id="").validate()
    with pytest.raises(ValueError, match="stimulus_retrigger"):
        StageSpec("bad-reset", 0.5, 10.0, stimulus_retrigger=1).validate()
