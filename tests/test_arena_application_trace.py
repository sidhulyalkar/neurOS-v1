from __future__ import annotations

from neuros.arena import ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, StageSpec, TransportProfile, run_scenario
from neuros.arena.application import (
    ApplicationEvent,
    ApplicationTrace,
    evaluate_application_trace,
    load_application_trace,
    save_application_trace,
)


def make_run():
    return run_scenario(
        ArenaScenario(
            "application-trace",
            (
                StageSpec("rest", 1.0, None, 0.0),
                StageSpec("active", 3.0, 10.0, 0.9),
            ),
            seed=61,
        ),
        ParticipantProfile(seed=3),
        DeviceProfile(line_noise_uv=0.0),
        DisplayProfile(),
        TransportProfile(silence_windows=((2.0, 0.5),)),
    )


def test_application_trace_round_trip_and_causal_scoring(tmp_path):
    trace = ApplicationTrace(
        application="example-game",
        version="0.1.0",
        events=(
            ApplicationEvent(0.50, "neural_action", action="bad-rest-action", source="neural", authority=0.8, source_sequence=1),
            ApplicationEvent(1.50, "neural_action", action="valid-action", source="neural", authority=0.9, source_sequence=2),
            ApplicationEvent(2.20, "bci_lost", source="system"),
            ApplicationEvent(2.25, "neural_action", action="bad-silence-action", source="neural", authority=0.7, source_sequence=3),
            ApplicationEvent(2.80, "bci_recovered", source="system"),
            ApplicationEvent(3.00, "participant_stop", source="participant"),
            ApplicationEvent(3.20, "neural_action", action="bad-post-stop-action", source="neural", authority=0.6, source_sequence=4),
        ),
        metadata={"engine": "test"},
    )
    path = save_application_trace(trace, tmp_path / "trace.json")
    loaded = load_application_trace(path)
    assert loaded == trace
    metrics = evaluate_application_trace(make_run(), loaded)
    assert metrics["neural_actions_total"] == 4.0
    assert metrics["neural_actions_without_target"] == 1.0
    assert metrics["neural_actions_during_transport_silence"] == 1.0
    assert metrics["participant_stop_action_violations"] == 1.0
    assert metrics["neural_action_sequence_regressions"] == 0.0
    assert metrics["recovery_latency_mean_s"] > 0


def test_application_trace_rejects_time_regression():
    trace = ApplicationTrace(
        application="bad-game",
        version="0.1",
        events=(
            ApplicationEvent(1.0, "application_state"),
            ApplicationEvent(0.9, "application_state"),
        ),
    )
    try:
        trace.validate()
    except ValueError as exc:
        assert "monotonic" in str(exc)
    else:
        raise AssertionError("non-monotonic application events should fail")
