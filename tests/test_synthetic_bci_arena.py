from __future__ import annotations

import json

import numpy as np

from neuros.arena import (
    ArenaDecision,
    ArenaManifest,
    ArenaScenario,
    DeviceProfile,
    DisplayProfile,
    ParticipantProfile,
    StageSpec,
    TransportProfile,
    WorldModelProfile,
    compare_feature_signatures,
    evaluate_decisions,
    feature_signature,
    load_manifest,
    run_scenario,
    save_manifest,
)


def tiny_world(
    *,
    participant: ParticipantProfile | None = None,
    display: DisplayProfile | None = None,
    transport: TransportProfile | None = None,
    world_model: WorldModelProfile | None = None,
):
    scenario = ArenaScenario("tiny", (
        StageSpec("rest", 1.5, None, 0.0),
        StageSpec("sight", 2.0, 10.0, 1.0),
        StageSpec("guard", 2.0, 12.0, 1.0),
    ), seed=11)
    return run_scenario(
        scenario,
        participant or ParticipantProfile(seed=11),
        DeviceProfile(chunk_samples=5),
        display or DisplayProfile(),
        transport or TransportProfile(),
        world_model,
    )


def test_arena_is_deterministic_for_same_manifest():
    first = tiny_world()
    second = tiny_world()
    np.testing.assert_array_equal(first.device_output.data_uv, second.device_output.data_uv)
    np.testing.assert_array_equal(first.device_output.timestamps_s, second.device_output.timestamps_s)
    assert first.report == second.report


def test_default_world_model_is_driven_by_emitted_display():
    clean = tiny_world(display=DisplayProfile(frame_drop_probability=0.0))
    distorted = tiny_world(display=DisplayProfile(frame_drop_probability=0.35, frame_jitter_ms=1.5))
    assert clean.world_model.name == "driven_state_space"
    assert clean.report["metrics"]["world_model"]["display_coupled"] is True
    assert distorted.report["metrics"]["world_model"]["display_coupled"] is True
    assert any(stage["frame_drop_fraction"] > 0 for stage in distorted.report["metrics"]["display"])
    assert not np.array_equal(clean.device_output.data_uv, distorted.device_output.data_uv)


def test_legacy_world_model_remains_available_as_explicit_regression_adapter():
    run = tiny_world(world_model=WorldModelProfile("legacy_synthetic"))
    assert run.world_model.name == "legacy_synthetic"
    assert run.report["metrics"]["world_model"]["display_coupled"] is False


def test_weak_responder_has_lower_target_snr():
    strong = tiny_world(participant=ParticipantProfile(seed=11, ssvep_amplitude_uv=9.0))
    weak = tiny_world(participant=ParticipantProfile(seed=11, ssvep_amplitude_uv=2.0, colored_noise_uv=7.0))
    assert strong.report["metrics"]["target_snr_db"]["10Hz"] > weak.report["metrics"]["target_snr_db"]["10Hz"]


def test_display_and_transport_faults_are_observable():
    run = tiny_world(
        display=DisplayProfile(frame_drop_probability=0.08, frame_jitter_ms=1.0),
        transport=TransportProfile(drop_probability=0.10, jitter_ms=12.0, silence_windows=((2.0, 0.6),)),
    )
    display = run.report["metrics"]["display"]
    transport = run.report["metrics"]["transport"]
    assert any(stage["frame_drop_fraction"] > 0 for stage in display)
    assert transport["packet_drop_fraction"] > 0
    assert transport["delivery_delay_p95_ms"] > 0


def test_external_decoder_decisions_score_against_ground_truth():
    run = tiny_world()
    decisions = [
        ArenaDecision(0.5, 10.0, True),  # false activation during rest
        ArenaDecision(1.9, 10.0, True),
        ArenaDecision(3.9, 12.0, True),
        ArenaDecision(4.6, None, False),
    ]
    metrics = evaluate_decisions(run, decisions)
    assert metrics["accepted_precision"] < 1.0
    assert metrics["false_activation_fraction"] > 0
    assert metrics["median_switch_latency_s"] >= 0


def test_feature_signature_comparison_is_identity_for_same_data():
    run = tiny_world()
    signature = feature_signature(run.device_output.data_uv, run.device.sampling_rate_hz)
    comparison = compare_feature_signatures(signature, signature)
    assert comparison["mean_abs_log_ratio"] == 0.0
    assert comparison["max_abs_log_ratio"] == 0.0


def test_manifest_round_trip(tmp_path):
    run = tiny_world(world_model=WorldModelProfile("driven_state_space", {"resonance_damping": 0.31}))
    manifest = ArenaManifest(
        run.scenario,
        run.participant,
        run.device,
        run.display,
        run.transport,
        run.world_model,
    )
    path = tmp_path / "world.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)
    assert loaded == manifest
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert raw["schema"] == "neuros.synthetic_bci_arena.manifest.v1"
    assert raw["world_model"]["name"] == "driven_state_space"
    assert raw["world_model"]["parameters"]["resonance_damping"] == 0.31
