from __future__ import annotations

import numpy as np
import pytest

import neuros.arena.runner as arena_runner
from neuros.arena import (
    ArenaScenario,
    ArtifactEvent,
    DeviceProfile,
    DisplayProfile,
    ParticipantProfile,
    StageSpec,
    TransportProfile,
    WorldModelEmission,
    WorldModelProfile,
    run_scenario,
)
from neuros.plugins import PluginKind, PluginRegistry


CHANNELS = np.asarray(["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"])


def scenario() -> ArenaScenario:
    return ArenaScenario(
        "world-model-contract",
        (
            StageSpec("rest", 1.0, None, 0.0),
            StageSpec("sight", 2.0, 10.0, 0.85),
        ),
        seed=23,
    )


def write_baseline(path, *, omit_oz: bool = False) -> None:
    fs = 250.0
    time = np.arange(int(12 * fs), dtype=float) / fs
    rng = np.random.default_rng(991)
    data = rng.normal(0.0, 4.0, size=(8, time.size))
    data += 2.0 * np.sin(2 * np.pi * 9.2 * time)[None, :]
    channels = CHANNELS.copy()
    if omit_oz:
        keep = channels != "Oz"
        data = data[keep]
        channels = channels[keep]
    np.savez(path, data_uv=data.astype(np.float32), sampling_rate_hz=np.asarray(fs), channel_names=channels)


def run_with_model(profile: WorldModelProfile):
    return run_scenario(
        scenario(),
        ParticipantProfile(seed=23, ssvep_amplitude_uv=5.5),
        DeviceProfile(chunk_samples=5, sensor_noise_uv=0.0, line_noise_uv=0.0),
        DisplayProfile(frame_drop_probability=0.03, frame_jitter_ms=0.5),
        TransportProfile(),
        profile,
    )


def test_world_model_plugins_are_discoverable_from_installed_arena():
    registry = PluginRegistry()
    registry.discover([PluginKind.WORLD_MODEL])
    names = {descriptor.name for descriptor in registry.list(PluginKind.WORLD_MODEL)}
    assert {"legacy_synthetic", "driven_state_space", "semi_synthetic_replay"} <= names


def test_semi_synthetic_replay_is_deterministic_and_display_coupled(tmp_path):
    baseline = tmp_path / "baseline.npz"
    write_baseline(baseline)
    profile = WorldModelProfile(
        "semi_synthetic_replay",
        {"path": str(baseline), "random_offset": False, "response_scale": 0.8},
    )
    first = run_with_model(profile)
    second = run_with_model(profile)
    np.testing.assert_array_equal(first.device_output.data_uv, second.device_output.data_uv)
    assert first.report["metrics"]["world_model"]["display_coupled"] is True
    assert first.report["metrics"]["world_model"]["stage_end_latent"][-1]["baseline_replay"] == 1.0


def test_semi_synthetic_replay_rejects_missing_required_montage_channel(tmp_path):
    baseline = tmp_path / "missing-oz.npz"
    write_baseline(baseline, omit_oz=True)
    with pytest.raises(ValueError, match="missing required Arena channels"):
        run_with_model(WorldModelProfile("semi_synthetic_replay", {"path": str(baseline)}))


def test_display_distortion_changes_causal_state_space_emission():
    base = dict(
        scenario=scenario(),
        participant=ParticipantProfile(seed=31, ssvep_amplitude_uv=6.0),
        device=DeviceProfile(chunk_samples=5, sensor_noise_uv=0.0, line_noise_uv=0.0),
        transport=TransportProfile(),
        world_model=WorldModelProfile("driven_state_space"),
    )
    clean = run_scenario(
        base["scenario"], base["participant"], base["device"],
        DisplayProfile(frame_drop_probability=0.0, frame_jitter_ms=0.0),
        base["transport"], base["world_model"],
    )
    broken = run_scenario(
        base["scenario"], base["participant"], base["device"],
        DisplayProfile(frame_drop_probability=0.45, frame_jitter_ms=2.0),
        base["transport"], base["world_model"],
    )
    assert not np.array_equal(clean.device_output.data_uv, broken.device_output.data_uv)
    clean_snr = clean.report["metrics"]["target_snr_db"]["10Hz"]
    broken_snr = broken.report["metrics"]["target_snr_db"]["10Hz"]
    assert np.isfinite(clean_snr) and np.isfinite(broken_snr)


def _artifact_world(events: tuple[ArtifactEvent, ...], *, device_chunk_samples: int):
    artifact_scenario = ArenaScenario(
        "artifact-order-contract",
        (
            StageSpec("rest", 0.5, None, 0.0),
            StageSpec("sight", 1.5, 10.0, 0.9, events),
        ),
        seed=43,
    )
    return run_scenario(
        artifact_scenario,
        ParticipantProfile(seed=43, ssvep_amplitude_uv=6.5),
        DeviceProfile(
            chunk_samples=device_chunk_samples,
            sensor_noise_uv=0.0,
            line_noise_uv=0.0,
            timestamp_jitter_ms=0.0,
        ),
        DisplayProfile(frame_drop_probability=0.0, frame_jitter_ms=0.0),
        TransportProfile(),
        WorldModelProfile("driven_state_space"),
    )


def _compiled_semantics(run) -> list[tuple[object, ...]]:
    schedule = run.report["metrics"]["world_model"]["compiled_artifact_schedule"]
    return [
        (
            item["event_id"],
            item["kind"],
            item["start_sample"],
            item["end_sample"],
            tuple(item["channel_indices"] or ()),
            item["seed"],
        )
        for item in schedule
    ]


def test_sample_indexed_artifacts_ignore_manifest_order_and_device_chunking():
    events = (
        ArtifactEvent(0.31, "controller", 0.42, 0.75),
        ArtifactEvent(0.36, "blink", 0.24, 0.55),
        ArtifactEvent(0.48, "dropout", 0.19, 1.0, channels=("PO7", "Oz")),
    )
    forward = _artifact_world(events, device_chunk_samples=5)
    reversed_world = _artifact_world(tuple(reversed(events)), device_chunk_samples=37)

    assert forward.report["metrics"]["world_model"]["artifact_execution"] == "sample_indexed"
    assert reversed_world.report["metrics"]["world_model"]["artifact_execution"] == "sample_indexed"
    np.testing.assert_array_equal(forward.device_output.data_uv, reversed_world.device_output.data_uv)
    np.testing.assert_array_equal(
        forward.device_output.ground_truth_timestamps_s,
        reversed_world.device_output.ground_truth_timestamps_s,
    )
    assert _compiled_semantics(forward) == _compiled_semantics(reversed_world)


def test_dropout_crosses_stage_boundary_with_exact_compiled_channel_support():
    artifact_scenario = ArenaScenario(
        "cross-stage-dropout",
        (
            StageSpec(
                "sight",
                1.0,
                10.0,
                0.85,
                (
                    ArtifactEvent(
                        0.92,
                        "dropout",
                        0.20,
                        1.0,
                        event_id="posterior-contact",
                        channels=("PO7", "Oz"),
                    ),
                ),
            ),
            StageSpec("rest", 0.5, None, 0.0),
        ),
        seed=47,
    )
    run = run_scenario(
        artifact_scenario,
        ParticipantProfile(seed=47),
        DeviceProfile(chunk_samples=37, sensor_noise_uv=0.0, line_noise_uv=0.0),
        DisplayProfile(frame_drop_probability=0.0, frame_jitter_ms=0.0),
        TransportProfile(),
        WorldModelProfile("driven_state_space"),
    )

    schedule = run.report["metrics"]["world_model"]["compiled_artifact_schedule"]
    assert len(schedule) == 1
    event = schedule[0]
    assert event["event_id"] == "posterior-contact"
    assert event["start_sample"] == 230
    assert event["end_sample"] == 280
    assert event["channel_indices"] == [5, 6]

    quantized_zero = run.device_output.lsb_uv / 2.0
    np.testing.assert_allclose(
        run.device_output.data_uv[[5, 6], 230:280],
        quantized_zero,
        atol=run.device_output.lsb_uv * 1e-4,
        rtol=0.0,
    )
    assert not np.allclose(run.device_output.data_uv[5, 220:230], quantized_zero)
    assert not np.allclose(run.device_output.data_uv[6, 280:290], quantized_zero)


def test_duplicate_explicit_artifact_ids_fail_before_rendering():
    duplicate = ArenaScenario(
        "duplicate-artifact-id",
        (
            StageSpec(
                "sight",
                1.0,
                10.0,
                0.8,
                (
                    ArtifactEvent(0.2, "blink", event_id="same-id"),
                    ArtifactEvent(0.4, "controller", event_id="same-id"),
                ),
            ),
        ),
        seed=53,
    )
    with pytest.raises(ValueError, match="event_id already scheduled"):
        run_scenario(
            duplicate,
            ParticipantProfile(seed=53),
            DeviceProfile(chunk_samples=5),
            DisplayProfile(),
            TransportProfile(),
            WorldModelProfile("driven_state_space"),
        )


def test_external_legacy_world_model_remains_runnable_and_is_labelled_weaker(monkeypatch):
    class LegacyPlugin:
        name = "external_legacy"
        channel_names = tuple(CHANNELS.tolist())

        def __init__(self) -> None:
            self.injected: list[tuple[str, float, float]] = []

        def inject_artifact(self, kind: str, duration_seconds: float, severity: float) -> None:
            self.injected.append((kind, duration_seconds, severity))

        def render(
            self,
            sample_times_s: np.ndarray,
            emitted_stimulus: np.ndarray,
            target_frequency_hz: float | None,
            attention_gain: float,
        ) -> WorldModelEmission:
            del emitted_stimulus, target_frequency_hz
            samples = int(np.asarray(sample_times_s).size)
            return WorldModelEmission(
                data_uv=np.zeros((8, samples), dtype=np.float32),
                latent={
                    "attention_gain": float(attention_gain),
                    "stimulus_coupling": 0.0,
                },
            )

    plugin = LegacyPlugin()
    monkeypatch.setattr(arena_runner, "load_plugin", lambda *args, **kwargs: plugin)
    legacy_scenario = ArenaScenario(
        "external-legacy-fallback",
        (
            StageSpec(
                "sight",
                0.5,
                10.0,
                0.8,
                (ArtifactEvent(0.13, "blink", 0.10, 0.5),),
            ),
        ),
        seed=59,
    )
    run = run_scenario(
        legacy_scenario,
        ParticipantProfile(seed=59),
        DeviceProfile(chunk_samples=37, sensor_noise_uv=0.0, line_noise_uv=0.0),
        DisplayProfile(frame_drop_probability=0.0, frame_jitter_ms=0.0),
        TransportProfile(),
        WorldModelProfile("external_legacy"),
    )

    world = run.report["metrics"]["world_model"]
    assert world["artifact_execution"] == "legacy_injection"
    assert world["compiled_artifact_schedule"] == []
    assert plugin.injected == [("blink", 0.10, 0.5)]
    assert run.report["world_model_evidence"]["evidence_level"] == "W?-external-unqualified"
