from __future__ import annotations

import numpy as np
import pytest

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
