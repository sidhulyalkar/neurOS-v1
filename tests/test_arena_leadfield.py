from __future__ import annotations

import numpy as np

from neuros.arena import ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, StageSpec, TransportProfile, WorldModelProfile, run_scenario
from neuros.arena.leadfield import BUNDLE_SCHEMA, save_leadfield_bundle


CHANNELS = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")


def make_bundle(path) -> None:
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
        metadata={"test": "toy-forward"},
    )


def test_leadfield_bundle_round_trip_and_display_driven_projection(tmp_path):
    path = tmp_path / "leadfield.npz"
    make_bundle(path)
    with np.load(path, allow_pickle=False) as payload:
        assert str(np.asarray(payload["schema"]).reshape(-1)[0]) == BUNDLE_SCHEMA
        assert tuple(payload["channel_names"].tolist()) == CHANNELS

    scenario = ArenaScenario(
        "leadfield",
        (
            StageSpec("rest", 1.0, None, 0.0),
            StageSpec("sight", 2.0, 10.0, 0.85),
        ),
        seed=13,
    )
    args = (
        scenario,
        ParticipantProfile(seed=13, ssvep_amplitude_uv=6.0),
        DeviceProfile(chunk_samples=5, sensor_noise_uv=0.0, line_noise_uv=0.0),
    )
    profile = WorldModelProfile("leadfield_driven", {"path": str(path)})
    clean = run_scenario(*args, DisplayProfile(frame_drop_probability=0.0), TransportProfile(), profile)
    broken = run_scenario(*args, DisplayProfile(frame_drop_probability=0.40, frame_jitter_ms=1.5), TransportProfile(), profile)
    assert clean.report["metrics"]["world_model"]["display_coupled"] is True
    latent = clean.report["metrics"]["world_model"]["stage_end_latent"][-1]
    assert latent["leadfield_projection"] == 1.0
    assert not np.array_equal(clean.device_output.data_uv, broken.device_output.data_uv)


def test_leadfield_bundle_normalizes_visual_topography(tmp_path):
    path = tmp_path / "normalized.npz"
    save_leadfield_bundle(path, channel_names=CHANNELS, visual_topography=np.arange(1, 9, dtype=float))
    with np.load(path, allow_pickle=False) as payload:
        topography = np.asarray(payload["visual_topography"], dtype=float)
    assert np.isclose(np.max(np.abs(topography)), 1.0)
