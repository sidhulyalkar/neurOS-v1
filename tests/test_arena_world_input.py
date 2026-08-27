from __future__ import annotations

import numpy as np
import pytest

from neuros.arena import (
    ArenaManifest,
    ArenaScenario,
    DeviceProfile,
    DisplayProfile,
    ParticipantProfile,
    StageSpec,
    TransportProfile,
    WorldInputBlock,
    WorldModelEmission,
    WorldModelProfile,
    run_scenario,
)
from neuros.arena.manifest import manifest_from_dict
from neuros.plugins import PluginKind, registry


class RichProbeWorldModel:
    channel_names = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
    seen: list[dict] = []

    def __init__(self, *, participant, sampling_rate_hz, seed, parameters):
        self.sampling_rate_hz = float(sampling_rate_hz)

    def inject_artifact(self, kind, duration_seconds, severity):
        return None

    def render_world(self, block):
        block.validate()
        RichProbeWorldModel.seen.append({
            "paradigm": block.paradigm,
            "stage": block.stage_label,
            "target": dict(block.target),
            "task_state": dict(block.task_state),
            "attention_gain": block.participant_state.get("attention_gain"),
            "attention_stream": block.attention_gain.copy(),
            "visual_shape": block.visual_luminance.shape,
        })
        amplitude = 2.0 if block.target.get("oddball") is True else 0.0
        data = np.full((len(self.channel_names), block.sample_times_s.size), amplitude, dtype=np.float32)
        return WorldModelEmission(data_uv=data, latent={"rich_path": 1.0})

    def evidence_card(self):
        return {
            "model_name": "rich_probe_test",
            "evidence_level": "W?-test-only",
            "model_family": "test probe",
            "stimulus_causal": True,
            "spatial_model": "none",
            "recorded_human_background": False,
            "known_intervention_ground_truth": True,
            "artifact_ground_truth": True,
            "uncertainty_representation": "none",
            "validated_against": ("test-only WorldInput contract",),
            "intended_uses": ("unit test",),
            "unsupported_claims": ("all physiological claims",),
            "notes": (),
        }


def _register_probe():
    registry.register(
        name="rich_probe_test",
        kind=PluginKind.WORLD_MODEL,
        factory=RichProbeWorldModel,
        replace=True,
    )


def test_runner_prefers_paradigm_neutral_render_world_contract():
    _register_probe()
    RichProbeWorldModel.seen.clear()
    scenario = ArenaScenario(
        "p300-probe",
        (
            StageSpec(
                "standard",
                0.5,
                None,
                0.0,
                target={"oddball": False, "symbol": "A"},
                task_state={"trial_type": "standard"},
            ),
            StageSpec(
                "oddball",
                0.5,
                None,
                1.0,
                target={"oddball": True, "symbol": "B"},
                task_state={"trial_type": "target"},
            ),
        ),
        seed=81,
        metadata={"paradigm": "p300"},
    )
    run = run_scenario(
        scenario,
        ParticipantProfile(seed=9),
        DeviceProfile(line_noise_uv=0.0, sensor_noise_uv=0.0),
        DisplayProfile(),
        TransportProfile(),
        WorldModelProfile("rich_probe_test"),
    )
    assert run.report["paradigm"] == "p300"
    assert run.report["world_model_evidence"]["evidence_level"] == "W?-test-only"
    assert any(item["target"].get("oddball") is True for item in RichProbeWorldModel.seen)
    assert all(item["paradigm"] == "p300" for item in RichProbeWorldModel.seen)
    assert all(item["visual_shape"] for item in RichProbeWorldModel.seen)
    # Arena's built-in participant_response.v1 is explicitly frequency-target
    # scoped. A P300 plugin receives the rich transport surface but does not
    # inherit made-up SSVEP attention dynamics.
    assert all(np.all(item["attention_stream"] == 0.0) for item in RichProbeWorldModel.seen)
    assert float(np.max(run.device_output.data_uv)) > 1.5


def test_world_input_rejects_bad_participant_stream_geometry_and_nonfinite_values():
    times = np.arange(5, dtype=float) / 250.0
    base = dict(
        sample_times_s=times,
        paradigm="ssvep",
        stage_label="target",
        emitted_streams={"visual_luminance": np.zeros(5)},
        target={"frequency_hz": 10.0},
    )
    with pytest.raises(ValueError, match="participant_streams.attention_gain"):
        WorldInputBlock(
            **base,
            participant_streams={"attention_gain": np.zeros(4)},
        ).validate()
    bad = np.zeros(5)
    bad[2] = np.nan
    with pytest.raises(ValueError, match="participant_streams.attention_gain"):
        WorldInputBlock(
            **base,
            participant_streams={"attention_gain": bad},
        ).validate()


def test_attention_gain_property_prefers_stream_and_falls_back_to_scalar():
    times = np.arange(4, dtype=float) / 250.0
    streamed = WorldInputBlock(
        sample_times_s=times,
        paradigm="ssvep",
        stage_label="target",
        emitted_streams={},
        participant_state={"attention_gain": 0.2},
        participant_streams={"attention_gain": np.asarray([0.0, 0.1, 0.3, 0.6])},
    )
    streamed.validate()
    np.testing.assert_array_equal(streamed.attention_gain, [0.0, 0.1, 0.3, 0.6])

    scalar = WorldInputBlock(
        sample_times_s=times,
        paradigm="ssvep",
        stage_label="target",
        emitted_streams={},
        participant_state={"attention_gain": 0.25},
    )
    scalar.validate()
    np.testing.assert_array_equal(scalar.attention_gain, np.full(4, 0.25))


def test_manifest_v2_round_trip_preserves_rich_stage_metadata():
    manifest = ArenaManifest(
        ArenaScenario(
            "rich-manifest",
            (
                StageSpec(
                    "cue",
                    1.0,
                    None,
                    0.0,
                    target={"class": "left"},
                    task_state={"instruction": "imagine"},
                ),
            ),
            metadata={"paradigm": "motor_imagery"},
        ),
        ParticipantProfile(),
        DeviceProfile(),
        DisplayProfile(),
        TransportProfile(),
        WorldModelProfile("rich_probe_test"),
    )
    loaded = manifest_from_dict(manifest.to_dict())
    assert loaded.scenario.metadata["paradigm"] == "motor_imagery"
    assert loaded.scenario.stages[0].target == {"class": "left"}
    assert loaded.scenario.stages[0].task_state == {"instruction": "imagine"}
