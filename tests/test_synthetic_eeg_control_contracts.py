from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from neuros.drivers.synthetic_eeg import SyntheticEEGConfig, SyntheticEEGGenerator


ROOT = Path(__file__).resolve().parents[1]
PHANTOM_PATH = ROOT / "examples" / "mindforge_phantom_unicorn.py"


def _load_phantom_module():
    spec = importlib.util.spec_from_file_location("mindforge_phantom_unicorn", PHANTOM_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("sampling_rate_hz", np.nan),
        ("colored_noise_uv", np.inf),
        ("white_noise_uv", np.nan),
        ("alpha_frequency_hz", np.inf),
        ("alpha_amplitude_uv", np.nan),
        ("ssvep_amplitude_uv", np.inf),
        ("first_harmonic_ratio", np.nan),
    ),
)
def test_generator_configuration_rejects_nonfinite_values(field: str, value: float):
    with pytest.raises(ValueError, match="finite"):
        SyntheticEEGGenerator(SyntheticEEGConfig(**{field: value}))


@pytest.mark.parametrize("seed", (-1, 7.5, True))
def test_generator_configuration_requires_nonnegative_integer_seed(seed):
    with pytest.raises(ValueError, match="seed must be a non-negative integer"):
        SyntheticEEGGenerator(SyntheticEEGConfig(seed=seed))


def test_generator_configuration_rejects_ambiguous_channel_names():
    with pytest.raises(ValueError, match="unique"):
        SyntheticEEGGenerator(
            SyntheticEEGConfig(
                channel_names=("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "Oz")
            )
        )


def test_runtime_controls_reject_nonfinite_values_before_state_mutates():
    generator = SyntheticEEGGenerator(SyntheticEEGConfig(seed=131))

    with pytest.raises(ValueError, match="positive and finite"):
        generator.set_attention(np.nan)
    assert generator.target_frequency_hz is None
    assert generator.attention_gain == 0.0

    generator.set_attention(10.0, 0.7)
    with pytest.raises(ValueError, match="gain must be finite"):
        generator.set_attention(12.0, np.nan)
    assert generator.target_frequency_hz == 10.0
    assert generator.attention_gain == 0.7

    before = generator.channel_gain.copy()
    with pytest.raises(ValueError, match="gain must be finite"):
        generator.set_channel_gain("Oz", np.inf)
    np.testing.assert_array_equal(generator.channel_gain, before)


def test_sample_exact_apis_reject_lossy_numeric_coercion():
    generator = SyntheticEEGGenerator(SyntheticEEGConfig(seed=137))

    with pytest.raises(ValueError, match="start_sample must be an integer"):
        generator.schedule_artifact(
            "blink",
            event_id="fractional-start",
            start_sample=10.7,
            duration_seconds=0.1,
        )
    with pytest.raises(ValueError, match="seed must be an integer"):
        generator.schedule_artifact(
            "controller",
            event_id="fractional-seed",
            start_sample=10,
            duration_seconds=0.1,
            seed=3.2,
        )
    with pytest.raises(ValueError, match="channel index must be an integer"):
        generator.set_channel_gain(6.2, 1.0)
    with pytest.raises(ValueError, match="samples must be an integer"):
        generator.render(10.5)

    event = generator.schedule_artifact(
        "controller",
        event_id="numpy-integer-controls",
        start_sample=np.int64(10),
        duration_seconds=0.1,
        seed=np.int64(17),
        channels=np.int64(6),
    )
    assert event.start_sample == 10
    assert event.seed == 17
    assert event.channel_indices == (6,)


def test_phantom_parser_accepts_explicit_future_schedule_and_channel_list():
    phantom = _load_phantom_module()
    parsed = phantom.parse_artifact_schedule_command(
        "artifact:posterior-loss:dropout:40:0.10:1.0:PO7,Oz:1234"
    )
    assert parsed == {
        "kind": "dropout",
        "event_id": "posterior-loss",
        "start_sample": 40,
        "duration_seconds": 0.10,
        "severity": 1.0,
        "channels": ("PO7", "Oz"),
        "seed": 1234,
    }


@pytest.mark.parametrize(
    "command",
    (
        "artifact::blink:10:0.1:1.0",
        "artifact:a:not-real:10:0.1:1.0",
        "artifact:a:blink:10.5:0.1:1.0",
        "artifact:a:blink:10:nan:1.0",
        "artifact:a:blink:10:0.1:inf",
        "artifact:a:blink:10:0.1:1.0:*:-1",
    ),
)
def test_phantom_parser_rejects_ambiguous_or_nonfinite_schedule(command: str):
    phantom = _load_phantom_module()
    with pytest.raises(ValueError):
        phantom.parse_artifact_schedule_command(command)


def test_phantom_schedule_resolves_lowercase_channel_labels_from_udp_normalization():
    phantom = _load_phantom_module()
    generator = SyntheticEEGGenerator(SyntheticEEGConfig(seed=139))

    event = phantom.schedule_artifact_command(
        generator,
        "artifact:posterior-loss:dropout:40:0.10:1.0:po7,oz:1234",
    )
    assert event.channel_indices == (5, 6)
    assert event.seed == 1234


def test_phantom_control_surface_can_build_a_true_overlapping_world():
    phantom = _load_phantom_module()
    cfg = SyntheticEEGConfig(seed=149)
    generator = SyntheticEEGGenerator(cfg)

    blink = phantom.schedule_artifact_command(
        generator,
        "artifact:blink-a:blink:20:0.30:0.7:*:101",
    )
    controller = phantom.schedule_artifact_command(
        generator,
        "artifact:controller-a:controller:10:0.40:0.9:*:202",
    )
    dropout = phantom.schedule_artifact_command(
        generator,
        "artifact:dropout-a:dropout:40:0.10:1.0:PO7,Oz:303",
    )

    assert tuple(event.event_id for event in generator.scheduled_artifacts) == (
        "controller-a",
        "blink-a",
        "dropout-a",
    )
    block = generator.render(150)
    assert set(block.artifact_ids) == {blink.event_id, controller.event_id, dropout.event_id}
    assert block.artifact == "multiple"
    assert np.allclose(block.data_uv[5:7, 40:65], 0.0)


def test_phantom_cancel_keeps_scenario_state_explicit():
    phantom = _load_phantom_module()
    generator = SyntheticEEGGenerator(SyntheticEEGConfig(seed=151))
    event = phantom.schedule_artifact_command(
        generator,
        "artifact:future-blink:blink:100:0.2:1.0:*:44",
    )
    assert generator.cancel_artifact(event.event_id) is True
    assert generator.scheduled_artifacts == ()
    assert generator.cancel_artifact(event.event_id) is False
