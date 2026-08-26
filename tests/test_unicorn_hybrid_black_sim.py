from __future__ import annotations

import numpy as np

from neuros.drivers.synthetic_eeg import SyntheticEEGConfig, SyntheticEEGGenerator
from neuros.drivers.unicorn_hybrid_black_sim import (
    UNICORN_DEVICE17_NAMES,
    UNICORN_RECORDER19_NAMES,
    UNICORN_SCALP_LABELS,
    UnicornHybridBlackSimulationConfig,
    UnicornHybridBlackSimulator,
    UnicornHybridBlackSpec,
    validate_unicorn_block,
)


def _render_device_partitioned(
    simulator: UnicornHybridBlackSimulator,
    chunks: tuple[int, ...],
) -> dict[str, np.ndarray]:
    blocks = [simulator.render(samples) for samples in chunks]
    return {
        "data": np.concatenate([block.data for block in blocks], axis=1),
        "eeg": np.concatenate([block.eeg_data_uv for block in blocks], axis=1),
        "sample_time": np.concatenate([block.sample_timestamps_s for block in blocks]),
        "available_time": np.concatenate([block.available_timestamps_s for block in blocks]),
        "counter": np.concatenate([block.counter for block in blocks]),
        "battery": np.concatenate([block.battery_percent for block in blocks]),
        "validation": np.concatenate([block.validation for block in blocks]),
    }


def test_published_device_constants_and_quantization_are_explicit():
    spec = UnicornHybridBlackSpec()
    spec.validate()
    assert spec.sampling_rate_hz == 250.0
    assert spec.eeg_channels == 8
    assert spec.acquired_channels == 17
    assert spec.resolution_bits == 24
    assert spec.sensitivity_uv == 750_000.0
    assert spec.input_impedance_lower_bound_ohm == 1_000_000_000.0
    assert spec.device_delay_ms == 40.0
    assert 0.08 < spec.eeg_lsb_uv < 0.10


def test_full_device_scan_matches_17_channel_api_order_and_units():
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(schema="device17_api", seed=11)
    )
    block = sim.render(25)
    assert block.data.shape == (17, 25)
    assert block.channel_names == UNICORN_DEVICE17_NAMES
    assert block.channel_units[:8] == ("microvolts",) * 8
    assert block.channel_units[8:11] == ("g",) * 3
    assert block.channel_units[11:14] == ("deg/s",) * 3
    assert block.channel_units[14:] == ("count", "percent", "boolean")
    assert np.all(np.diff(block.counter) == 1)
    assert np.all(block.validation == 1)
    report = validate_unicorn_block(block)
    assert report.passed, report.to_dict()


def test_full_device_world_is_invariant_to_render_partitioning():
    config = UnicornHybridBlackSimulationConfig(
        schema="device17_api",
        seed=13,
        accelerometer_noise_g=0.003,
        gyroscope_noise_dps=0.08,
        acquisition_delay_jitter_ms=0.5,
        counter_start=500,
    )

    whole = UnicornHybridBlackSimulator(config=config)
    whole.eeg.set_attention(12.0, 0.65)
    expected_block = whole.render(500)
    expected = {
        "data": expected_block.data,
        "eeg": expected_block.eeg_data_uv,
        "sample_time": expected_block.sample_timestamps_s,
        "available_time": expected_block.available_timestamps_s,
        "counter": expected_block.counter,
        "battery": expected_block.battery_percent,
        "validation": expected_block.validation,
    }

    partitioned = UnicornHybridBlackSimulator(config=config)
    partitioned.eeg.set_attention(12.0, 0.65)
    actual = _render_device_partitioned(
        partitioned,
        (1, 7, 42, 100, 3, 97, 250),
    )

    for key in ("data", "eeg", "sample_time", "available_time", "battery"):
        np.testing.assert_allclose(actual[key], expected[key], rtol=0.0, atol=1e-6)
    np.testing.assert_array_equal(actual["counter"], expected["counter"])
    np.testing.assert_array_equal(actual["validation"], expected["validation"])
    assert partitioned.counter_value == whole.counter_value == 1000


def test_recorder_view_adds_delta_time_and_status_without_inventing_eeg_channels():
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(schema="recorder19", seed=17)
    )
    sim.set_status(23)
    block = sim.render(20)
    assert block.data.shape == (19, 20)
    assert block.channel_names == UNICORN_RECORDER19_NAMES
    assert np.allclose(block.data[17], 4.0)
    assert np.all(block.data[18] == 23.0)
    report = validate_unicorn_block(block)
    assert report.passed, report.to_dict()


def test_anatomical_eeg_view_keeps_standard_cap_order_for_game_decoders():
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(schema="eeg8_anatomical", seed=19)
    )
    block = sim.render(10)
    assert block.data.shape == (8, 10)
    assert block.channel_names == UNICORN_SCALP_LABELS
    assert validate_unicorn_block(block).passed


def test_motion_aux_channels_do_not_silently_change_eeg_physiology():
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(
            schema="device17_api",
            seed=23,
            accelerometer_noise_g=0.0,
            gyroscope_noise_dps=0.0,
        )
    )
    sim.set_motion((0.25, -0.50, 0.90), (25.0, -10.0, 4.0))
    block = sim.render(12)
    assert np.allclose(block.data[8:11], np.asarray([[0.25], [-0.50], [0.90]]))
    assert np.allclose(block.data[11:14], np.asarray([[25.0], [-10.0], [4.0]]))
    # Motion telemetry is a device observable. Motion-to-EEG contamination must
    # be injected explicitly in the neural/world layer rather than hidden here.
    assert np.all(np.isfinite(block.eeg_data_uv))


def test_validation_and_counter_can_exercise_fail_closed_consumers():
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(
            schema="device17_api",
            seed=29,
            counter_start=100,
        )
    )
    first = sim.render(5)
    sim.set_validation(0)
    second = sim.render(5)
    assert first.counter.tolist() == [100, 101, 102, 103, 104]
    assert second.counter.tolist() == [105, 106, 107, 108, 109]
    assert np.all(first.validation == 1)
    assert np.all(second.validation == 0)


def test_acquisition_availability_delay_is_separate_from_neural_sample_time():
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(
            schema="device17_api",
            seed=31,
            acquisition_delay_jitter_ms=0.0,
        )
    )
    block = sim.render(50)
    delay_ms = (
        block.available_timestamps_s - block.sample_timestamps_s
    ) * 1000.0
    assert np.allclose(delay_ms, 40.0)
    assert np.allclose(np.diff(block.sample_timestamps_s), 1.0 / 250.0)


def test_24_bit_clipping_and_quantization_are_applied_at_device_boundary():
    generator = SyntheticEEGGenerator(
        SyntheticEEGConfig(
            sampling_rate_hz=250.0,
            channel_names=UNICORN_SCALP_LABELS,
            colored_noise_uv=0.0,
            white_noise_uv=0.0,
            alpha_amplitude_uv=0.0,
            ssvep_amplitude_uv=0.0,
            seed=37,
        )
    )
    generator.inject_artifact("saturation", duration_seconds=0.1, severity=3000.0)
    sim = UnicornHybridBlackSimulator(
        generator,
        config=UnicornHybridBlackSimulationConfig(
            schema="eeg8_anatomical",
            seed=37,
        ),
    )
    block = sim.render(10)
    spec = UnicornHybridBlackSpec()
    assert block.clipped_fraction > 0.0
    assert np.max(np.abs(block.eeg_data_uv)) <= spec.sensitivity_uv + spec.eeg_lsb_uv
    # Quantized values should sit on the advertised 24-bit grid to floating
    # precision, after subtracting the negative full-scale origin.
    grid = (
        block.eeg_data_uv.astype(float) + spec.sensitivity_uv
    ) / spec.eeg_lsb_uv
    assert np.max(np.abs(grid - np.round(grid))) < 0.25
