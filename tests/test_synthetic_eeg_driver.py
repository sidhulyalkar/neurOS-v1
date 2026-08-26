from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from neuros.drivers.synthetic_eeg import SyntheticEEGConfig, SyntheticEEGGenerator
from neuros.drivers.synthetic_eeg_driver import (
    SYNTHETIC_EEG_GENERATOR_CONTRACT,
    SyntheticEEGDriver,
)
from neuros.recording.archive import SessionArchiveWriter


def spectral_amplitude(signal: np.ndarray, sampling_rate_hz: float, frequency_hz: float) -> float:
    spectrum = np.fft.rfft(signal - np.mean(signal))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate_hz)
    index = int(np.argmin(np.abs(frequencies - frequency_hz)))
    return float(np.abs(spectrum[index]))


def _render_partitioned(generator: SyntheticEEGGenerator, chunks: tuple[int, ...]) -> np.ndarray:
    return np.concatenate([generator.render(samples).data_uv for samples in chunks], axis=1)


def _provenance_config() -> SyntheticEEGConfig:
    return SyntheticEEGConfig(
        seed=41,
        colored_noise_uv=3.25,
        white_noise_uv=0.75,
        alpha_frequency_hz=10.2,
        alpha_amplitude_uv=1.8,
        ssvep_amplitude_uv=7.2,
        first_harmonic_ratio=0.42,
    )


def _expected_generator_config() -> dict[str, object]:
    return {
        "sampling_rate_hz": 250.0,
        "channel_names": ["Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8"],
        "colored_noise_uv": 3.25,
        "white_noise_uv": 0.75,
        "alpha_frequency_hz": 10.2,
        "alpha_amplitude_uv": 1.8,
        "ssvep_amplitude_uv": 7.2,
        "first_harmonic_ratio": 0.42,
        "seed": 41,
    }


def test_ssvep_strength_is_controllable_and_posterior():
    cfg = SyntheticEEGConfig(seed=11, ssvep_amplitude_uv=8.0)
    strong = SyntheticEEGGenerator(cfg)
    strong.set_attention(12.0, 1.0)
    strong_block = strong.render(500).data_uv

    weak = SyntheticEEGGenerator(cfg)
    weak.set_attention(12.0, 0.2)
    weak_block = weak.render(500).data_uv

    strong_oz = spectral_amplitude(strong_block[6], cfg.sampling_rate_hz, 12.0)
    weak_oz = spectral_amplitude(weak_block[6], cfg.sampling_rate_hz, 12.0)
    strong_fz = spectral_amplitude(strong_block[0], cfg.sampling_rate_hz, 12.0)
    assert strong_oz > weak_oz * 2.0
    assert strong_oz > strong_fz * 4.0


def test_seeded_world_is_invariant_to_render_partitioning():
    cfg = SyntheticEEGConfig(seed=23)
    whole = SyntheticEEGGenerator(cfg)
    whole.set_attention(12.0, 0.7)
    expected = whole.render(1000).data_uv

    partitioned = SyntheticEEGGenerator(cfg)
    partitioned.set_attention(12.0, 0.7)
    actual = _render_partitioned(partitioned, (1, 7, 92, 250, 13, 637))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    assert partitioned.sample_index == whole.sample_index == 1000


def test_seeded_controller_artifact_is_also_partition_invariant():
    cfg = SyntheticEEGConfig(seed=29)
    whole = SyntheticEEGGenerator(cfg)
    whole.set_attention(10.0, 0.5)
    whole.inject_artifact("controller", duration_seconds=0.2, severity=0.8)
    expected = whole.render(100).data_uv

    partitioned = SyntheticEEGGenerator(cfg)
    partitioned.set_attention(10.0, 0.5)
    partitioned.inject_artifact("controller", duration_seconds=0.2, severity=0.8)
    actual = _render_partitioned(partitioned, (11, 26, 9, 54))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)


def test_dropout_duration_is_sample_exact_when_render_block_outlives_artifact():
    cfg = SyntheticEEGConfig(seed=31)
    generator = SyntheticEEGGenerator(cfg)
    generator.inject_artifact("dropout", duration_seconds=10 / cfg.sampling_rate_hz)
    block = generator.render(25).data_uv
    assert np.allclose(block[6, :10], 0.0)
    assert not np.allclose(block[6, 10:], 0.0)


def test_dropout_is_partition_invariant_across_expiration_boundary():
    cfg = SyntheticEEGConfig(seed=37)
    whole = SyntheticEEGGenerator(cfg)
    whole.inject_artifact("dropout", duration_seconds=0.2)
    expected = whole.render(100).data_uv

    partitioned = SyntheticEEGGenerator(cfg)
    partitioned.inject_artifact("dropout", duration_seconds=0.2)
    actual = _render_partitioned(partitioned, (20, 20, 20, 40))

    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-6)
    assert np.allclose(actual[6, :50], 0.0)
    assert not np.allclose(actual[6, 50:], 0.0)


def test_artifacts_and_channel_contact_are_explicit():
    cfg = SyntheticEEGConfig(seed=5)
    generator = SyntheticEEGGenerator(cfg)
    generator.set_attention(10.0)
    generator.inject_artifact("blink", 0.4, 1.0)
    blink = generator.render(100)
    assert blink.artifact == "blink"
    assert float(np.max(np.abs(blink.data_uv[0]))) > 40.0

    generator.set_channel_gain("Oz", 0.0)
    dropout = generator.render(64)
    assert np.allclose(dropout.data_uv[6], 0.0)


def test_driver_exposes_versioned_replay_configuration():
    driver = SyntheticEEGDriver(_provenance_config(), realtime=False, stream_id="phantom")
    descriptor = driver.descriptor
    assert descriptor.stream_id == "phantom"
    assert descriptor.modality == "eeg"
    assert descriptor.sample_rate_hz == 250.0
    assert descriptor.channel_names == ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
    assert descriptor.metadata["synthetic"] is True
    assert descriptor.metadata["generator"] == SYNTHETIC_EEG_GENERATOR_CONTRACT
    assert descriptor.metadata["generator_config"] == _expected_generator_config()


def test_generator_contract_and_seed_survive_session_archive_serialization(tmp_path: Path):
    driver = SyntheticEEGDriver(_provenance_config(), realtime=False, stream_id="phantom")
    archive = SessionArchiveWriter(tmp_path / "session", session_id="synthetic-provenance")
    archive.register_stream(driver.descriptor)

    manifest = json.loads((tmp_path / "session" / "manifest.json").read_text(encoding="utf-8"))
    metadata = manifest["streams"]["phantom"]["descriptor"]["metadata"]
    assert metadata["generator"] == SYNTHETIC_EEG_GENERATOR_CONTRACT
    assert metadata["generator_config"] == _expected_generator_config()
