from __future__ import annotations

import numpy as np

from neuros.drivers.synthetic_eeg import SyntheticEEGConfig, SyntheticEEGGenerator
from neuros.drivers.synthetic_eeg_driver import SyntheticEEGDriver


def spectral_amplitude(signal: np.ndarray, sampling_rate_hz: float, frequency_hz: float) -> float:
    spectrum = np.fft.rfft(signal - np.mean(signal))
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sampling_rate_hz)
    index = int(np.argmin(np.abs(frequencies - frequency_hz)))
    return float(np.abs(spectrum[index]))


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


def test_driver_exposes_unicorn_like_descriptor():
    driver = SyntheticEEGDriver(SyntheticEEGConfig(seed=1), realtime=False, stream_id="phantom")
    descriptor = driver.descriptor
    assert descriptor.stream_id == "phantom"
    assert descriptor.modality == "eeg"
    assert descriptor.sample_rate_hz == 250.0
    assert descriptor.channel_names == ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
    assert descriptor.metadata["synthetic"] is True
