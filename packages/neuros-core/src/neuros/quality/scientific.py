"""Small deterministic scientific oracles for BCI processing validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from neuros.processing.feature_extraction import BandPowerExtractor


@dataclass(frozen=True, slots=True)
class FrequencyProbeResult:
    frequency_hz: float
    expected_band: str
    winning_band: str
    powers: dict[str, float]
    selectivity_ratio: float

    @property
    def passed(self) -> bool:
        return self.winning_band == self.expected_band and self.selectivity_ratio > 1.0


def synthetic_tone(
    frequency_hz: float,
    *,
    sample_rate_hz: float = 250.0,
    duration_s: float = 4.0,
    amplitude: float = 1.0,
    noise_std: float = 0.02,
    seed: int = 0,
) -> np.ndarray:
    if frequency_hz <= 0 or sample_rate_hz <= 0 or duration_s <= 0:
        raise ValueError("frequency, sample rate, and duration must be positive")
    rng = np.random.default_rng(seed)
    t = np.arange(int(sample_rate_hz * duration_s), dtype=float) / sample_rate_hz
    signal = amplitude * np.sin(2.0 * np.pi * frequency_hz * t)
    signal += rng.normal(0.0, noise_std, size=signal.shape)
    return signal[np.newaxis, :].astype(np.float32)


def expected_eeg_band(frequency_hz: float) -> str:
    for name, (low, high) in BandPowerExtractor.DEFAULT_BANDS.items():
        if low <= frequency_hz <= high:
            return name
    raise ValueError(f"Frequency {frequency_hz} Hz is outside canonical EEG bands")


def frequency_selectivity_probe(
    frequency_hz: float,
    *,
    sample_rate_hz: float = 250.0,
    seed: int = 0,
) -> FrequencyProbeResult:
    signal = synthetic_tone(
        frequency_hz,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
    )
    extractor = BandPowerExtractor(fs=sample_rate_hz)
    features = extractor.extract(signal)
    names = list(extractor.bands)
    powers = {name: float(features[index]) for index, name in enumerate(names)}
    ordered = sorted(powers.items(), key=lambda item: item[1], reverse=True)
    winner, top_power = ordered[0]
    second_power = max(ordered[1][1], np.finfo(float).eps)
    return FrequencyProbeResult(
        frequency_hz=frequency_hz,
        expected_band=expected_eeg_band(frequency_hz),
        winning_band=winner,
        powers=powers,
        selectivity_ratio=top_power / second_power,
    )
