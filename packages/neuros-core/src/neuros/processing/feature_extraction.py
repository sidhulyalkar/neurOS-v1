"""
Feature extraction utilities.

The :class:`BandPowerExtractor` computes the average spectral power in
canonical EEG frequency bands (delta, theta, alpha, beta, gamma) for each
channel. These features are widely used in BCI literature and serve as a
simple baseline.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch


class BandPowerExtractor:
    """Compute band power features using Welch's method."""

    DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 50.0),
    }

    def __init__(self, fs: float, bands: Dict[str, Tuple[float, float]] | None = None) -> None:
        if fs <= 0:
            raise ValueError("fs must be positive")
        self.fs = fs
        self.bands = bands or self.DEFAULT_BANDS

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute band powers for each channel."""
        data = np.asarray(data)
        if data.ndim == 1:
            data = data[:, np.newaxis]
        if data.ndim != 2:
            raise ValueError("BandPowerExtractor expects [channels] or [channels, samples]")
        n_channels, n_samples = data.shape
        if n_samples == 0:
            raise ValueError("BandPowerExtractor cannot process an empty sample window")

        features: list[float] = []
        for ch in range(n_channels):
            f, pxx = welch(data[ch], fs=self.fs, nperseg=min(256, n_samples))
            for low, high in self.bands.values():
                idx = np.logical_and(f >= low, f <= high)
                # SciPy's trapezoid is stable across the NumPy 1.x -> 2.x
                # transition, where np.trapz was deprecated and later removed.
                band_power = trapezoid(pxx[idx], f[idx]) if np.any(idx) else 0.0
                features.append(float(band_power))
        return np.asarray(features, dtype=np.float32)


class HeartRateExtractor:
    """Compute lightweight baseline ECG statistics."""

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        values = data if data.ndim == 1 else data.flatten()
        return np.array([float(np.mean(values)), float(np.std(values))], dtype=np.float32)


class SkinConductanceExtractor:
    """Compute mean and range from GSR data."""

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        values = data if data.ndim == 1 else data.flatten()
        return np.array(
            [float(np.mean(values)), float(np.max(values) - np.min(values))],
            dtype=np.float32,
        )


class RespirationExtractor:
    """Compute mean and standard deviation of a respiration waveform."""

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        values = data if data.ndim == 1 else data.flatten()
        return np.array([float(np.mean(values)), float(np.std(values))], dtype=np.float32)


class HormoneExtractor:
    """Return the latest value from a slowly varying biochemical signal."""

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        value = float(data[0]) if data.ndim == 1 else float(data[0, -1])
        return np.array([value], dtype=np.float32)


class AudioFeatureExtractor:
    """Compute RMS amplitude and spectral centroid from an audio window."""

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        values = data.flatten() if data.ndim > 1 else data
        rms = float(np.sqrt(np.mean(values**2)))
        n = len(values)
        if n > 0:
            freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
            magnitude = np.abs(np.fft.rfft(values))
            mag_sum = np.sum(magnitude)
            centroid = float(np.sum(freqs * magnitude) / mag_sum) if mag_sum > 0 else 0.0
        else:
            centroid = 0.0
        return np.array([rms, centroid], dtype=np.float32)
