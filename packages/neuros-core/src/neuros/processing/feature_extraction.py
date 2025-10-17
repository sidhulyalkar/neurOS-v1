"""
Feature extraction utilities.

The :class:`BandPowerExtractor` computes the average spectral power in
canonical EEG frequency bands (delta, theta, alpha, beta, gamma) for each
channel.  These features are widely used in BCI literature and serve as a
simple baseline.
"""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np
from scipy.signal import welch


class BandPowerExtractor:
    """Compute band power features using Welch's method."""

    # Define canonical EEG frequency bands
    DEFAULT_BANDS: Dict[str, Tuple[float, float]] = {
        "delta": (1.0, 4.0),
        "theta": (4.0, 8.0),
        "alpha": (8.0, 13.0),
        "beta": (13.0, 30.0),
        "gamma": (30.0, 50.0),
    }

    def __init__(self, fs: float, bands: Dict[str, Tuple[float, float]] | None = None) -> None:
        self.fs = fs
        self.bands = bands or self.DEFAULT_BANDS

    def extract(self, data: np.ndarray) -> np.ndarray:
        """Compute band powers for each channel.

        Parameters
        ----------
        data : np.ndarray
            Array of shape (channels,) or (channels, samples) containing time
            domain signals.  If 1‑D, it is treated as a single sample per
            channel.

        Returns
        -------
        np.ndarray
            Feature vector of length ``len(bands) * channels``.
        """
        # ensure 2‑D shape (channels, samples)
        if data.ndim == 1:
            # single sample; replicate along new axis
            data = data[:, np.newaxis]
        n_channels, n_samples = data.shape
        features: list[float] = []
        for ch in range(n_channels):
            f, Pxx = welch(data[ch], fs=self.fs, nperseg=min(256, n_samples))
            for band_name, (low, high) in self.bands.items():
                # integrate power spectral density over band
                idx = np.logical_and(f >= low, f <= high)
                band_power = np.trapz(Pxx[idx], f[idx])
                features.append(band_power)
        return np.array(features, dtype=np.float32)


# ---------------------------------------------------------------------------
# New feature extractors for non‑EEG modalities

class HeartRateExtractor:
    """Compute simple heart‑related features from ECG data.

    This extractor calculates the mean and standard deviation of the
    input waveform over the current window.  In a real ECG analysis
    pipeline, more sophisticated methods such as peak detection or
    beat‑to‑beat interval estimation would be used to compute heart
    rate and variability.  However, these simple statistics provide a
    lightweight baseline for multi‑modal experiments.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.  Currently unused but included for
        future extensions.
    """

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        # ensure data is 1‑D (channels,) or (channels, samples)
        if data.ndim == 1:
            values = data
        else:
            # flatten all samples for basic stats
            values = data.flatten()
        mean = float(np.mean(values))
        std = float(np.std(values))
        return np.array([mean, std], dtype=np.float32)


class SkinConductanceExtractor:
    """Compute features from galvanic skin response (GSR) data.

    The extractor returns the mean level and the range (max‑min) of the
    skin conductance within the current window.  These simple features
    capture both tonic and phasic components of the GSR signal.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.  Currently unused but kept for
        consistency.
    """

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            values = data
        else:
            values = data.flatten()
        mean = float(np.mean(values))
        range_val = float(np.max(values) - np.min(values))
        return np.array([mean, range_val], dtype=np.float32)


class RespirationExtractor:
    """Compute features from respiration (breathing) waveform.

    This extractor returns the mean and standard deviation of the
    respiration signal.  More advanced respiratory analytics (e.g.
    breath rate estimation) could be added in future versions.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.  Currently unused but kept for
        consistency.
    """

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        if data.ndim == 1:
            values = data
        else:
            values = data.flatten()
        mean = float(np.mean(values))
        std = float(np.std(values))
        return np.array([mean, std], dtype=np.float32)


class HormoneExtractor:
    """Extract features from a slow hormone or biochemical signal.

    Hormonal signals tend to change slowly, so the current level is
    informative.  This extractor simply returns the latest value.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.  Currently unused but kept for
        consistency.
    """

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        # return the first (and only) value
        if data.ndim == 1:
            value = float(data[0])
        else:
            value = float(data[0, -1])
        return np.array([value], dtype=np.float32)


class AudioFeatureExtractor:
    """Compute audio features from a waveform segment.

    This extractor calculates two simple audio descriptors: the root
    mean square (RMS) amplitude and the spectral centroid.  RMS
    captures the energy of the signal, while the spectral centroid
    characterises its brightness (higher centroid implies more high
    frequencies).  These features are commonly used in audio analysis
    and music information retrieval.

    Parameters
    ----------
    fs : float
        Sampling frequency in Hz.
    """

    def __init__(self, fs: float) -> None:
        self.fs = fs

    def extract(self, data: np.ndarray) -> np.ndarray:
        # flatten to 1‑D
        if data.ndim > 1:
            values = data.flatten()
        else:
            values = data
        # root mean square amplitude
        rms = float(np.sqrt(np.mean(values**2)))
        # spectral centroid: compute discrete Fourier transform
        n = len(values)
        if n > 0:
            freqs = np.fft.rfftfreq(n, d=1.0 / self.fs)
            magnitude = np.abs(np.fft.rfft(values))
            mag_sum = np.sum(magnitude)
            if mag_sum > 0:
                centroid = float(np.sum(freqs * magnitude) / mag_sum)
            else:
                centroid = 0.0
        else:
            centroid = 0.0
        return np.array([rms, centroid], dtype=np.float32)