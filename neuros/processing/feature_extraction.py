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