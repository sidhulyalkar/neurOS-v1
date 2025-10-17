"""
Filtering utilities for neural signals.

The provided classes implement simple bandpass and smoothing filters using
SciPy and NumPy.  These filters operate on 1‑D or 2‑D arrays and return
arrays of the same shape.  They can be used inside a processing agent to
clean raw data before feature extraction.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


class BandpassFilter:
    """Butterworth bandpass filter.

    Parameters
    ----------
    lowcut : float
        Low cut‑off frequency in Hz.
    highcut : float
        High cut‑off frequency in Hz.
    fs : float
        Sampling frequency in Hz.
    order : int, optional
        Order of the filter.  Defaults to 4.
    """

    def __init__(self, lowcut: float, highcut: float, fs: float, order: int = 4) -> None:
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b, self.a = butter(order, [low, high], btype="band")

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply the filter to an array of samples.

        The filter is applied along the last axis.  If `data` is 2‑D, each
        channel is filtered independently.
        """
        return lfilter(self.b, self.a, data, axis=-1)


class SmoothingFilter:
    """Simple moving average (boxcar) filter."""

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = max(1, int(window_size))
        self.kernel = np.ones(self.window_size) / self.window_size

    def apply(self, data: np.ndarray) -> np.ndarray:
        # for 1‑D or 2‑D array, convolve along last axis
        return np.apply_along_axis(
            lambda m: np.convolve(m, self.kernel, mode="same"), axis=-1, arr=data
        )