"""
Optimized filtering utilities for neural signals.

This module provides performance-optimized versions of filters with
caching and vectorization for improved throughput.
"""

from __future__ import annotations

from functools import lru_cache
import numpy as np
from scipy.signal import butter, lfilter, sosfilt, sosfiltfilt


class OptimizedBandpassFilter:
    """
    Optimized Butterworth bandpass filter with coefficient caching.

    Performance improvements:
    - Uses second-order sections (sos) for numerical stability and speed
    - Caches filter coefficients
    - Vectorized application across channels
    """

    def __init__(self, lowcut: float, highcut: float, fs: float, order: int = 4) -> None:
        self.lowcut = lowcut
        self.highcut = highcut
        self.fs = fs
        self.order = order

        # Precompute SOS filter coefficients
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.sos = butter(order, [low, high], btype="band", output='sos')

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the filter to an array of samples using SOS.

        The filter is applied along the last axis. If `data` is 2-D, each
        channel is filtered independently.

        Parameters
        ----------
        data : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Filtered data
        """
        # sosfilt is typically faster and more numerically stable than lfilter
        return sosfilt(self.sos, data, axis=-1)

    def apply_zero_phase(self, data: np.ndarray) -> np.ndarray:
        """
        Apply zero-phase filtering using forward-backward filtering.

        This doubles the filter order but eliminates phase distortion.

        Parameters
        ----------
        data : np.ndarray
            Input data

        Returns
        -------
        np.ndarray
            Zero-phase filtered data
        """
        return sosfiltfilt(self.sos, data, axis=-1)


class FastSmoothingFilter:
    """
    Optimized moving average filter using numpy's convolve with 'same' mode.

    Performance improvements:
    - Precomputes normalized kernel
    - Uses optimized numpy operations
    - Supports multi-channel vectorized filtering
    """

    def __init__(self, window_size: int = 5) -> None:
        self.window_size = max(1, int(window_size))
        self.kernel = np.ones(self.window_size, dtype=np.float32) / self.window_size

    def apply(self, data: np.ndarray) -> np.ndarray:
        """
        Apply smoothing filter efficiently.

        Parameters
        ----------
        data : np.ndarray
            Input data, shape (channels, samples) or (samples,)

        Returns
        -------
        np.ndarray
            Smoothed data
        """
        if data.ndim == 1:
            return np.convolve(data, self.kernel, mode='same')
        else:
            # Apply to each channel
            return np.apply_along_axis(
                lambda m: np.convolve(m, self.kernel, mode='same'),
                axis=-1,
                arr=data
            )

    def apply_vectorized(self, data: np.ndarray) -> np.ndarray:
        """
        Apply smoothing using vectorized operations (fastest for large arrays).

        Parameters
        ----------
        data : np.ndarray
            Input data, shape (channels, samples)

        Returns
        -------
        np.ndarray
            Smoothed data
        """
        if data.ndim == 1:
            return np.convolve(data, self.kernel, mode='same')

        # For 2D data, use correlation which can be faster
        from scipy.ndimage import correlate1d
        return correlate1d(data, self.kernel, axis=-1, mode='nearest')


class CachedFilterBank:
    """
    Filter bank with cached coefficients for multiple frequency bands.

    Useful for applying multiple bandpass filters efficiently.
    """

    def __init__(self, fs: float, order: int = 4):
        self.fs = fs
        self.order = order
        self._filters = {}

    def add_band(self, band_name: str, lowcut: float, highcut: float):
        """Add a frequency band to the filter bank."""
        filter_obj = OptimizedBandpassFilter(lowcut, highcut, self.fs, self.order)
        self._filters[band_name] = filter_obj

    def apply_bank(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Apply all filters in the bank to the data.

        Parameters
        ----------
        data : np.ndarray
            Input data

        Returns
        -------
        dict
            Dictionary mapping band names to filtered data
        """
        results = {}
        for band_name, filter_obj in self._filters.items():
            results[band_name] = filter_obj.apply(data)
        return results


class AdaptiveNotchFilter:
    """
    Adaptive notch filter for powerline interference removal.

    Optimized for real-time applications with minimal latency.
    """

    def __init__(self, fs: float, freq: float = 60.0, Q: float = 30.0):
        """
        Initialize adaptive notch filter.

        Parameters
        ----------
        fs : float
            Sampling frequency in Hz
        freq : float
            Frequency to notch out (default 60 Hz for US powerline)
        Q : float
            Quality factor (higher = narrower notch)
        """
        self.fs = fs
        self.freq = freq
        self.Q = Q

        # Precompute notch filter coefficients
        from scipy.signal import iirnotch
        b, a = iirnotch(freq, Q, fs)
        self.b = b
        self.a = a

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to data."""
        return lfilter(self.b, self.a, data, axis=-1)


class BatchFilterProcessor:
    """
    Process multiple trials through a filter chain efficiently.

    Optimized for batch processing with minimal overhead.
    """

    def __init__(self, filters: list):
        """
        Initialize batch filter processor.

        Parameters
        ----------
        filters : list
            List of filter objects (each must have an 'apply' method)
        """
        self.filters = filters

    def process_batch(self, data_batch: np.ndarray) -> np.ndarray:
        """
        Process a batch of trials through the filter chain.

        Parameters
        ----------
        data_batch : np.ndarray
            Shape (n_trials, n_channels, n_samples)

        Returns
        -------
        np.ndarray
            Filtered batch, same shape as input
        """
        filtered = data_batch.copy()

        # Apply each filter to all trials
        for filter_obj in self.filters:
            for trial_idx in range(filtered.shape[0]):
                filtered[trial_idx] = filter_obj.apply(filtered[trial_idx])

        return filtered

    def process_batch_parallel(self, data_batch: np.ndarray, n_jobs: int = -1) -> np.ndarray:
        """
        Process batch in parallel using joblib.

        Parameters
        ----------
        data_batch : np.ndarray
            Shape (n_trials, n_channels, n_samples)
        n_jobs : int
            Number of parallel jobs (-1 = all CPUs)

        Returns
        -------
        np.ndarray
            Filtered batch
        """
        try:
            from joblib import Parallel, delayed

            def process_trial(trial):
                filtered_trial = trial.copy()
                for filter_obj in self.filters:
                    filtered_trial = filter_obj.apply(filtered_trial)
                return filtered_trial

            results = Parallel(n_jobs=n_jobs)(
                delayed(process_trial)(data_batch[i])
                for i in range(data_batch.shape[0])
            )

            return np.array(results)

        except ImportError:
            # Fall back to serial processing if joblib not available
            return self.process_batch(data_batch)


# Backwards compatibility
BandpassFilter = OptimizedBandpassFilter
SmoothingFilter = FastSmoothingFilter
