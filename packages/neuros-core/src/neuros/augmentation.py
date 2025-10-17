"""
Data augmentation utilities for EEG/BCI signals.

This module provides domain-specific augmentation techniques for EEG and other
biosignals, designed to improve model generalization while preserving the
neurophysiological characteristics of the data.
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, List, Callable
from enum import Enum

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

logger = logging.getLogger(__name__)


class AugmentationType(Enum):
    """Enumeration of available augmentation types."""

    TIME_SHIFT = "time_shift"
    AMPLITUDE_SCALE = "amplitude_scale"
    GAUSSIAN_NOISE = "gaussian_noise"
    TIME_WARP = "time_warp"
    CHANNEL_DROPOUT = "channel_dropout"
    FREQUENCY_SHIFT = "frequency_shift"
    SMOOTH = "smooth"
    MIXUP = "mixup"


def time_shift(
    X: np.ndarray, max_shift: int = 10, *, random_state: Optional[int] = None
) -> np.ndarray:
    """Shift signals in time by a random amount.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints) or
        (n_channels, n_timepoints).
    max_shift : int, default=10
        Maximum number of timepoints to shift (in either direction).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Time-shifted data with same shape as input.

    Examples
    --------
    >>> X = np.random.randn(10, 8, 200)  # 10 samples, 8 channels, 200 timepoints
    >>> X_aug = time_shift(X, max_shift=20)
    >>> X_aug.shape
    (10, 8, 200)
    """
    rng = np.random.default_rng(random_state)

    if X.ndim == 2:
        # Single sample: (n_channels, n_timepoints)
        shift = rng.integers(-max_shift, max_shift + 1)
        return np.roll(X, shift, axis=1)
    elif X.ndim == 3:
        # Multiple samples: (n_samples, n_channels, n_timepoints)
        X_aug = X.copy()
        for i in range(len(X)):
            shift = rng.integers(-max_shift, max_shift + 1)
            X_aug[i] = np.roll(X[i], shift, axis=1)
        return X_aug
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")


def amplitude_scale(
    X: np.ndarray,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    *,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Scale signal amplitudes by a random factor.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints) or
        (n_channels, n_timepoints).
    scale_range : tuple of float, default=(0.8, 1.2)
        Range for amplitude scaling factors (min, max).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Amplitude-scaled data with same shape as input.

    Examples
    --------
    >>> X = np.random.randn(10, 8, 200)
    >>> X_aug = amplitude_scale(X, scale_range=(0.5, 1.5))
    """
    rng = np.random.default_rng(random_state)

    if X.ndim == 2:
        # Single sample: apply same scale to all channels
        scale = rng.uniform(scale_range[0], scale_range[1])
        return X * scale
    elif X.ndim == 3:
        # Multiple samples: different scale per sample
        X_aug = X.copy()
        for i in range(len(X)):
            scale = rng.uniform(scale_range[0], scale_range[1])
            X_aug[i] = X[i] * scale
        return X_aug
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")


def gaussian_noise(
    X: np.ndarray, noise_level: float = 0.05, *, random_state: Optional[int] = None
) -> np.ndarray:
    """Add Gaussian noise to signals.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints) or
        (n_channels, n_timepoints).
    noise_level : float, default=0.05
        Standard deviation of noise relative to signal std.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Noisy data with same shape as input.

    Examples
    --------
    >>> X = np.random.randn(10, 8, 200)
    >>> X_noisy = gaussian_noise(X, noise_level=0.1)
    """
    rng = np.random.default_rng(random_state)

    signal_std = np.std(X)
    noise = rng.normal(0, noise_level * signal_std, X.shape)
    return X + noise


def channel_dropout(
    X: np.ndarray, dropout_prob: float = 0.2, *, random_state: Optional[int] = None
) -> np.ndarray:
    """Randomly zero out entire channels to simulate electrode failures.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints) or
        (n_channels, n_timepoints).
    dropout_prob : float, default=0.2
        Probability of dropping each channel.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Data with some channels zeroed out.

    Examples
    --------
    >>> X = np.random.randn(10, 8, 200)
    >>> X_aug = channel_dropout(X, dropout_prob=0.3)
    """
    rng = np.random.default_rng(random_state)

    if X.ndim == 2:
        # Single sample: (n_channels, n_timepoints)
        n_channels = X.shape[0]
        mask = rng.random(n_channels) > dropout_prob
        X_aug = X.copy()
        X_aug[~mask, :] = 0
        return X_aug
    elif X.ndim == 3:
        # Multiple samples: (n_samples, n_channels, n_timepoints)
        X_aug = X.copy()
        for i in range(len(X)):
            n_channels = X.shape[1]
            mask = rng.random(n_channels) > dropout_prob
            X_aug[i, ~mask, :] = 0
        return X_aug
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")


def time_warp(
    X: np.ndarray, warp_factor: float = 0.1, *, random_state: Optional[int] = None
) -> np.ndarray:
    """Apply time warping by resampling signals with variable speed.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints) or
        (n_channels, n_timepoints).
    warp_factor : float, default=0.1
        Maximum relative change in time scale (0.1 = ±10%).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Time-warped data with same shape as input.

    Examples
    --------
    >>> X = np.random.randn(10, 8, 200)
    >>> X_warped = time_warp(X, warp_factor=0.15)
    """
    rng = np.random.default_rng(random_state)

    if X.ndim == 2:
        # Single sample: (n_channels, n_timepoints)
        n_channels, n_timepoints = X.shape
        warp_scale = 1 + rng.uniform(-warp_factor, warp_factor)
        new_length = int(n_timepoints * warp_scale)

        # Resample and then crop/pad to original length
        X_warped = np.zeros_like(X)
        for ch in range(n_channels):
            resampled = signal.resample(X[ch], new_length)
            if new_length > n_timepoints:
                # Crop
                X_warped[ch] = resampled[:n_timepoints]
            else:
                # Pad
                X_warped[ch, :new_length] = resampled
        return X_warped
    elif X.ndim == 3:
        # Multiple samples
        X_aug = np.zeros_like(X)
        for i in range(len(X)):
            X_aug[i] = time_warp(X[i], warp_factor=warp_factor, random_state=rng)
        return X_aug
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")


def frequency_shift(
    X: np.ndarray,
    fs: float,
    shift_range: Tuple[float, float] = (-2.0, 2.0),
    *,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """Shift signal frequencies by modulating with a carrier frequency.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints) or
        (n_channels, n_timepoints).
    fs : float
        Sampling frequency in Hz.
    shift_range : tuple of float, default=(-2.0, 2.0)
        Range of frequency shifts in Hz (min, max).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Frequency-shifted data with same shape as input.

    Examples
    --------
    >>> X = np.random.randn(10, 8, 200)
    >>> X_shifted = frequency_shift(X, fs=250.0, shift_range=(-3, 3))
    """
    rng = np.random.default_rng(random_state)

    if X.ndim == 2:
        # Single sample
        n_channels, n_timepoints = X.shape
        shift_hz = rng.uniform(shift_range[0], shift_range[1])
        t = np.arange(n_timepoints) / fs
        carrier = np.exp(1j * 2 * np.pi * shift_hz * t)

        X_shifted = np.zeros_like(X)
        for ch in range(n_channels):
            # Apply frequency shift via complex modulation
            fft_signal = np.fft.fft(X[ch])
            fft_carrier = np.fft.fft(carrier)
            shifted_fft = fft_signal * np.abs(fft_carrier)
            X_shifted[ch] = np.real(np.fft.ifft(shifted_fft))
        return X_shifted
    elif X.ndim == 3:
        # Multiple samples
        X_aug = np.zeros_like(X)
        for i in range(len(X)):
            X_aug[i] = frequency_shift(
                X[i], fs=fs, shift_range=shift_range, random_state=rng
            )
        return X_aug
    else:
        raise ValueError(f"Expected 2D or 3D array, got shape {X.shape}")


def smooth(
    X: np.ndarray, sigma: float = 1.0, *, axis: int = -1
) -> np.ndarray:
    """Apply Gaussian smoothing along the time axis.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints) or
        (n_channels, n_timepoints).
    sigma : float, default=1.0
        Standard deviation for Gaussian kernel.
    axis : int, default=-1
        Axis along which to smooth (typically the time axis).

    Returns
    -------
    np.ndarray
        Smoothed data with same shape as input.

    Examples
    --------
    >>> X = np.random.randn(10, 8, 200)
    >>> X_smooth = smooth(X, sigma=2.0)
    """
    return gaussian_filter1d(X, sigma=sigma, axis=axis)


def mixup(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float = 0.2,
    *,
    random_state: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply mixup augmentation by linearly interpolating between samples.

    Mixup creates virtual training examples by mixing pairs of samples and
    their labels, which can improve generalization.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, ...).
    y : np.ndarray
        Labels of shape (n_samples,) or one-hot encoded (n_samples, n_classes).
    alpha : float, default=0.2
        Beta distribution parameter for mixup ratio.
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    X_mixed : np.ndarray
        Mixed input data.
    y_mixed : np.ndarray
        Mixed labels (soft labels).

    Examples
    --------
    >>> X = np.random.randn(100, 8, 200)
    >>> y = np.random.randint(0, 2, 100)
    >>> X_mixed, y_mixed = mixup(X, y, alpha=0.3)

    References
    ----------
    Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
    """
    rng = np.random.default_rng(random_state)

    n_samples = len(X)
    indices = rng.permutation(n_samples)

    # Sample mixing ratio from Beta distribution
    lam = rng.beta(alpha, alpha) if alpha > 0 else 1.0

    # Mix inputs
    X_mixed = lam * X + (1 - lam) * X[indices]

    # Convert labels to one-hot if needed
    if y.ndim == 1:
        # Get max label value to determine n_classes (handles missing intermediate classes)
        n_classes = int(np.max(y)) + 1
        y_onehot = np.eye(n_classes)[y]
    else:
        y_onehot = y

    # Mix labels
    y_mixed = lam * y_onehot + (1 - lam) * y_onehot[indices]

    return X_mixed, y_mixed


def augment_batch(
    X: np.ndarray,
    y: Optional[np.ndarray] = None,
    *,
    augmentations: Optional[List[str]] = None,
    fs: float = 250.0,
    random_state: Optional[int] = None,
    **kwargs,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Apply multiple augmentations to a batch of EEG data.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (n_samples, n_channels, n_timepoints).
    y : np.ndarray, optional
        Labels (required for mixup).
    augmentations : list of str, optional
        List of augmentation names to apply. Default: ['time_shift', 'amplitude_scale', 'gaussian_noise'].
    fs : float, default=250.0
        Sampling frequency in Hz (required for frequency_shift).
    random_state : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional keyword arguments for specific augmentations.

    Returns
    -------
    X_aug : np.ndarray
        Augmented data.
    y_aug : np.ndarray, optional
        Augmented labels (only different if mixup is used).

    Examples
    --------
    >>> X = np.random.randn(100, 8, 200)
    >>> y = np.random.randint(0, 2, 100)
    >>> X_aug, y_aug = augment_batch(X, y, augmentations=['time_shift', 'gaussian_noise'])
    """
    if augmentations is None:
        augmentations = ["time_shift", "amplitude_scale", "gaussian_noise"]

    rng = np.random.default_rng(random_state)
    X_aug = X.copy()
    y_aug = y.copy() if y is not None else None

    for aug_name in augmentations:
        try:
            if aug_name == "time_shift":
                max_shift = kwargs.get("max_shift", 10)
                X_aug = time_shift(X_aug, max_shift=max_shift, random_state=rng)
            elif aug_name == "amplitude_scale":
                scale_range = kwargs.get("scale_range", (0.8, 1.2))
                X_aug = amplitude_scale(X_aug, scale_range=scale_range, random_state=rng)
            elif aug_name == "gaussian_noise":
                noise_level = kwargs.get("noise_level", 0.05)
                X_aug = gaussian_noise(X_aug, noise_level=noise_level, random_state=rng)
            elif aug_name == "time_warp":
                warp_factor = kwargs.get("warp_factor", 0.1)
                X_aug = time_warp(X_aug, warp_factor=warp_factor, random_state=rng)
            elif aug_name == "channel_dropout":
                dropout_prob = kwargs.get("dropout_prob", 0.2)
                X_aug = channel_dropout(X_aug, dropout_prob=dropout_prob, random_state=rng)
            elif aug_name == "frequency_shift":
                shift_range = kwargs.get("shift_range", (-2.0, 2.0))
                X_aug = frequency_shift(X_aug, fs=fs, shift_range=shift_range, random_state=rng)
            elif aug_name == "smooth":
                sigma = kwargs.get("sigma", 1.0)
                X_aug = smooth(X_aug, sigma=sigma)
            elif aug_name == "mixup":
                if y is None:
                    logger.warning("Mixup requires labels, skipping")
                else:
                    alpha = kwargs.get("alpha", 0.2)
                    X_aug, y_aug = mixup(X_aug, y_aug, alpha=alpha, random_state=rng)
            else:
                logger.warning(f"Unknown augmentation: {aug_name}")
        except Exception as e:
            logger.error(f"Failed to apply {aug_name}: {e}")

    return X_aug, y_aug


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations sequentially.

    Parameters
    ----------
    augmentations : list of tuple
        List of (augmentation_name, kwargs) tuples.
    fs : float, default=250.0
        Sampling frequency in Hz.
    random_state : int, optional
        Random seed for reproducibility.

    Examples
    --------
    >>> pipeline = AugmentationPipeline([
    ...     ('time_shift', {'max_shift': 15}),
    ...     ('gaussian_noise', {'noise_level': 0.08}),
    ...     ('amplitude_scale', {'scale_range': (0.7, 1.3)}),
    ... ])
    >>> X_aug = pipeline.transform(X)
    """

    def __init__(
        self,
        augmentations: List[Tuple[str, dict]],
        fs: float = 250.0,
        random_state: Optional[int] = None,
    ):
        self.augmentations = augmentations
        self.fs = fs
        self.random_state = random_state

    def transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Apply all augmentations in the pipeline.

        Parameters
        ----------
        X : np.ndarray
            Input data.
        y : np.ndarray, optional
            Labels (required for mixup).

        Returns
        -------
        X_aug : np.ndarray
            Augmented data.
        y_aug : np.ndarray, optional
            Augmented labels.
        """
        aug_names = [name for name, _ in self.augmentations]
        aug_kwargs = {}
        for name, kwargs in self.augmentations:
            aug_kwargs.update(kwargs)

        return augment_batch(
            X,
            y,
            augmentations=aug_names,
            fs=self.fs,
            random_state=self.random_state,
            **aug_kwargs,
        )

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None):
        """Fit method for sklearn compatibility (no-op)."""
        return self

    def fit_transform(
        self, X: np.ndarray, y: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Fit and transform (same as transform for augmentation)."""
        return self.transform(X, y)
