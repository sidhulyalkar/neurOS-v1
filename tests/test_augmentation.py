"""
Tests for data augmentation utilities.
"""

import numpy as np
import pytest

from neuros.augmentation import (
    time_shift,
    amplitude_scale,
    gaussian_noise,
    channel_dropout,
    time_warp,
    frequency_shift,
    smooth,
    mixup,
    augment_batch,
    AugmentationPipeline,
)


class TestTimeShift:
    """Test time shifting augmentation."""

    def test_time_shift_2d(self):
        """Test time shift on single sample."""
        X = np.random.randn(8, 200)  # 8 channels, 200 timepoints
        X_aug = time_shift(X, max_shift=10, random_state=42)

        assert X_aug.shape == X.shape
        # Data should be different but similar magnitude
        assert not np.allclose(X, X_aug)

    def test_time_shift_3d(self):
        """Test time shift on batch."""
        X = np.random.randn(20, 8, 200)  # 20 samples, 8 channels, 200 timepoints
        X_aug = time_shift(X, max_shift=15, random_state=42)

        assert X_aug.shape == X.shape

    def test_time_shift_reproducible(self):
        """Test that time shift is reproducible with same seed."""
        X = np.random.randn(10, 8, 200)

        X_aug1 = time_shift(X, max_shift=10, random_state=42)
        X_aug2 = time_shift(X, max_shift=10, random_state=42)

        np.testing.assert_array_equal(X_aug1, X_aug2)


class TestAmplitudeScale:
    """Test amplitude scaling augmentation."""

    def test_amplitude_scale_2d(self):
        """Test amplitude scaling on single sample."""
        X = np.random.randn(8, 200)
        X_aug = amplitude_scale(X, scale_range=(0.5, 1.5), random_state=42)

        assert X_aug.shape == X.shape
        # Scaled data should have different magnitude
        assert not np.allclose(np.std(X), np.std(X_aug))

    def test_amplitude_scale_3d(self):
        """Test amplitude scaling on batch."""
        X = np.random.randn(20, 8, 200)
        X_aug = amplitude_scale(X, scale_range=(0.8, 1.2), random_state=42)

        assert X_aug.shape == X.shape

    def test_amplitude_scale_range(self):
        """Test that scaling is within specified range."""
        X = np.ones((5, 8, 100))  # Constant signal
        X_aug = amplitude_scale(X, scale_range=(2.0, 3.0), random_state=42)

        # All samples should be scaled between 2x and 3x
        for i in range(len(X_aug)):
            sample_mean = np.mean(X_aug[i])
            assert 2.0 <= sample_mean <= 3.0


class TestGaussianNoise:
    """Test Gaussian noise augmentation."""

    def test_gaussian_noise_2d(self):
        """Test noise addition on single sample."""
        X = np.random.randn(8, 200)
        X_aug = gaussian_noise(X, noise_level=0.1, random_state=42)

        assert X_aug.shape == X.shape
        assert not np.allclose(X, X_aug)

    def test_gaussian_noise_3d(self):
        """Test noise addition on batch."""
        X = np.random.randn(20, 8, 200)
        X_aug = gaussian_noise(X, noise_level=0.05, random_state=42)

        assert X_aug.shape == X.shape

    def test_gaussian_noise_level(self):
        """Test that noise level affects variance."""
        X = np.random.randn(10, 8, 100)  # Random signal with non-zero std

        X_low_noise = gaussian_noise(X, noise_level=0.01, random_state=42)
        X_high_noise = gaussian_noise(X, noise_level=0.5, random_state=43)

        # Higher noise should have larger deviation from original
        low_noise_mag = np.mean((X_low_noise - X) ** 2)
        high_noise_mag = np.mean((X_high_noise - X) ** 2)
        assert high_noise_mag > low_noise_mag


class TestChannelDropout:
    """Test channel dropout augmentation."""

    def test_channel_dropout_2d(self):
        """Test channel dropout on single sample."""
        X = np.ones((8, 200))  # All ones
        X_aug = channel_dropout(X, dropout_prob=0.5, random_state=42)

        assert X_aug.shape == X.shape
        # Some channels should be zeroed
        zero_channels = np.sum(np.all(X_aug == 0, axis=1))
        assert zero_channels > 0

    def test_channel_dropout_3d(self):
        """Test channel dropout on batch."""
        X = np.ones((20, 8, 200))
        X_aug = channel_dropout(X, dropout_prob=0.3, random_state=42)

        assert X_aug.shape == X.shape

    def test_channel_dropout_probability(self):
        """Test that dropout probability is approximately correct."""
        X = np.ones((100, 10, 50))  # 100 samples, 10 channels
        X_aug = channel_dropout(X, dropout_prob=0.5, random_state=42)

        # Count how many channels were dropped across all samples
        total_channels = 100 * 10
        dropped_channels = 0
        for i in range(100):
            dropped_channels += np.sum(np.all(X_aug[i] == 0, axis=1))

        dropout_rate = dropped_channels / total_channels
        # Should be approximately 0.5 (within 20% tolerance)
        assert abs(dropout_rate - 0.5) < 0.2


class TestTimeWarp:
    """Test time warping augmentation."""

    def test_time_warp_2d(self):
        """Test time warp on single sample."""
        X = np.random.randn(8, 200)
        X_aug = time_warp(X, warp_factor=0.1, random_state=42)

        assert X_aug.shape == X.shape

    def test_time_warp_3d(self):
        """Test time warp on batch."""
        X = np.random.randn(20, 8, 200)
        X_aug = time_warp(X, warp_factor=0.15, random_state=42)

        assert X_aug.shape == X.shape


class TestFrequencyShift:
    """Test frequency shifting augmentation."""

    def test_frequency_shift_2d(self):
        """Test frequency shift on single sample."""
        X = np.random.randn(8, 200)
        X_aug = frequency_shift(X, fs=250.0, shift_range=(-2, 2), random_state=42)

        assert X_aug.shape == X.shape

    def test_frequency_shift_3d(self):
        """Test frequency shift on batch."""
        X = np.random.randn(20, 8, 200)
        X_aug = frequency_shift(X, fs=250.0, shift_range=(-3, 3), random_state=42)

        assert X_aug.shape == X.shape


class TestSmooth:
    """Test smoothing augmentation."""

    def test_smooth_2d(self):
        """Test smoothing on single sample."""
        X = np.random.randn(8, 200)
        X_smooth = smooth(X, sigma=2.0)

        assert X_smooth.shape == X.shape
        # Smoothed signal should have lower variance
        assert np.std(X_smooth) < np.std(X)

    def test_smooth_3d(self):
        """Test smoothing on batch."""
        X = np.random.randn(20, 8, 200)
        X_smooth = smooth(X, sigma=1.5)

        assert X_smooth.shape == X.shape

    def test_smooth_reduces_high_frequency(self):
        """Test that smoothing reduces high-frequency content."""
        # Create signal with high-frequency noise
        t = np.linspace(0, 1, 200)
        signal_clean = np.sin(2 * np.pi * 5 * t)  # 5 Hz sine wave
        noise = 0.5 * np.sin(2 * np.pi * 50 * t)  # 50 Hz noise
        X = (signal_clean + noise).reshape(1, -1)

        X_smooth = smooth(X, sigma=3.0)

        # Smoothed signal should be closer to clean signal
        assert np.mean((X_smooth - signal_clean) ** 2) < np.mean((X - signal_clean) ** 2)


class TestMixup:
    """Test mixup augmentation."""

    def test_mixup_with_1d_labels(self):
        """Test mixup with integer labels."""
        X = np.random.randn(20, 8, 200)
        y = np.random.randint(0, 2, 20)

        X_mixed, y_mixed = mixup(X, y, alpha=0.2, random_state=42)

        assert X_mixed.shape == X.shape
        assert y_mixed.shape == (20, 2)  # One-hot encoded
        # Mixed labels should be soft (between 0 and 1)
        assert np.all(y_mixed >= 0) and np.all(y_mixed <= 1)

    def test_mixup_with_onehot_labels(self):
        """Test mixup with one-hot encoded labels."""
        X = np.random.randn(20, 8, 200)
        y = np.eye(3)[np.random.randint(0, 3, 20)]  # 3 classes

        X_mixed, y_mixed = mixup(X, y, alpha=0.3, random_state=42)

        assert X_mixed.shape == X.shape
        assert y_mixed.shape == y.shape

    def test_mixup_alpha_zero(self):
        """Test that alpha=0 means no mixing."""
        X = np.random.randn(10, 8, 100)
        y = np.random.randint(0, 2, 10)

        X_mixed, y_mixed = mixup(X, y, alpha=0.0, random_state=42)

        # With alpha=0, output should be same as input (no mixing)
        np.testing.assert_array_almost_equal(X_mixed, X)


class TestAugmentBatch:
    """Test batch augmentation function."""

    def test_augment_batch_default(self):
        """Test batch augmentation with default settings."""
        X = np.random.randn(20, 8, 200)
        y = np.random.randint(0, 2, 20)

        X_aug, y_aug = augment_batch(X, y, random_state=42)

        assert X_aug.shape == X.shape
        assert not np.allclose(X, X_aug)

    def test_augment_batch_custom_augmentations(self):
        """Test batch augmentation with custom augmentation list."""
        X = np.random.randn(20, 8, 200)

        X_aug, _ = augment_batch(
            X,
            augmentations=["time_shift", "gaussian_noise"],
            random_state=42,
        )

        assert X_aug.shape == X.shape

    def test_augment_batch_with_mixup(self):
        """Test batch augmentation including mixup."""
        X = np.random.randn(20, 8, 200)
        y = np.random.randint(0, 2, 20)

        X_aug, y_aug = augment_batch(
            X, y, augmentations=["time_shift", "mixup"], random_state=42
        )

        assert X_aug.shape == X.shape
        assert y_aug.shape == (20, 2)  # One-hot encoded

    def test_augment_batch_no_labels(self):
        """Test batch augmentation without labels."""
        X = np.random.randn(20, 8, 200)

        X_aug, y_aug = augment_batch(
            X, augmentations=["amplitude_scale", "gaussian_noise"], random_state=42
        )

        assert X_aug.shape == X.shape
        assert y_aug is None


class TestAugmentationPipeline:
    """Test AugmentationPipeline class."""

    def test_pipeline_transform(self):
        """Test pipeline transform method."""
        pipeline = AugmentationPipeline(
            [
                ("time_shift", {"max_shift": 10}),
                ("gaussian_noise", {"noise_level": 0.05}),
            ],
            random_state=42,
        )

        X = np.random.randn(20, 8, 200)
        X_aug, _ = pipeline.transform(X)

        assert X_aug.shape == X.shape
        assert not np.allclose(X, X_aug)

    def test_pipeline_fit_transform(self):
        """Test pipeline fit_transform method."""
        pipeline = AugmentationPipeline(
            [
                ("amplitude_scale", {"scale_range": (0.8, 1.2)}),
                ("smooth", {"sigma": 1.0}),
            ],
            random_state=42,
        )

        X = np.random.randn(20, 8, 200)
        y = np.random.randint(0, 2, 20)

        pipeline.fit(X, y)
        X_aug, y_aug = pipeline.fit_transform(X, y)

        assert X_aug.shape == X.shape

    def test_pipeline_with_mixup(self):
        """Test pipeline including mixup."""
        pipeline = AugmentationPipeline(
            [
                ("time_shift", {"max_shift": 15}),
                ("mixup", {"alpha": 0.2}),
            ],
            random_state=42,
        )

        X = np.random.randn(20, 8, 200)
        y = np.random.randint(0, 3, 20)

        X_aug, y_aug = pipeline.transform(X, y)

        assert X_aug.shape == X.shape
        assert y_aug.shape == (20, 3)  # One-hot for 3 classes


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_input_shape(self):
        """Test error handling for invalid input shapes."""
        X_1d = np.random.randn(100)  # 1D array

        with pytest.raises(ValueError):
            time_shift(X_1d)

    def test_augmentation_preserves_dtype(self):
        """Test that augmentations preserve data type."""
        X = np.random.randn(10, 8, 100).astype(np.float32)
        X_aug = time_shift(X, random_state=42)

        assert X_aug.dtype == X.dtype

    def test_zero_noise_level(self):
        """Test that zero noise level returns unchanged data."""
        X = np.random.randn(10, 8, 100)
        X_aug = gaussian_noise(X, noise_level=0.0, random_state=42)

        np.testing.assert_array_equal(X, X_aug)

    def test_unknown_augmentation_in_batch(self):
        """Test that unknown augmentations are skipped."""
        X = np.random.randn(10, 8, 100)

        # Should not raise error, just skip unknown augmentation
        X_aug, _ = augment_batch(
            X, augmentations=["time_shift", "unknown_aug"], random_state=42
        )

        assert X_aug.shape == X.shape
