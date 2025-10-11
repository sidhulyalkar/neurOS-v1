"""
Tests for time series alignment and DTW utilities.
"""

import numpy as np
import pytest

from neuros.alignment import (
    piecewise_linear_warp,
    align_trials,
    dynamic_time_warping_distance,
    apply_warp_to_new_data,
    estimate_template,
    WarpResult,
)


class TestPiecewiseLinearWarp:
    """Test piecewise linear warping."""

    def test_identity_warp(self):
        """Test that zero warp parameters give identity transform."""
        X = np.random.randn(100, 5)
        warp_params = np.zeros(5)

        X_warped = piecewise_linear_warp(X, warp_params, n_knots=5)

        # Should be very close to original (up to interpolation error)
        np.testing.assert_array_almost_equal(X, X_warped, decimal=1)

    def test_warp_1d_signal(self):
        """Test warping a 1D signal."""
        X = np.sin(np.linspace(0, 2*np.pi, 100))
        warp_params = np.array([0, 5, 10, 5, 0])

        X_warped = piecewise_linear_warp(X, warp_params, n_knots=5)

        assert X_warped.shape == X.shape

    def test_warp_preserves_shape(self):
        """Test that warping preserves data shape."""
        X = np.random.randn(200, 10)
        warp_params = np.random.randn(5) * 2

        X_warped = piecewise_linear_warp(X, warp_params, n_knots=5)

        assert X_warped.shape == X.shape

    def test_warp_with_different_knots(self):
        """Test warping with different numbers of knots."""
        X = np.random.randn(100, 5)

        # Test with 3 knots
        warp_params_3 = np.random.randn(3)
        X_warped_3 = piecewise_linear_warp(X, warp_params_3, n_knots=3)
        assert X_warped_3.shape == X.shape

        # Test with 10 knots
        warp_params_10 = np.random.randn(10)
        X_warped_10 = piecewise_linear_warp(X, warp_params_10, n_knots=10)
        assert X_warped_10.shape == X.shape


class TestAlignTrials:
    """Test trial alignment functionality."""

    def test_align_two_trials(self):
        """Test aligning two simple trials."""
        # Create two signals with temporal offset
        t = np.linspace(0, 2*np.pi, 100)
        trial1 = np.sin(t)[:, np.newaxis]
        trial2 = np.sin(t + 0.3)[:, np.newaxis]  # Shifted

        trials = [trial1, trial2]

        result = align_trials(trials, n_knots=5, max_iter=50)

        assert isinstance(result, WarpResult)
        assert result.warped_data.shape == (2, 100, 1)
        assert len(result.warp_functions) == 2
        assert result.alignment_cost is not None

    def test_align_multiple_trials(self):
        """Test aligning multiple trials."""
        trials = []
        for i in range(5):
            t = np.linspace(0, 2*np.pi, 100)
            # Add random temporal jitter
            trial = np.sin(t + np.random.randn() * 0.2)[:, np.newaxis]
            trials.append(trial)

        result = align_trials(trials, n_knots=5, max_iter=30)

        assert result.warped_data.shape == (5, 100, 1)
        assert len(result.warp_functions) == 5

    def test_align_improves_alignment(self):
        """Test that alignment actually reduces variance."""
        # Create trials with systematic temporal shifts
        trials = []
        t = np.linspace(0, 2*np.pi, 100)
        for shift in np.linspace(0, 0.5, 10):
            trial = np.sin(t + shift)[:, np.newaxis]
            trials.append(trial)

        # Measure variance before alignment
        trials_stack = np.stack(trials, axis=0)
        variance_before = np.mean(np.var(trials_stack, axis=0))

        # Align
        result = align_trials(trials, n_knots=7, max_iter=50)

        # Measure variance after alignment
        variance_after = np.mean(np.var(result.warped_data, axis=0))

        # Alignment should reduce variance
        assert variance_after < variance_before

    def test_align_with_template(self):
        """Test alignment to a fixed template."""
        # Create template
        t = np.linspace(0, 2*np.pi, 100)
        template = np.sin(t)[:, np.newaxis]

        # Create trials with shifts
        trials = []
        for shift in [0.2, -0.3, 0.1]:
            trial = np.sin(t + shift)[:, np.newaxis]
            trials.append(trial)

        result = align_trials(trials, n_knots=5, template=template, max_iter=30)

        assert result.warped_data.shape == (3, 100, 1)

    def test_align_knot_points(self):
        """Test that knot points are correctly stored."""
        trials = [np.random.randn(100, 5) for _ in range(3)]

        result = align_trials(trials, n_knots=5)

        assert result.knot_points is not None
        assert len(result.knot_points) == 5


class TestDynamicTimeWarping:
    """Test DTW distance computation."""

    def test_dtw_identical_sequences(self):
        """Test DTW distance for identical sequences."""
        x = np.random.randn(50)
        y = x.copy()

        distance, path = dynamic_time_warping_distance(x, y)

        # Distance should be very small (near zero)
        assert distance < 0.01

    def test_dtw_different_lengths(self):
        """Test DTW with sequences of different lengths."""
        x = np.sin(np.linspace(0, 2*np.pi, 100))
        y = np.sin(np.linspace(0, 2*np.pi, 120))

        distance, path = dynamic_time_warping_distance(x, y)

        # Distance should be finite and path should exist
        assert np.isfinite(distance)
        assert path.shape[1] == 2
        assert len(path) > 0

    def test_dtw_path_valid(self):
        """Test that DTW path is valid."""
        x = np.random.randn(30)
        y = np.random.randn(40)

        distance, path = dynamic_time_warping_distance(x, y)

        # Path should start near (0, 0) and end near (len(x)-1, len(y)-1)
        assert path[0, 0] == 0
        assert path[0, 1] == 0
        assert path[-1, 0] == len(x) - 1
        assert path[-1, 1] == len(y) - 1

    def test_dtw_multidimensional(self):
        """Test DTW with multi-dimensional features."""
        x = np.random.randn(50, 3)
        y = np.random.randn(60, 3)

        distance, path = dynamic_time_warping_distance(x, y)

        assert np.isfinite(distance)
        assert path.shape[1] == 2

    def test_dtw_similar_sequences_small_distance(self):
        """Test that similar sequences have small DTW distance."""
        t = np.linspace(0, 2*np.pi, 100)
        x = np.sin(t)
        y = np.sin(t) + np.random.randn(100) * 0.1  # Add small noise

        distance, path = dynamic_time_warping_distance(x, y)

        # Distance should be relatively small
        assert distance < 5.0  # Somewhat arbitrary threshold


class TestApplyWarp:
    """Test applying learned warps to new data."""

    def test_apply_warp_to_new_data(self):
        """Test applying a warp function to new data."""
        # Create training trials and learn alignment
        trials = []
        for i in range(3):
            t = np.linspace(0, 2*np.pi, 100)
            trial = np.sin(t + np.random.randn() * 0.1)[:, np.newaxis]
            trials.append(trial)

        result = align_trials(trials, n_knots=5, max_iter=30)

        # Apply learned warp to new data
        new_data = np.random.randn(100, 1)
        warped_new = apply_warp_to_new_data(new_data, result.warp_functions[0])

        assert warped_new.shape == new_data.shape

    def test_apply_warp_1d(self):
        """Test applying warp to 1D data."""
        trials = [np.random.randn(100, 1) for _ in range(3)]
        result = align_trials(trials, n_knots=5, max_iter=20)

        new_data = np.random.randn(100)
        warped_new = apply_warp_to_new_data(new_data, result.warp_functions[0])

        assert warped_new.shape == new_data.shape


class TestEstimateTemplate:
    """Test template estimation."""

    def test_estimate_template_mean(self):
        """Test mean template estimation."""
        trials = [np.random.randn(100, 5) for _ in range(10)]

        template = estimate_template(trials, method='mean')

        assert template.shape == (100, 5)

        # Should be close to manual mean computation
        manual_mean = np.mean(np.stack(trials, axis=0), axis=0)
        np.testing.assert_array_almost_equal(template, manual_mean)

    def test_estimate_template_median(self):
        """Test median template estimation."""
        trials = [np.random.randn(100, 5) for _ in range(10)]

        template = estimate_template(trials, method='median')

        assert template.shape == (100, 5)

    def test_estimate_template_pca(self):
        """Test PCA template estimation."""
        trials = [np.random.randn(100, 5) for _ in range(10)]

        template = estimate_template(trials, method='pca')

        assert template.shape == (100, 5)

    def test_estimate_template_invalid_method(self):
        """Test that invalid method raises error."""
        trials = [np.random.randn(100, 5) for _ in range(3)]

        with pytest.raises(ValueError):
            estimate_template(trials, method='invalid_method')


class TestWarpResult:
    """Test WarpResult dataclass."""

    def test_warp_result_creation(self):
        """Test creating WarpResult."""
        warped_data = np.random.randn(5, 100, 3)
        warp_functions = [lambda x: x for _ in range(5)]
        knot_points = np.linspace(0, 99, 5)

        result = WarpResult(
            warped_data=warped_data,
            warp_functions=warp_functions,
            knot_points=knot_points,
            alignment_cost=10.5,
        )

        assert result.warped_data.shape == (5, 100, 3)
        assert len(result.warp_functions) == 5
        assert result.alignment_cost == 10.5


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_align_single_trial(self):
        """Test aligning a single trial (trivial case)."""
        trials = [np.random.randn(100, 5)]

        result = align_trials(trials, n_knots=5, max_iter=10)

        # Should still work, just identity alignment
        assert result.warped_data.shape == (1, 100, 5)

    def test_warp_short_sequence(self):
        """Test warping very short sequences."""
        X = np.random.randn(10, 2)
        warp_params = np.zeros(3)

        X_warped = piecewise_linear_warp(X, warp_params, n_knots=3)

        assert X_warped.shape == X.shape

    def test_dtw_single_point(self):
        """Test DTW with single time point."""
        x = np.array([1.0])
        y = np.array([1.5])

        distance, path = dynamic_time_warping_distance(x, y)

        assert np.isfinite(distance)
        assert len(path) == 1

    def test_align_trials_few_iterations(self):
        """Test alignment with very few iterations."""
        trials = [np.random.randn(50, 3) for _ in range(3)]

        result = align_trials(trials, n_knots=3, max_iter=5)

        # Should still produce valid result
        assert result.warped_data.shape == (3, 50, 3)
