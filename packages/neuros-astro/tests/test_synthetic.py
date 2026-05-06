"""Test synthetic data generation."""

import numpy as np
from neuros_astro.io.synthetic import (
    generate_synthetic_astro_traces,
    generate_synthetic_astro_movie,
)


def test_synthetic_traces_shape():
    """Test that synthetic traces have correct shape."""
    traces, events = generate_synthetic_astro_traces(
        n_regions=5, duration_s=10.0, frame_rate_hz=10.0, seed=42
    )

    expected_frames = int(10.0 * 10.0)
    assert traces.shape == (5, expected_frames)
    assert len(events) > 0


def test_synthetic_traces_deterministic():
    """Test that same seed produces same output."""
    traces1, _ = generate_synthetic_astro_traces(
        n_regions=3, duration_s=10.0, frame_rate_hz=10.0, seed=42
    )
    traces2, _ = generate_synthetic_astro_traces(
        n_regions=3, duration_s=10.0, frame_rate_hz=10.0, seed=42
    )

    np.testing.assert_array_equal(traces1, traces2)


def test_synthetic_movie_shape():
    """Test that synthetic movie has correct shape."""
    movie, events = generate_synthetic_astro_movie(
        duration_s=5.0, frame_rate_hz=10.0, height=64, width=64, seed=42
    )

    expected_frames = int(5.0 * 10.0)
    assert movie.shape == (expected_frames, 64, 64)
    assert len(events) > 0


def test_synthetic_movie_nonnegative():
    """Test that movie values are non-negative."""
    movie, _ = generate_synthetic_astro_movie(
        duration_s=5.0, frame_rate_hz=10.0, height=32, width=32, seed=42
    )

    assert np.all(movie >= 0)


def test_synthetic_movie_deterministic():
    """Test that same seed produces same movie."""
    movie1, _ = generate_synthetic_astro_movie(
        duration_s=5.0, frame_rate_hz=10.0, height=32, width=32, seed=42
    )
    movie2, _ = generate_synthetic_astro_movie(
        duration_s=5.0, frame_rate_hz=10.0, height=32, width=32, seed=42
    )

    np.testing.assert_array_equal(movie1, movie2)
