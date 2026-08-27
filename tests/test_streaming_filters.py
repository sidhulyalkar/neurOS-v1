from __future__ import annotations

import numpy as np
import pytest

from neuros.processing.filters import BandpassFilter, SmoothingFilter
from neuros.processing.plugin_transforms import BandpassTransform, SmoothingTransform


def _partition(data: np.ndarray, lengths: tuple[int, ...]) -> list[np.ndarray]:
    assert sum(lengths) == data.shape[-1]
    chunks: list[np.ndarray] = []
    start = 0
    for length in lengths:
        chunks.append(data[..., start : start + length])
        start += length
    return chunks


def test_bandpass_is_chunk_partition_invariant_for_one_dimensional_stream():
    rng = np.random.default_rng(7)
    data = rng.normal(size=2048)

    whole_filter = BandpassFilter(1.0, 45.0, fs=250.0, order=4)
    expected = whole_filter.apply(data)

    chunked_filter = BandpassFilter(1.0, 45.0, fs=250.0, order=4)
    actual = np.concatenate(
        [chunked_filter.apply(chunk) for chunk in _partition(data, (1, 7, 64, 511, 3, 1462))]
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_bandpass_is_chunk_partition_invariant_for_multichannel_stream():
    rng = np.random.default_rng(11)
    data = rng.normal(size=(8, 1500))

    whole_filter = BandpassFilter(8.0, 30.0, fs=250.0, order=6)
    expected = whole_filter.apply(data)

    chunked_filter = BandpassFilter(8.0, 30.0, fs=250.0, order=6)
    actual = np.concatenate(
        [chunked_filter.apply(chunk) for chunk in _partition(data, (113, 1, 256, 509, 621))],
        axis=-1,
    )

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_bandpass_reset_restores_fresh_filter_state():
    rng = np.random.default_rng(17)
    prefix = rng.normal(size=(4, 200))
    target = rng.normal(size=(4, 300))
    bandpass = BandpassFilter(2.0, 40.0, fs=250.0)
    bandpass.apply(prefix)
    continued = bandpass.apply(target)

    bandpass.reset()
    reset_output = bandpass.apply(target)
    fresh_output = BandpassFilter(2.0, 40.0, fs=250.0).apply(target)

    assert not np.allclose(continued, fresh_output)
    np.testing.assert_allclose(reset_output, fresh_output, rtol=0.0, atol=0.0)


def test_bandpass_requires_explicit_reset_before_geometry_change():
    bandpass = BandpassFilter(1.0, 30.0, fs=250.0)
    bandpass.apply(np.zeros((2, 20)))
    with pytest.raises(ValueError, match="geometry changed"):
        bandpass.apply(np.zeros((3, 20)))
    bandpass.reset()
    assert bandpass.apply(np.zeros((3, 20))).shape == (3, 20)


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((0.0, 30.0, 250.0, 4), "cutoffs"),
        ((30.0, 10.0, 250.0, 4), "cutoffs"),
        ((1.0, 125.0, 250.0, 4), "Nyquist"),
        ((1.0, 30.0, 0.0, 4), "fs must be positive"),
        ((1.0, 30.0, 250.0, 0), "positive integer"),
        ((1.0, 30.0, 250.0, 4.0), "positive integer"),
    ],
)
def test_bandpass_rejects_invalid_configuration(args, message):
    with pytest.raises(ValueError, match=message):
        BandpassFilter(*args)


def test_smoothing_is_causal_and_chunk_partition_invariant():
    data = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    whole_filter = SmoothingFilter(window_size=3)
    expected = whole_filter.apply(data)

    # No sample before the impulse can depend on that future impulse.
    np.testing.assert_allclose(expected[:3], 0.0, rtol=0.0, atol=0.0)
    np.testing.assert_allclose(expected[3:6], 1.0 / 3.0, rtol=0.0, atol=1e-15)
    assert expected[6] == pytest.approx(0.0)

    chunked_filter = SmoothingFilter(window_size=3)
    actual = np.concatenate(
        [chunked_filter.apply(chunk) for chunk in _partition(data, (1, 2, 1, 3))]
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_smoothing_is_chunk_partition_invariant_for_multichannel_stream():
    rng = np.random.default_rng(23)
    data = rng.normal(size=(5, 777))
    expected = SmoothingFilter(window_size=9).apply(data)

    chunked = SmoothingFilter(window_size=9)
    actual = np.concatenate(
        [chunked.apply(chunk) for chunk in _partition(data, (8, 1, 41, 250, 477))],
        axis=-1,
    )
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)


def test_smoothing_reset_and_geometry_contract():
    smoothing = SmoothingFilter(window_size=5)
    prefix = np.arange(20.0).reshape(2, 10)
    target = np.ones((2, 6))
    smoothing.apply(prefix)
    continued = smoothing.apply(target)

    smoothing.reset()
    reset_output = smoothing.apply(target)
    fresh_output = SmoothingFilter(window_size=5).apply(target)
    assert not np.allclose(continued, fresh_output)
    np.testing.assert_allclose(reset_output, fresh_output, rtol=0.0, atol=0.0)

    with pytest.raises(ValueError, match="geometry changed"):
        smoothing.apply(np.ones((3, 2)))


def test_smoothing_requires_strict_positive_integer_window():
    with pytest.raises(ValueError, match="positive integer"):
        SmoothingFilter(window_size=0)
    with pytest.raises(ValueError, match="positive integer"):
        SmoothingFilter(window_size=3.5)


def test_plugin_transforms_preserve_state_and_expose_epoch_reset():
    data = np.linspace(-1.0, 1.0, 120)

    bandpass = BandpassTransform(1.0, 30.0, fs=250.0)
    chunked = np.concatenate((bandpass.transform(data[:50]), bandpass.transform(data[50:])))
    whole = BandpassTransform(1.0, 30.0, fs=250.0).transform(data)
    np.testing.assert_allclose(chunked, whole, rtol=1e-12, atol=1e-12)
    bandpass.reset()
    np.testing.assert_allclose(bandpass.transform(data), whole, rtol=0.0, atol=0.0)

    smoothing = SmoothingTransform(window_size=5)
    chunked_smooth = np.concatenate(
        (smoothing.transform(data[:17]), smoothing.transform(data[17:91]), smoothing.transform(data[91:]))
    )
    whole_smooth = SmoothingTransform(window_size=5).transform(data)
    np.testing.assert_allclose(chunked_smooth, whole_smooth, rtol=0.0, atol=0.0)
    smoothing.reset()
    np.testing.assert_allclose(smoothing.transform(data), whole_smooth, rtol=0.0, atol=0.0)
