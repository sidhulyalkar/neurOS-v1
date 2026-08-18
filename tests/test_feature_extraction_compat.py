import numpy as np

from neuros.processing.feature_extraction import BandPowerExtractor


def test_bandpower_extraction_is_numpy_2_compatible():
    fs = 250.0
    t = np.arange(500) / fs
    signal = np.sin(2 * np.pi * 10.0 * t)
    features = BandPowerExtractor(fs=fs).extract(signal[np.newaxis, :])
    assert features.shape == (5,)
    assert np.all(np.isfinite(features))
    assert features[2] > features[0]
