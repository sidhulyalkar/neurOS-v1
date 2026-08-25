"""Signal processing and feature extraction modules."""

from neuros.processing.adaptation import AdaptiveThreshold  # noqa: F401
from neuros.processing.feature_extraction import (  # noqa: F401
    AudioFeatureExtractor,
    BandPowerExtractor,
    HeartRateExtractor,
    HormoneExtractor,
    RespirationExtractor,
    SkinConductanceExtractor,
)
from neuros.processing.filters import BandpassFilter, SmoothingFilter  # noqa: F401
from neuros.processing.operators import FeatureTransform  # noqa: F401
from neuros.processing.windowing import (  # noqa: F401
    DiscontinuityPolicy,
    SlidingWindowTransform,
)

__all__ = [
    "AdaptiveThreshold",
    "AudioFeatureExtractor",
    "BandPowerExtractor",
    "BandpassFilter",
    "DiscontinuityPolicy",
    "FeatureTransform",
    "HeartRateExtractor",
    "HormoneExtractor",
    "RespirationExtractor",
    "SkinConductanceExtractor",
    "SlidingWindowTransform",
    "SmoothingFilter",
]
