"""
Signal processing and feature extraction modules.

These functions and classes operate on raw neural data to clean it and
extract informative features for downstream models.  They are designed to be
small and composable so that they can be arranged flexibly in a pipeline.
"""

from neuros.processing.filters import BandpassFilter, SmoothingFilter  # noqa: F401
from neuros.processing.feature_extraction import (
    BandPowerExtractor,
    HeartRateExtractor,
    SkinConductanceExtractor,
    RespirationExtractor,
    HormoneExtractor,
    AudioFeatureExtractor,
)  # noqa: F401
from neuros.processing.adaptation import AdaptiveThreshold  # noqa: F401
