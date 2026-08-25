"""Runtime Transform adapters exposed through the neurOS plugin registry."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from neuros.contracts import SignalFrame, WindowSpec
from neuros.processing.feature_extraction import BandPowerExtractor
from neuros.processing.filters import BandpassFilter, SmoothingFilter
from neuros.processing.windowing import SlidingWindowTransform


def _map(item: Any, data: np.ndarray, *, representation: str) -> Any:
    if not isinstance(item, SignalFrame):
        return data
    return replace(
        item,
        data=np.asarray(data),
        metadata={**dict(item.metadata), "representation": representation},
    )


class BandpassTransform:
    def __init__(self, lowcut: float, highcut: float, fs: float, order: int = 4) -> None:
        self.filter = BandpassFilter(lowcut=lowcut, highcut=highcut, fs=fs, order=order)

    def transform(self, item: Any) -> Any:
        data = np.asarray(item.data if isinstance(item, SignalFrame) else item)
        return _map(item, self.filter.apply(data), representation="bandpass")


class SmoothingTransform:
    def __init__(self, window_size: int = 5) -> None:
        self.filter = SmoothingFilter(window_size=window_size)

    def transform(self, item: Any) -> Any:
        data = np.asarray(item.data if isinstance(item, SignalFrame) else item)
        return _map(item, self.filter.apply(data), representation="smoothed")


class BandPowerTransform:
    def __init__(self, fs: float, bands: dict[str, tuple[float, float]] | None = None) -> None:
        self.extractor = BandPowerExtractor(fs=fs, bands=bands)

    def transform(self, item: Any) -> Any:
        data = np.asarray(item.data if isinstance(item, SignalFrame) else item)
        return _map(item, self.extractor.extract(data), representation="bandpower")


class NeuralWindowTransform(SlidingWindowTransform):
    """Configuration-friendly adapter for canonical decoder windows."""

    def __init__(
        self,
        window_samples: int,
        stride_samples: int,
        discontinuity: str = "error",
    ) -> None:
        super().__init__(
            WindowSpec(window_samples=window_samples, stride_samples=stride_samples),
            discontinuity=discontinuity,
        )
