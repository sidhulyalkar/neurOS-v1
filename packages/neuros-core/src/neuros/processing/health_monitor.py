"""Quality monitoring for neurOS runtime streams."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

import numpy as np

from neuros.contracts import SignalFrame


class QualityMonitor:
    """Accumulate lightweight amplitude/variability quality metrics.

    ``update`` accepts raw arrays for backwards compatibility and the structured
    monitor event emitted by :class:`RuntimeExecutor`.  SignalFrame inputs are
    reduced over their data payload rather than their metadata.
    """

    def __init__(self) -> None:
        self.sum_mean: float = 0.0
        self.sum_std: float = 0.0
        self.count: int = 0

    def update(self, sample: Iterable[float] | np.ndarray | Mapping[str, Any] | SignalFrame) -> None:
        if isinstance(sample, Mapping) and "item" in sample:
            sample = sample["item"]
        if isinstance(sample, SignalFrame):
            sample = sample.data
        # Decoder outputs and other non-numeric runtime events are not raw signal
        # quality observations; ignore them rather than fabricating a statistic.
        try:
            arr = np.asarray(sample, dtype=np.float32)
        except (TypeError, ValueError):
            return
        if arr.size == 0:
            return
        flat = arr.ravel()
        self.sum_mean += float(flat.mean())
        self.sum_std += float(flat.std())
        self.count += 1

    def result(self) -> dict[str, float]:
        if self.count == 0:
            return {"quality_mean": 0.0, "quality_std": 0.0}
        return {
            "quality_mean": self.sum_mean / self.count,
            "quality_std": self.sum_std / self.count,
        }
