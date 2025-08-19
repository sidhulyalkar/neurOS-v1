"""
Quality monitor for neurOS.

This module defines a simple class for monitoring data quality during
pipeline runs.  The monitor accumulates the mean and standard
deviation of incoming samples over time.  At the end of a run, the
mean of the sample means and the mean of the sample standard
deviations are returned as quality metrics.  These metrics
approximate overall signal amplitude and variability.

The monitor can handle 1D arrays (e.g. EEG or EMG channels), 2D arrays
(e.g. video or calcium imaging frames) and scalar values.  It uses
``numpy`` to compute statistics.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np


class QualityMonitor:
    """Accumulate basic quality metrics for streaming data."""

    def __init__(self) -> None:
        self.sum_mean: float = 0.0
        self.sum_std: float = 0.0
        self.count: int = 0

    def update(self, sample: Iterable[float] | np.ndarray) -> None:
        """Update the monitor with a new sample.

        Parameters
        ----------
        sample : array‑like
            The raw data sample.  Can be a NumPy array of any shape or
            an iterable of numbers.  The monitor flattens the sample
            and computes its mean and standard deviation.
        """
        arr = np.asarray(sample, dtype=np.float32)
        # flatten to one dimension for statistics
        flat = arr.ravel()
        self.sum_mean += float(flat.mean())
        self.sum_std += float(flat.std())
        self.count += 1

    def result(self) -> dict:
        """Return averaged quality metrics.

        Returns
        -------
        dict
            Dictionary with keys ``quality_mean`` and ``quality_std``.
            If no samples were observed, both values are zero.
        """
        if self.count == 0:
            return {"quality_mean": 0.0, "quality_std": 0.0}
        return {
            "quality_mean": self.sum_mean / self.count,
            "quality_std": self.sum_std / self.count,
        }