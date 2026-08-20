"""Feature-distribution summaries for domain and source comparison."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _as_samples_features(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
        raise ValueError("features must contain at least one sample and feature")
    if not np.all(np.isfinite(arr)):
        raise ValueError("features must be finite")
    return arr


def summarize_features(
    features: np.ndarray,
    *,
    statistics: tuple[str, ...] = ("mean", "log_std"),
) -> np.ndarray:
    """Convert a sample-by-feature matrix into a comparable domain summary."""
    x = _as_samples_features(features)
    chunks: list[np.ndarray] = []
    for name in statistics:
        if name == "mean":
            value = x.mean(axis=0)
        elif name == "std":
            value = x.std(axis=0)
        elif name == "log_std":
            value = np.log(np.maximum(x.std(axis=0), 1e-8))
        elif name == "median":
            value = np.median(x, axis=0)
        elif name == "iqr":
            value = np.quantile(x, 0.75, axis=0) - np.quantile(x, 0.25, axis=0)
        elif name == "q25":
            value = np.quantile(x, 0.25, axis=0)
        elif name == "q75":
            value = np.quantile(x, 0.75, axis=0)
        else:
            raise ValueError(f"unsupported statistic: {name}")
        chunks.append(np.asarray(value, dtype=float))
    if not chunks:
        raise ValueError("statistics cannot be empty")
    return np.concatenate(chunks)


@dataclass
class RunningFeatureSummary:
    """Welford streaming estimator for mean/variance domain summaries."""

    n_features: int
    count: int = 0

    def __post_init__(self) -> None:
        if self.n_features <= 0:
            raise ValueError("n_features must be positive")
        self.mean_ = np.zeros(self.n_features, dtype=float)
        self.m2_ = np.zeros(self.n_features, dtype=float)

    def update(self, batch: np.ndarray) -> "RunningFeatureSummary":
        x = _as_samples_features(batch)
        if x.shape[1] != self.n_features:
            raise ValueError(
                f"batch has {x.shape[1]} features; expected {self.n_features}"
            )
        batch_n = x.shape[0]
        batch_mean = x.mean(axis=0)
        batch_m2 = np.sum((x - batch_mean) ** 2, axis=0)
        if self.count == 0:
            self.mean_ = batch_mean
            self.m2_ = batch_m2
            self.count = batch_n
            return self

        delta = batch_mean - self.mean_
        total = self.count + batch_n
        self.mean_ = self.mean_ + delta * (batch_n / total)
        self.m2_ = self.m2_ + batch_m2 + delta**2 * (self.count * batch_n / total)
        self.count = total
        return self

    @property
    def variance(self) -> np.ndarray:
        if self.count == 0:
            raise RuntimeError("no observations have been added")
        return self.m2_ / self.count

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(np.maximum(self.variance, 0.0))

    def vector(self, *, log_std: bool = True) -> np.ndarray:
        if self.count == 0:
            raise RuntimeError("no observations have been added")
        spread = np.log(np.maximum(self.std, 1e-8)) if log_std else self.std
        return np.concatenate([self.mean_.copy(), spread])
