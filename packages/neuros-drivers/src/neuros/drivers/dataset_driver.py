"""Dataset replay driver for neurOS."""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class DatasetDriver(BaseDriver):
    """Stream a stored tabular dataset through the legacy driver interface.

    ``sampling_rate <= 0`` retains the historical meaning of "emit as fast as
    possible". The canonical stream descriptor still carries a positive nominal
    rate; replay pacing is represented separately by ``replay_rate_hz``.
    """

    def __init__(
        self,
        dataset_name: str = "iris",
        sampling_rate: float = 1.0,
        *,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        if data is None:
            try:
                from sklearn import datasets  # type: ignore
            except ImportError as exc:
                raise ImportError(
                    "scikit-learn must be installed to load named DatasetDriver datasets"
                ) from exc
            dataset_name = dataset_name.lower()
            if dataset_name == "iris":
                ds = datasets.load_iris()
            elif dataset_name == "digits":
                ds = datasets.load_digits()
            elif dataset_name == "wine":
                ds = datasets.load_wine()
            elif dataset_name in ("breast_cancer", "cancer"):
                ds = datasets.load_breast_cancer()
            else:
                raise ValueError(
                    f"Unknown dataset_name '{dataset_name}'. Supported: iris, digits, wine, breast_cancer"
                )
            X = ds.data
            y = ds.target
        else:
            X = np.asarray(data)
            y = np.asarray(labels) if labels is not None else np.zeros(len(X), dtype=int)

        self.data = np.asarray(X)
        self.labels = np.asarray(y)
        if self.data.ndim == 1:
            self.data = self.data[:, None]
        if len(self.labels) != len(self.data):
            raise ValueError("labels length must match data length")

        self.replay_rate_hz: float | None = sampling_rate if sampling_rate > 0 else None
        nominal_rate_hz = self.replay_rate_hz or 1.0
        super().__init__(
            sampling_rate=nominal_rate_hz,
            channels=self.data.shape[1],
            modality="dataset",
        )
        self.total_samples = self.data.shape[0]
        self._index = 0

    def get_duration(self) -> float:
        if self.replay_rate_hz is None:
            return 0.0
        return float(self.total_samples) / self.replay_rate_hz

    async def _stream(self) -> asyncio.AsyncIterator[Tuple[float, np.ndarray]]:
        try:
            while self._index < self.total_samples:
                data = np.array(self.data[self._index], dtype=float)
                timestamp = time.time()
                self._index += 1
                yield timestamp, data
                if not self._running:
                    break
                if self.replay_rate_hz is not None:
                    await asyncio.sleep(1.0 / self.replay_rate_hz)
                else:
                    await asyncio.sleep(0)
        finally:
            self._running = False
