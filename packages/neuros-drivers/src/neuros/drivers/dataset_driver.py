"""
Dataset driver for neurOS.

This driver allows neurOS to stream samples from a stored dataset as if they
were coming from a live device.  It is primarily intended for offline
reprocessing and benchmarking of existing datasets.  When instantiated,
the driver loads a dataset using scikit‑learn loaders (currently supports
`iris`, `digits`, `wine`, `breast_cancer`) or accepts pre‑provided data and
labels.  It then yields samples at the specified sampling rate.  Unlike
other drivers, the dataset driver stops streaming after all samples have
been emitted; the total number of samples and approximate run duration
can be obtained via :meth:`get_duration`.

Example
-------
>>> from neuros.drivers.dataset_driver import DatasetDriver
>>> driver = DatasetDriver(dataset_name="iris", sampling_rate=10.0)
>>> # streaming yields 10 samples per second until the 150 iris samples are consumed
>>> async for ts, data in driver:
...     print(ts, data)
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional, Tuple

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class DatasetDriver(BaseDriver):
    """Driver that streams data from a preloaded dataset.

    Parameters
    ----------
    dataset_name : str, optional
        Name of the dataset to load.  Supported values include
        ``"iris"``, ``"digits"``, ``"wine"`` and ``"breast_cancer"``.
        Defaults to ``"iris"``.
    sampling_rate : float, optional
        Rate at which samples are yielded (in Hz).  Defaults to 1.0 Hz.  If
        set to 0 or a negative value, the driver will yield samples as
        quickly as possible (no waiting between samples).
    data : np.ndarray, optional
        Preloaded feature matrix of shape (n_samples, n_features).  If
        provided, it overrides ``dataset_name``.  Labels may still be
        provided via ``labels``.
    labels : np.ndarray, optional
        Optional label vector of length ``n_samples``.  If both
        ``data`` and ``labels`` are provided, they are used directly.
    """

    def __init__(
        self,
        dataset_name: str = "iris",
        sampling_rate: float = 1.0,
        *,
        data: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        # load dataset if no data provided
        if data is None:
            try:
                from sklearn import datasets  # type: ignore
            except ImportError as e:
                raise ImportError(
                    "scikit‑learn must be installed to use DatasetDriver"
                ) from e
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
            # if labels not provided, set to zeros
            y = np.asarray(labels) if labels is not None else np.zeros(len(X), dtype=int)
        self.data: np.ndarray = X
        self.labels: np.ndarray = y
        # ensure 2D
        if self.data.ndim == 1:
            self.data = self.data[:, None]
        # set channels equal to number of features
        channels = self.data.shape[1]
        # clip sampling_rate to non‑negative
        self.sampling_rate = sampling_rate if sampling_rate > 0 else 0.0
        super().__init__(sampling_rate=self.sampling_rate, channels=channels)
        # compute approximate duration (seconds)
        self.total_samples: int = self.data.shape[0]
        self._index: int = 0

    def get_duration(self) -> float:
        """Return the approximate duration of the dataset stream in seconds.

        The duration is computed as ``total_samples / sampling_rate``.  If
        ``sampling_rate == 0``, the duration is returned as 0.
        """
        if self.sampling_rate > 0:
            return float(self.total_samples) / self.sampling_rate
        return 0.0

    async def _stream(self) -> asyncio.AsyncIterator[Tuple[float, np.ndarray]]:
        """Asynchronously iterate over dataset samples.

        Yields timestamped samples until the end of the dataset is reached.
        If ``sampling_rate > 0``, waits ``1/sampling_rate`` seconds between
        samples; otherwise yields samples as fast as possible.
        """
        try:
            # produce each sample
            while self._index < self.total_samples:
                sample = self.data[self._index]
                # copy data to avoid inadvertent modifications downstream
                data = np.array(sample, dtype=float)
                timestamp = time.time()
                self._index += 1
                yield timestamp, data
                if not self._running:
                    break
                # wait based on sampling rate
                if self.sampling_rate > 0:
                    await asyncio.sleep(1.0 / self.sampling_rate)
                else:
                    # yield control to event loop
                    await asyncio.sleep(0)
        finally:
            # automatically stop after streaming all samples
            self._running = False