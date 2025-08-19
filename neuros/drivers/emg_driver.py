"""
EMG driver for simulated electromyography signals.

The :class:`EMGDriver` generates synthetic electromyography (EMG)
signals at a moderate sampling rate.  EMG recordings reflect muscle
activity and are characterized by bursts of broadband noise.  This
driver produces random noise with occasional higher amplitude bursts
to approximate muscle contractions.

Use this driver to develop and test neurOS pipelines for muscle signal
processing.  In production, a BrainFlow driver with EMG hardware
should be used instead.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from .base_driver import BaseDriver


class EMGDriver(BaseDriver):
    """Simulated electromyography device.

    Parameters
    ----------
    sampling_rate : float, optional
        Sampling frequency in Hz.  EMG systems often sample around
        several hundred Hz.  Defaults to 500.0.
    channels : int, optional
        Number of muscle channels to simulate.  Defaults to 8.
    """

    def __init__(self, sampling_rate: float = 500.0, channels: int = 8) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=channels)

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        while self._running:
            timestamp = time.time()
            # generate broadband noise with occasional bursts
            noise = 0.1 * np.random.randn(self.channels)
            bursts = np.random.rand(self.channels) < 0.05  # 5% chance of burst per channel
            burst_amp = 0.5 * np.random.randn(self.channels) * bursts.astype(np.float32)
            data = noise + burst_amp
            yield timestamp, data.astype(np.float32)
            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break