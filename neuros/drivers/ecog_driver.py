"""
ECoG driver for simulated electrocorticography signals.

The :class:`ECoGDriver` generates synthetic electrocorticography (ECoG)
samples at a configurable sampling rate and channel count.  ECoG
recordings typically exhibit higher spatial resolution and broader
bandwidth than scalp EEG.  To approximate this, the driver produces
multi‑frequency sinusoids plus noise at a higher sampling rate.

This driver is useful for testing neurOS pipelines with data that
resembles high‑density intracranial recordings.  In a production
setting, a BrainFlow driver or other hardware interface would be used
instead.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from .base_driver import BaseDriver


class ECoGDriver(BaseDriver):
    """Simulated electrocorticography device.

    Parameters
    ----------
    sampling_rate : float, optional
        Sampling frequency in Hz.  ECoG systems commonly sample at
        kilohertz rates.  Defaults to 1000.0.
    channels : int, optional
        Number of cortical electrodes to simulate.  Defaults to 32.

    Examples
    --------
    >>> driver = ECoGDriver(sampling_rate=2000.0, channels=64)
    >>> await driver.start()
    >>> async for ts, data in driver:
    ...     process(data)
    """

    def __init__(self, sampling_rate: float = 1000.0, channels: int = 32) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=channels)
        # generate channel‑specific frequencies spanning beta and gamma bands
        # Use a wider range to reflect high‑frequency ECoG content
        self._freqs = np.linspace(15.0, 120.0, channels)
        self._t = 0.0

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        while self._running:
            timestamp = time.time()
            # time vector for each channel
            t_vec = self._t + np.arange(self.channels) / self.sampling_rate
            # simulate multi‑frequency sinusoidal activity
            signal = np.sin(2 * np.pi * self._freqs * t_vec)
            noise = 0.05 * np.random.randn(self.channels)
            data = signal + noise
            self._t += period
            yield timestamp, data.astype(np.float32)
            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break