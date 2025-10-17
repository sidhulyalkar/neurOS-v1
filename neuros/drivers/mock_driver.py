"""
Mock driver for generating synthetic neural data.

The :class:`MockDriver` yields random Gaussian noise that approximates
band‑limited neural signals.  It is useful for testing pipelines without
physical hardware.  The sampling rate and number of channels are
configurable.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class MockDriver(BaseDriver):
    """Simulated neural device that produces random data."""

    def __init__(self, sampling_rate: float = 250.0, channels: int = 8) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=channels)
        # generate a fixed sine wave to simulate a basic oscillatory pattern
        self._t = 0.0
        self._freqs = np.linspace(8.0, 12.0, channels)  # simulate alpha band

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        while self._running:
            timestamp = time.time()
            # create sinusoidal waves plus random noise
            t_vec = self._t + np.arange(self.channels) / self.sampling_rate
            signal = np.sin(2 * np.pi * self._freqs * t_vec)
            noise = 0.05 * np.random.randn(self.channels)
            data = signal + noise
            self._t += period
            yield timestamp, data.astype(np.float32)
            await asyncio.sleep(period)
