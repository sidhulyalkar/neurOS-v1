"""
Hormone level driver for neurOS.

This driver produces a simulated hormone or biochemical signal.  The
signal evolves slowly over time as a random walk with small noise.
Its low sampling rate reflects the slow dynamics of many hormonal
processes.  The driver yields one value per sample.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class HormoneDriver(BaseDriver):
    """Simulated hormone driver."""

    def __init__(self, sampling_rate: float = 1.0, noise_level: float = 0.001) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=1)
        self.noise_level = noise_level
        self._level = 0.0

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        while self._running:
            timestamp = time.time()
            # random walk
            self._level += np.random.randn() * 0.01
            noise = np.random.randn() * self.noise_level
            data = np.array([self._level + noise], dtype=np.float32)
            yield timestamp, data
            await asyncio.sleep(period)
