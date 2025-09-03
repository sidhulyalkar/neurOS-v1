"""
Respiration driver for neurOS.

This driver produces a simulated respiration (breathing) waveform as a
sinusoid with noise.  The sampling rate and breath frequency can be
configured.  The signal models inhalation and exhalation cycles and is
useful for testing multi‑modal experiments involving cardiorespiratory
data.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from .base_driver import BaseDriver


class RespirationDriver(BaseDriver):
    """Simulated respiration driver."""

    def __init__(self, sampling_rate: float = 25.0, breaths_per_minute: float = 12.0, noise_level: float = 0.01) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=1)
        self.noise_level = noise_level
        self._freq = breaths_per_minute / 60.0  # cycles per second
        self._phase = 0.0

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        while self._running:
            timestamp = time.time()
            t = self._phase
            signal = np.sin(2 * np.pi * self._freq * t)
            noise = np.random.randn() * self.noise_level
            data = np.array([signal + noise], dtype=np.float32)
            self._phase += period
            yield timestamp, data
            await asyncio.sleep(period)