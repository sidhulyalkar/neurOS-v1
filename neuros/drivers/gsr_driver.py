"""
Galvanic Skin Response (GSR) driver for neurOS.

This driver produces a simulated skin conductance signal.  The
signal consists of a slowly varying tonic level with occasional
phasic peaks representing sympathetic arousal responses.  Noise
simulates measurement variability.  Use this driver to test
multi‑modal pipelines with peripheral physiological signals.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class GSRDriver(BaseDriver):
    """Simulated galvanic skin response driver."""

    def __init__(self, sampling_rate: float = 50.0, noise_level: float = 0.01) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=1)
        self.noise_level = noise_level
        self._tonic = 0.1

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        while self._running:
            timestamp = time.time()
            # slow drift
            self._tonic += np.random.randn() * 0.001
            self._tonic = max(0.0, self._tonic)
            # phasic peak with low probability
            phasic = 0.0
            if np.random.rand() < 0.02:
                phasic = np.random.rand() * 0.2
            noise = np.random.randn() * self.noise_level
            data = np.array([self._tonic + phasic + noise], dtype=np.float32)
            yield timestamp, data
            await asyncio.sleep(period)
