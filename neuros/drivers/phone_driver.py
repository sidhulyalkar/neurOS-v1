"""
Phone usage driver for neurOS.

This driver simulates behavioural data collected from a subject's
phone during an experiment.  It emits low‑frequency metrics such as
screen state (on/off), tap rate, orientation changes and app usage.
The ``PhoneDriver`` inherits from :class:`BaseDriver` and produces
timestamped samples at a configurable sampling rate.  Each sample is
a one‑dimensional NumPy array containing the simulated metrics.  In
a future iteration this driver could interface with a mobile SDK or
companion app to stream real sensor data over BLE or Wi‑Fi.
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import AsyncIterator, Tuple

import numpy as np

from .base_driver import BaseDriver


class PhoneDriver(BaseDriver):
    """Simulate phone usage metrics.

    Parameters
    ----------
    sampling_rate : float
        Rate at which samples are generated (Hz).  Defaults to 1 Hz
        because phone metrics typically change slowly compared to
        electrophysiological signals.
    channels : int
        Number of metrics produced per sample.  Defaults to 3 and
        corresponds to ``[screen_on, tap_rate, orientation]``.  An
        additional channel could be added for volume or other usage
        statistics.
    """

    def __init__(self, sampling_rate: float = 1.0, channels: int = 3) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=channels)
        # internal state to simulate user interactions
        self.screen_on: bool = True
        self.orientation: float = 0.0  # radians, 0 = portrait
        self.tap_rate: float = 0.0  # taps per second over last interval

    async def _stream(self) -> AsyncIterator[Tuple[float, np.ndarray]]:
        """Generate phone usage samples asynchronously.

        At each interval this coroutine updates the internal state
        randomly to mimic human behaviour: the screen may turn off/on,
        the user may tap the screen at varying rates and the phone
        orientation may change slightly.  The metrics are emitted as
        a NumPy array with ``channels`` elements.  Values are normalised
        to the range [0, 1] where appropriate.
        """
        period = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 1.0
        while self._running:
            ts = time.time()
            # update internal state with some randomness
            # screen toggles on/off with small probability
            if random.random() < 0.01:
                self.screen_on = not self.screen_on
            # tap rate: if screen is on, simulate bursts of taps
            if self.screen_on:
                # occasionally high activity
                if random.random() < 0.1:
                    self.tap_rate = random.uniform(5.0, 10.0)
                else:
                    self.tap_rate = random.uniform(0.0, 3.0)
            else:
                self.tap_rate = 0.0
            # orientation drift: small random walk around 0 and π/2
            delta = random.uniform(-0.1, 0.1)
            self.orientation += delta
            # wrap orientation between 0 and 2π
            self.orientation = self.orientation % (2 * np.pi)
            # assemble metrics: normalise orientation to [0, 1]
            screen_val = 1.0 if self.screen_on else 0.0
            # normalise tap_rate to [0, 1] assuming max ~10 taps/s
            tap_val = min(self.tap_rate / 10.0, 1.0)
            orient_val = self.orientation / (2 * np.pi)
            # Build array; additional metrics could be appended
            values = np.array([screen_val, tap_val, orient_val], dtype=np.float32)
            yield ts, values
            await asyncio.sleep(period)


__all__ = ["PhoneDriver"]