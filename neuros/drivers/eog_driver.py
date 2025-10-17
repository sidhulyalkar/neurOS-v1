"""
EOG driver for simulated electrooculography signals.

The :class:`EOGDriver` provides synthetic electrooculography (EOG)
signals, which capture eye movements and blinks.  EOG signals are
slower and lower amplitude than EEG or EMG.  This driver simulates
baseline drift plus occasional eye blinks or saccades as transient
spikes.

Such a driver is useful for prototyping eye movement detection in
neurOS.  Real systems would use an appropriate hardware driver via
BrainFlow or other interfaces.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class EOGDriver(BaseDriver):
    """Simulated electrooculography device.

    Parameters
    ----------
    sampling_rate : float, optional
        Sampling frequency in Hz.  EOG recordings often sample around
        a few hundred Hz.  Defaults to 250.0.
    channels : int, optional
        Number of EOG channels to simulate.  Defaults to 4.
    """

    def __init__(self, sampling_rate: float = 250.0, channels: int = 4) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=channels)
        # drift offset per channel
        self._offsets = np.zeros(channels, dtype=np.float32)

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        while self._running:
            timestamp = time.time()
            # slow baseline drift
            self._offsets += 0.001 * np.random.randn(self.channels)
            # simulate random saccade or blink events
            events = np.random.rand(self.channels) < 0.02  # 2% chance
            event_amp = 0.3 * (np.random.randn(self.channels)) * events.astype(np.float32)
            noise = 0.05 * np.random.randn(self.channels)
            data = self._offsets + event_amp + noise
            yield timestamp, data.astype(np.float32)
            try:
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break
