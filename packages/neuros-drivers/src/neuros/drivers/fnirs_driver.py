"""
fNIRS/HD‑DOT driver for neurOS.

This driver simulates functional near‑infrared spectroscopy (fNIRS) or
high‑density diffuse optical tomography (HD‑DOT) data by generating
multichannel signals representing haemodynamic responses across the
cortex.  Real fNIRS/HD‑DOT systems measure changes in oxy‑ and
deoxy‑haemoglobin concentrations at sampling rates around 10 Hz.  The
``FnirsDriver`` inherits from :class:`BaseDriver` and yields
timestamped samples where each sample is a 1‑D NumPy array with one
element per optode/channel.  In this synthetic implementation we
generate slow oscillatory waves superimposed on Gaussian noise to
mimic physiological signals.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import AsyncIterator, Tuple

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class FnirsDriver(BaseDriver):
    """Simulate fNIRS/HD‑DOT multichannel haemodynamic signals.

    Parameters
    ----------
    sampling_rate : float
        Rate at which samples are produced (Hz).  Typical fNIRS
        systems operate between 2–50 Hz; default is 10 Hz.
    channels : int
        Number of optodes/channels.  Defaults to 16.
    noise_level : float
        Standard deviation of the Gaussian noise added to each
        channel.  Defaults to 0.01.
    freq_range : tuple[float, float]
        Frequency range (Hz) for the simulated haemodynamic
        oscillations.  Defaults to (0.05, 0.2) corresponding to
        Mayer waves and respiration rhythms.
    amplitude : float
        Base amplitude of the oscillatory component relative to noise.
        Defaults to 0.1.
    """

    def __init__(
        self,
        sampling_rate: float = 10.0,
        channels: int = 16,
        noise_level: float = 0.01,
        freq_range: Tuple[float, float] | Tuple[float, float] = (0.05, 0.2),
        amplitude: float = 0.1,
    ) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=channels)
        self.noise_level = noise_level
        self.amplitude = amplitude
        # assign each channel a random frequency within range
        self.freqs = np.linspace(freq_range[0], freq_range[1], channels)
        self._t = 0.0

    async def _stream(self) -> AsyncIterator[Tuple[float, np.ndarray]]:
        """Generate multichannel fNIRS samples asynchronously."""
        period = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0.1
        while self._running:
            ts = time.time()
            # compute phase vector for each channel
            t_vec = self._t + np.arange(self.channels) / self.sampling_rate
            # simulated haemodynamic signal: slow sinusoids with amplitude
            signal = self.amplitude * np.sin(2 * np.pi * self.freqs * t_vec)
            noise = self.noise_level * np.random.randn(self.channels)
            data = signal + noise
            self._t += period
            yield ts, data.astype(np.float32)
            await asyncio.sleep(period)


__all__ = ["FnirsDriver"]