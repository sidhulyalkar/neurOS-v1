"""
Audio driver for neurOS.

This driver simulates an audio signal for testing multi‑modal
pipelines.  In its default configuration it outputs a simple
sinusoidal waveform with optional white noise.  The sampling rate
defaults to 16 kHz, which is typical for audio processing.  Real
audio input from files or microphones could be added in future
iterations by extending this class.

The driver yields timestamped NumPy arrays of shape ``(channels,)``
where ``channels`` is 1 for mono audio.  Each call to the internal
``_stream`` coroutine produces a single sample.  The slowest driver
in a multi‑modal setting will determine the effective fusion
frequency, so audio streams may be downsampled or aggregated in the
processing stage if high temporal resolution is not required.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import AsyncIterator

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class AudioDriver(BaseDriver):
    """Simulated audio driver.

    Parameters
    ----------
    sampling_rate : float, optional
        Sampling frequency in Hz.  Defaults to 16000 Hz.
    frequency : float, optional
        Base frequency of the sinusoid in Hz.  Defaults to 440 Hz
        (concert A).  Ignored if ``noise_only`` is True.
    noise_level : float, optional
        Standard deviation of Gaussian noise added to the waveform.
    noise_only : bool, optional
        If True, output random noise instead of a sinusoid.  Defaults
        to False.
    """

    def __init__(
        self,
        sampling_rate: float = 16000.0,
        frequency: float = 440.0,
        noise_level: float = 0.01,
        noise_only: bool = False,
    ) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=1)
        self.frequency = frequency
        self.noise_level = noise_level
        self.noise_only = noise_only
        self._phase = 0.0

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        two_pi_f = 2.0 * math.pi * self.frequency
        while self._running:
            timestamp = time.time()
            if self.noise_only:
                value = np.random.randn() * self.noise_level
            else:
                value = math.sin(two_pi_f * self._phase) + np.random.randn() * self.noise_level
            data = np.array([value], dtype=np.float32)
            self._phase += period
            yield timestamp, data
            await asyncio.sleep(period)