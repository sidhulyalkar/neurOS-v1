"""
ECG driver for neurOS.

This driver produces a simulated electrocardiogram (ECG) waveform with
noise.  It yields timestamped samples at a configurable sampling rate.  The
waveform approximates P, QRS and T waves using Gaussian functions.  This
driver is useful for testing multi‑modal pipelines that include
physiological signals beyond EEG.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class ECGDriver(BaseDriver):
    """Simulated ECG driver.

    Parameters
    ----------
    sampling_rate : float, optional
        Sampling frequency in Hz.  Defaults to 250 Hz.
    noise_level : float, optional
        Standard deviation of Gaussian noise added to the waveform.
    """

    def __init__(self, sampling_rate: float = 250.0, noise_level: float = 0.05) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=1)
        self.noise_level = noise_level
        self._period = 1.0  # heartbeat period in seconds (60 bpm)
        self._phase = 0.0

    def _ecg_waveform(self, t: float) -> float:
        # simple synthetic ECG: sum of Gaussians for P, QRS, T waves
        p = 0.1 * np.exp(-((t - 0.1) ** 2) / 0.001)
        qrs = 1.0 * np.exp(-((t - 0.2) ** 2) / 0.0001)
        t_wave = 0.3 * np.exp(-((t - 0.4) ** 2) / 0.003)
        return p + qrs + t_wave

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        while self._running:
            timestamp = time.time()
            t = (self._phase % self._period)
            waveform = self._ecg_waveform(t)
            noise = np.random.randn() * self.noise_level
            data = np.array([waveform + noise], dtype=np.float32)
            self._phase += period
            yield timestamp, data
            await asyncio.sleep(period)