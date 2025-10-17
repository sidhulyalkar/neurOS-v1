"""
Calcium imaging driver for neurOS.

This driver simulates calcium imaging data by producing a stream of
two‑dimensional frames representing neural activity captured via
fluorescence microscopy.  Each sample consists of a timestamp and a
2D numpy array with shape ``(height, width)``.  The driver can be
configured with a frame rate and resolution.  Pixel intensities are
generated using Gaussian noise and optional periodic signals to
mimic spontaneous neural firing events.

In a future implementation, this driver could interface with
BrainFlow or other hardware SDKs to stream real calcium imaging data
from microscopes or neural probes.  For now, it provides synthetic
data useful for prototyping pipelines.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import AsyncGenerator, Iterable, Tuple

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class CalciumImagingDriver(BaseDriver):
    """Simulate calcium imaging frames.

    Parameters
    ----------
    frame_rate : float
        Frames per second to generate.  Defaults to 30 Hz.
    resolution : tuple[int, int]
        Height and width of each frame.  Defaults to (64, 64).
    noise_level : float
        Standard deviation of the Gaussian noise added to each pixel.
        Defaults to 0.1.
    event_rate : float
        Frequency of periodic events (in Hz) added across the frame.
        A low event rate simulates occasional bursts of activity.
        Defaults to 0.5 Hz.
    amplitude : float
        Amplitude of the periodic events relative to noise.  Defaults
        to 1.0.
    """

    def __init__(
        self,
        frame_rate: float = 30.0,
        resolution: Tuple[int, int] = (64, 64),
        noise_level: float = 0.1,
        event_rate: float = 0.5,
        amplitude: float = 1.0,
    ) -> None:
        # For calcium imaging, treat each frame as a single sample.  The
        # sampling rate corresponds to the frame rate, and we set
        # channels=1 because the notion of channels does not apply to
        # two‑dimensional images.  The BaseDriver uses sampling_rate and
        # channels to schedule streaming and allocate internal buffers.
        super().__init__(sampling_rate=frame_rate, channels=1)
        self.frame_rate = frame_rate
        self.resolution = resolution
        self.noise_level = noise_level
        self.event_rate = event_rate
        self.amplitude = amplitude
        self.running = False

    async def _stream(self) -> AsyncGenerator[Tuple[float, np.ndarray], None]:
        """Internal coroutine that yields timestamped calcium imaging frames."""
        self.running = True
        period = 1.0 / self.frame_rate if self.frame_rate > 0 else 0.033
        t_start = time.time()
        while self.running:
            t_now = time.time() - t_start
            # generate base noise
            noise = np.random.normal(loc=0.0, scale=self.noise_level, size=self.resolution)
            # add a global oscillatory signal to simulate coordinated bursts
            if self.event_rate > 0:
                signal = self.amplitude * math.sin(2 * math.pi * self.event_rate * t_now)
                frame = noise + signal
            else:
                frame = noise
            yield (time.time(), frame.astype(np.float32))
            await asyncio.sleep(period)

    async def stop(self) -> None:
        """Stop streaming frames."""
        self.running = False