"""
Motion sensor driver for neurOS.

This driver simulates inertial measurement unit (IMU) data by producing
random accelerometer and gyroscope readings at a specified sampling
rate.  It generates six channels corresponding to acceleration
(x, y, z) and angular velocity (x, y, z).  The driver can be used
to emulate movement and motion input in multimodal pipelines.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator, Tuple

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class MotionSensorDriver(BaseDriver):
    """Simulated IMU data source.

    Parameters
    ----------
    sampling_rate : float, optional
        Samples per second.  Defaults to 100.
    """

    def __init__(self, sampling_rate: float = 100.0) -> None:
        super().__init__(sampling_rate=sampling_rate, channels=6)

    async def stream(self) -> AsyncGenerator[Tuple[float, np.ndarray], None]:
        """Asynchronously yield timestamped IMU samples.

        Each sample is a NumPy array of shape ``(6,)`` containing
        acceleration (x, y, z) and gyroscope (x, y, z) values drawn
        from a standard normal distribution.  Timestamps are UNIX
        times.  The coroutine sleeps for ``1.0 / self.fs`` seconds
        between samples.
        """
        period = 1.0 / self.fs
        self.running = True
        try:
            while self.running:
                # 3D accelerometer and gyroscope data
                sample = np.random.randn(6).astype("float32")
                ts = time.time()
                yield ts, sample
                await asyncio.sleep(period)
        finally:
            self.running = False