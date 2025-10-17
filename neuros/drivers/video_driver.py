"""
Video driver for neurOS.

This driver simulates a video stream by generating random image frames
at a specified frame rate.  Each frame is represented as a NumPy array
with shape (height, width, channels).  The driver conforms to the
``BaseDriver`` interface so that it can be used interchangeably with
other data sources in neurOS.  In a real deployment, this driver could
wrap a camera capture library such as OpenCV or connect to a network
video source.
"""

from __future__ import annotations

import asyncio
import time
from typing import AsyncGenerator, Tuple

import numpy as np

from neuros.drivers.base_driver import BaseDriver


class VideoDriver(BaseDriver):
    """Simulated video stream driver.

    Parameters
    ----------
    frame_rate : float, optional
        Frames per second to generate.  Defaults to 30.
    resolution : tuple[int, int], optional
        Frame resolution (height, width).  Defaults to (64, 64).
    channels : int, optional
        Number of colour channels (e.g. 3 for RGB).  Defaults to 3.
    """

    def __init__(self, frame_rate: float = 30.0, resolution: Tuple[int, int] = (64, 64), channels: int = 3) -> None:
        super().__init__(sampling_rate=frame_rate, channels=channels)
        self.resolution = resolution
        self.channels = channels

    async def _stream(self) -> AsyncGenerator[Tuple[float, np.ndarray], None]:
        """Yield timestamped video frames at the configured frame rate.

        This coroutine produces random frames with values in the range
        [0, 1] and shape ``(height, width, channels)``.  Each yield
        returns a tuple ``(timestamp, frame)`` where timestamp is a
        UNIX time float.  Sleeping between frames respects the
        configured sampling rate; if the sampling rate is zero or
        negative, frames are emitted as fast as possible.
        """
        # compute period from sampling_rate; avoid division by zero
        period: float = 1.0 / self.sampling_rate if self.sampling_rate > 0 else 0.0
        try:
            while self._running:
                frame = np.random.rand(self.resolution[0], self.resolution[1], self.channels).astype("float32")
                ts = time.time()
                yield ts, frame
                if period > 0:
                    await asyncio.sleep(period)
                else:
                    await asyncio.sleep(0)
        finally:
            self._running = False
