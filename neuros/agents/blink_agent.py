"""
Blink detection agent for neurOS.

This agent processes video frames and extracts features indicative of
eye blinks or eyelid movements.  It divides the frame into a fixed
number of horizontal bands and computes the average brightness in
each band.  Brightness patterns over time can be used to detect
blinks when combined with a classifier.  The number of bands can be
configured via the ``bands`` parameter.
"""

from __future__ import annotations

import asyncio
from typing import Tuple

import numpy as np

from neuros.agents.base_agent import BaseAgent


class BlinkAgent(BaseAgent):
    """Agent computing bandwise brightness features for blink detection."""

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        *,
        bands: int = 4,
        name: str = "BlinkAgent",
    ) -> None:
        super().__init__(name=name)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.bands = bands
        self.running = False

    async def run(self) -> None:
        self.running = True
        while self.running:
            try:
                timestamp, frame = await self.input_queue.get()
            except asyncio.CancelledError:
                break
            # convert to grayscale
            if frame.ndim == 3:
                gray = frame.mean(axis=2)
            else:
                gray = frame
            h, w = gray.shape
            band_size = h // self.bands
            features = []
            for i in range(self.bands):
                start = i * band_size
                end = (i + 1) * band_size if i < self.bands - 1 else h
                band = gray[start:end, :]
                features.append(float(band.mean()))
            feat_arr = np.array(features, dtype=np.float32)
            try:
                self.output_queue.put_nowait((timestamp, feat_arr))
            except asyncio.QueueFull:
                self.logger.debug("BlinkAgent: output queue full – dropping frame")
                pass

    async def stop(self) -> None:
        self.running = False
