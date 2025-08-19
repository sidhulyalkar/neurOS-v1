"""
Motion processing agent for neurOS.

The :class:`MotionAgent` consumes IMU samples (e.g. accelerometer and
gyroscope data) and forwards them to the output queue without
modification.  This agent serves as a bridge between the driver and
model when no filtering or feature extraction is required.  For more
sophisticated motion analysis, additional processing steps could be
added.
"""

from __future__ import annotations

import asyncio
from typing import Tuple

import numpy as np

from .base_agent import BaseAgent


class MotionAgent(BaseAgent):
    """Agent that forwards motion sensor samples unchanged."""

    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue, *, name: str = "MotionAgent") -> None:
        super().__init__(name=name)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False

    async def run(self) -> None:
        self.running = True
        while self.running:
            try:
                timestamp, sample = await self.input_queue.get()
            except asyncio.CancelledError:
                break
            features = np.array(sample, dtype=np.float32)
            try:
                self.output_queue.put_nowait((timestamp, features))
            except asyncio.QueueFull:
                self.logger.debug("MotionAgent: output queue full – dropping sample")
                pass

    async def stop(self) -> None:
        self.running = False