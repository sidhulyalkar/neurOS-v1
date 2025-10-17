"""
Video processing agent for neurOS.

The :class:`VideoAgent` consumes video frames from an input queue and
extracts simple statistical features.  For each frame, it computes
per‑channel mean and variance and concatenates them into a feature
vector.  The agent writes feature vectors to an output queue.

This agent is useful for prototyping video‑based BCIs or behaviour
analysis pipelines.  In a real system, one might replace the feature
extraction with a deep convolutional network or other vision
techniques.
"""

from __future__ import annotations

import asyncio
from typing import Tuple

import numpy as np

from .base_agent import BaseAgent


class VideoAgent(BaseAgent):
    """Agent that extracts mean and variance from video frames."""

    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue, *, name: str = "VideoAgent") -> None:
        super().__init__(name=name)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False

    async def run(self) -> None:
        self.running = True
        while self.running:
            try:
                timestamp, frame = await self.input_queue.get()
            except asyncio.CancelledError:
                break
            # frame shape (H, W, C)
            # compute mean and variance per channel
            means: np.ndarray = frame.mean(axis=(0, 1))
            vars: np.ndarray = frame.var(axis=(0, 1))
            features = np.concatenate([means, vars])
            try:
                self.output_queue.put_nowait((timestamp, features))
            except asyncio.QueueFull:
                self.logger.debug("VideoAgent: output queue full – dropping frame")
                pass

    async def stop(self) -> None:
        self.running = False