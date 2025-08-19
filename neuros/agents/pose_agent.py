"""
Pose processing agent for neurOS.

This agent consumes video frames and computes simple pose‑related
features by treating the frame as a grayscale density map.  It
calculates the centre of mass (centroid) of the pixel intensities
along the x and y axes and outputs a feature vector ``[cx, cy]`` where
both values are normalised to the range [0, 1].  The agent is a
placeholder for more sophisticated skeletal pose estimators.
"""

from __future__ import annotations

import asyncio
from typing import Tuple

import numpy as np

from .base_agent import BaseAgent


class PoseAgent(BaseAgent):
    """Agent computing centre of mass of video frames."""

    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue, *, name: str = "PoseAgent") -> None:
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
            # convert to grayscale by averaging channels
            if frame.ndim == 3:
                gray = frame.mean(axis=2)
            else:
                gray = frame
            # normalise intensities to avoid extremely small values
            gray = gray - gray.min()
            total = gray.sum() + 1e-8
            # compute centroid coordinates
            h, w = gray.shape
            x_indices = np.arange(w)
            y_indices = np.arange(h)
            cx = (gray.sum(axis=0) @ x_indices) / total
            cy = (gray.sum(axis=1) @ y_indices) / total
            # normalise to [0,1]
            cx_norm = float(cx / (w - 1)) if w > 1 else 0.0
            cy_norm = float(cy / (h - 1)) if h > 1 else 0.0
            features = np.array([cx_norm, cy_norm], dtype=np.float32)
            try:
                self.output_queue.put_nowait((timestamp, features))
            except asyncio.QueueFull:
                self.logger.debug("PoseAgent: output queue full – dropping frame")
                pass

    async def stop(self) -> None:
        self.running = False