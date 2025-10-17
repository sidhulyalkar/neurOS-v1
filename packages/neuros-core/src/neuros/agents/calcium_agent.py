"""
Processing agent for calcium imaging data.

The :class:`CalciumAgent` consumes timestamped 2D numpy arrays from an
input queue and computes simple summary features.  For each frame it
computes the mean intensity and standard deviation of the pixel
intensities.  These two values form the feature vector passed to the
model agent.  This simple approach demonstrates how calcium imaging
data can be reduced to a compact representation while preserving
information about average activity and variability.

Future versions could implement more sophisticated feature
extraction, such as spatial filtering, event detection or deep
convolutional encoders.
"""

from __future__ import annotations

import asyncio
from typing import Optional, Tuple

import numpy as np

from neuros.agents.base_agent import BaseAgent


class CalciumAgent(BaseAgent):
    """Agent to process calcium imaging frames into feature vectors."""

    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        *,
        name: str = "CalciumAgent",
        **kwargs,
    ) -> None:
        super().__init__(name=name)
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.running = False

    async def run(self) -> None:
        """Consume frames and produce mean/std features."""
        self.running = True
        while self.running:
            try:
                ts, frame = await self.input_queue.get()
            except asyncio.CancelledError:
                break
            # ensure frame is a numpy array
            arr = np.asarray(frame, dtype=np.float32)
            # compute mean and standard deviation
            mean_val = float(arr.mean())
            std_val = float(arr.std())
            features = np.array([mean_val, std_val], dtype=np.float32)
            try:
                self.output_queue.put_nowait((ts, features))
            except asyncio.QueueFull:
                # drop features if downstream queue is full
                pass

    async def stop(self) -> None:
        self.running = False