"""
Facial processing agent for neurOS.

The :class:`FacialAgent` consumes video frames and computes simple
features related to facial asymmetry and eye openness.  It divides
each frame into left/right and top/bottom halves (in grayscale) and
computes the sum of pixel intensities in each region.  The features
returned are the ratios of left vs. right and top vs. bottom
intensities.  These ratios can loosely correlate with facial
expressions or eye blink detection.

This is a lightweight placeholder for more sophisticated facial
analysis pipelines.  In practice, one might use landmarks or deep
models to detect facial action units, eye blinks or emotions.
"""

from __future__ import annotations

import asyncio
from typing import Tuple

import numpy as np

from neuros.agents.base_agent import BaseAgent


class FacialAgent(BaseAgent):
    """Agent computing simple facial ratio features."""

    def __init__(self, input_queue: asyncio.Queue, output_queue: asyncio.Queue, *, name: str = "FacialAgent") -> None:
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
            # convert to grayscale
            if frame.ndim == 3:
                gray = frame.mean(axis=2)
            else:
                gray = frame
            h, w = gray.shape
            # split frame into left/right and top/bottom halves
            left = gray[:, : w // 2].sum()
            right = gray[:, w // 2 :].sum()
            top = gray[: h // 2, :].sum()
            bottom = gray[h // 2 :, :].sum()
            total_lr = left + right + 1e-8
            total_tb = top + bottom + 1e-8
            lr_ratio = float(left / total_lr)
            tb_ratio = float(top / total_tb)
            features = np.array([lr_ratio, tb_ratio], dtype=np.float32)
            try:
                self.output_queue.put_nowait((timestamp, features))
            except asyncio.QueueFull:
                self.logger.debug("FacialAgent: output queue full – dropping frame")
                pass

    async def stop(self) -> None:
        self.running = False