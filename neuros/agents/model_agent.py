"""
Model agent for neurOS.

The :class:`ModelAgent` reads feature vectors from an input queue, invokes
its model to produce predictions and optionally adapts its decision
threshold based on recent confidence scores.  It can send predictions to
an output queue or a callback for further processing.
"""

from __future__ import annotations

import asyncio
import time
from typing import Callable, Optional, Tuple

import numpy as np

from ..models.base_model import BaseModel
from ..processing.adaptation import AdaptiveThreshold
from .base_agent import BaseAgent


class ModelAgent(BaseAgent):
    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: Optional[asyncio.Queue],
        model: BaseModel,
        adaptation: Optional[AdaptiveThreshold] = None,
        callback: Optional[
            Callable[[float, np.ndarray, float, int, float], None]
        ] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=kwargs.get("name", "ModelAgent"))
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model = model
        self.adaptation = adaptation
        self.callback = callback
        self.running = False
        # ensure the model is trained; otherwise raise later

    async def run(self) -> None:
        if not self.model.is_trained:
            raise RuntimeError(
                "Model must be trained before running ModelAgent.  Call model.train()."
            )
        self.running = True
        while self.running:
            try:
                timestamp, features = await self.input_queue.get()
            except asyncio.CancelledError:
                break
            start = time.time()
            # features expected as 1-D vector; reshape to (1, -1)
            X = features.reshape(1, -1)
            # try to get probability estimates.  Check for _model attribute
            # (used by many neurOS models) or fallback to the model itself.
            underlying = getattr(self.model, "_model", None)
            if underlying is None:
                underlying = self.model
            if hasattr(underlying, "predict_proba"):
                try:
                    probs = underlying.predict_proba(X)[0]
                    conf = float(np.max(probs))
                    label = int(np.argmax(probs))
                except Exception:
                    # fallback to predict if probability fails
                    pred = underlying.predict(X)
                    label = int(pred[0])
                    conf = 1.0
            else:
                # fallback: use predict and assign full confidence
                pred = underlying.predict(X)
                label = int(pred[0])
                conf = 1.0
            # update adaptation and threshold if provided
            trigger = True
            if self.adaptation is not None:
                self.adaptation.update(conf)
                trigger = self.adaptation.should_trigger(conf)
            latency = time.time() - timestamp
            # send to output queue if triggered
            if trigger and self.output_queue is not None:
                try:
                    self.output_queue.put_nowait((timestamp, label, conf, latency))
                except asyncio.QueueFull:
                    self.logger.debug("Model output queue full – dropping result")
                    pass
            # call callback for metrics collection
            if self.callback is not None:
                try:
                    self.callback(timestamp, features, latency, label, conf)
                except Exception as e:
                    self.logger.exception("Error in callback: %s", e)

    async def stop(self) -> None:
        self.running = False