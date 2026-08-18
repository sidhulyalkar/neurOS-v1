"""Model inference operator for the neurOS agent runtime."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Optional

import numpy as np

from neuros.agents.base_agent import BaseAgent
from neuros.contracts import DecoderOutput
from neuros.processing.adaptation import AdaptiveThreshold
from neuros.runtime import OverflowPolicy, QueueStats, put_with_policy


class ModelAgent(BaseAgent):
    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: Optional[asyncio.Queue],
        model: Any,
        adaptation: Optional[AdaptiveThreshold] = None,
        callback: Optional[
            Callable[[float, np.ndarray, float, int, float | None], None]
        ] = None,
        *,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        queue_stats: QueueStats | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=kwargs.get("name", "ModelAgent"))
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.model = model
        self.adaptation = adaptation
        self.callback = callback
        self.overflow_policy = overflow_policy
        self.queue_stats = queue_stats or QueueStats()
        self.running = False

    def _infer_legacy(self, X: np.ndarray) -> DecoderOutput:
        started_ns = time.perf_counter_ns()
        pred = np.asarray(self.model.predict(X))
        prediction = pred.reshape(-1)[0].item() if pred.size == 1 else pred
        probabilities = None
        confidence = None
        predict_proba = getattr(self.model, "predict_proba", None)
        if callable(predict_proba):
            try:
                raw = predict_proba(X)
                if raw is not None:
                    probs = np.asarray(raw, dtype=float)
                    probabilities = probs[0] if probs.ndim > 1 else probs
                    if probabilities.size:
                        confidence = float(np.max(probabilities))
            except (AttributeError, NotImplementedError):
                probabilities = None
                confidence = None
        return DecoderOutput(
            prediction=prediction,
            confidence=confidence,
            probabilities=probabilities,
            model_id=self.model.__class__.__name__,
            inference_time_ns=time.perf_counter_ns() - started_ns,
        )

    def _infer(self, X: np.ndarray) -> DecoderOutput:
        infer = getattr(self.model, "infer", None)
        if callable(infer):
            output = infer(X)
            if not isinstance(output, DecoderOutput):
                raise TypeError("model.infer() must return DecoderOutput")
            return output
        return self._infer_legacy(X)

    async def run(self) -> None:
        if not getattr(self.model, "is_trained", True):
            raise RuntimeError(
                "Model must be trained before running ModelAgent. Call model.train()."
            )
        self.running = True
        while self.running:
            try:
                timestamp, features = await self.input_queue.get()
            except asyncio.CancelledError:
                break
            try:
                X = np.asarray(features).reshape(1, -1)
                output = self._infer(X)
                prediction = output.prediction
                if isinstance(prediction, np.ndarray):
                    if prediction.size != 1:
                        raise ValueError(
                            "Streaming ModelAgent requires one prediction per input"
                        )
                    prediction = prediction.reshape(-1)[0].item()
                label = int(prediction)
                confidence = output.confidence
                trigger = True
                if self.adaptation is not None and confidence is not None:
                    self.adaptation.update(confidence)
                    trigger = self.adaptation.should_trigger(confidence)
                latency = max(0.0, time.time() - float(timestamp))
                if trigger and self.output_queue is not None:
                    accepted = await put_with_policy(
                        self.output_queue,
                        (timestamp, label, confidence, latency),
                        policy=self.overflow_policy,
                        stats=self.queue_stats,
                    )
                    if not accepted:
                        self.logger.debug("Result queue full; newest result dropped")
                if self.callback is not None:
                    try:
                        self.callback(
                            float(timestamp),
                            np.asarray(features),
                            latency,
                            label,
                            confidence,
                        )
                    except Exception as exc:
                        self.logger.exception("Error in callback: %s", exc)
            finally:
                self.input_queue.task_done()

    async def stop(self) -> None:
        self.running = False
