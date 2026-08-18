"""Single-stream runtime orchestrator for neurOS."""

from __future__ import annotations

import asyncio
import time
import warnings
from typing import Any, AsyncIterator, Dict, List, Optional

import numpy as np

from neuros.agents.base_agent import BaseAgent
from neuros.agents.device_agent import DeviceAgent
from neuros.agents.model_agent import ModelAgent
from neuros.agents.processing_agent import ProcessingAgent
from neuros.processing.adaptation import AdaptiveThreshold
from neuros.processing.feature_extraction import BandPowerExtractor
from neuros.processing.filters import SmoothingFilter
from neuros.runtime import OverflowPolicy, QueueStats, RuntimeState


class Orchestrator(BaseAgent):
    """Coordinate a source, processing chain, and decoder."""

    def __init__(
        self,
        driver: Any,
        model: Any,
        *,
        fs: float,
        duration: Optional[float] = None,
        bands: Optional[Dict[str, tuple[float, float]]] = None,
        adaptation: bool = True,
        filters: Optional[List[object]] = None,
        processing_agent_class: type[BaseAgent] | None = None,
        processing_kwargs: Optional[Dict[str, Any]] = None,
        monitor: Optional[object] = None,
        queue_capacity: int = 100,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> None:
        super().__init__(name="Orchestrator")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.driver = driver
        self.model = model
        self.fs = fs
        self.duration = duration
        self.bands = bands
        self.filters = filters or []
        self.adaptation_enabled = adaptation
        self.processing_agent_class = processing_agent_class
        self.processing_kwargs = processing_kwargs or {}
        self.monitor = monitor
        self.queue_capacity = queue_capacity
        self.overflow_policy = overflow_policy
        self.latencies: List[float] = []
        self.sample_count = 0
        self.state = RuntimeState.CREATED
        self._result_queue: asyncio.Queue | None = None
        self._agent_tasks: List[asyncio.Task] = []
        self._queue_stats: dict[str, QueueStats] = {}

    def _on_result(
        self,
        timestamp: float,
        features: np.ndarray,
        latency: float,
        label: int,
        confidence: float | None,
    ) -> None:
        self.latencies.append(latency)
        self.sample_count += 1

    def _build_agents(self) -> tuple[list[BaseAgent], asyncio.Queue]:
        raw_queue = asyncio.Queue(maxsize=self.queue_capacity)
        feat_queue = asyncio.Queue(maxsize=self.queue_capacity)
        result_queue = asyncio.Queue(maxsize=self.queue_capacity)
        self._queue_stats = {
            "raw": QueueStats(),
            "features": QueueStats(),
            "results": QueueStats(),
        }
        device_agent = DeviceAgent(
            self.driver,
            raw_queue,
            overflow_policy=self.overflow_policy,
            queue_stats=self._queue_stats["raw"],
        )
        if self.processing_agent_class is not None:
            processing_agent = self.processing_agent_class(
                raw_queue, feat_queue, **self.processing_kwargs
            )
        else:
            filters = self.filters.copy()
            if not any(isinstance(f, SmoothingFilter) for f in filters):
                filters.append(SmoothingFilter(window_size=5))
            extractor = BandPowerExtractor(fs=self.fs, bands=self.bands)
            processing_agent = ProcessingAgent(
                raw_queue,
                feat_queue,
                filters=filters,
                extractor=extractor,
                monitor=self.monitor,
                overflow_policy=self.overflow_policy,
                queue_stats=self._queue_stats["features"],
            )
        adapt_obj = AdaptiveThreshold(window_size=50) if self.adaptation_enabled else None
        model_agent = ModelAgent(
            feat_queue,
            result_queue,
            model=self.model,
            adaptation=adapt_obj,
            callback=self._on_result,
            overflow_policy=self.overflow_policy,
            queue_stats=self._queue_stats["results"],
        )
        return [device_agent, processing_agent, model_agent], result_queue

    async def start(self) -> None:
        """Start the runtime once and transition it to RUNNING."""
        if self.state is RuntimeState.RUNNING:
            return
        if self.state is RuntimeState.STARTING:
            raise RuntimeError("Orchestrator is already starting")
        self.state = RuntimeState.STARTING
        self.latencies.clear()
        self.sample_count = 0
        try:
            agents, self._result_queue = self._build_agents()
            self._agent_tasks = [asyncio.create_task(agent.run()) for agent in agents]
            await asyncio.sleep(0)
            for task in self._agent_tasks:
                if task.done() and not task.cancelled() and task.exception() is not None:
                    raise task.exception()
            self.state = RuntimeState.RUNNING
        except Exception:
            self.state = RuntimeState.FAILED
            for task in self._agent_tasks:
                task.cancel()
            raise

    async def run(self) -> Dict[str, Any]:
        """Run for the configured duration or until cancelled."""
        started = time.perf_counter()
        await self.start()
        try:
            if self.duration is not None:
                await asyncio.sleep(self.duration)
            else:
                while self.state is RuntimeState.RUNNING:
                    await asyncio.sleep(1)
        finally:
            await self.stop()
        runtime = max(0.0, time.perf_counter() - started)
        metrics: Dict[str, Any] = {
            "duration": runtime,
            "samples": self.sample_count,
            "throughput": self.sample_count / runtime if runtime else 0.0,
            "mean_latency": float(np.mean(self.latencies)) if self.latencies else 0.0,
            "model": self.model.__class__.__name__,
            "driver": self.driver.__class__.__name__,
            "runtime_state": self.state.value,
        }
        for name, stats in self._queue_stats.items():
            metrics[f"{name}_queue_accepted"] = stats.accepted
            metrics[f"{name}_queue_dropped"] = stats.dropped
            metrics[f"{name}_queue_high_water_mark"] = stats.high_water_mark
        if self.monitor is not None:
            try:
                metrics.update(self.monitor.result())
            except Exception:
                self.logger.exception("Quality monitor result failed")
        return metrics

    async def stream_results(self) -> AsyncIterator[tuple[float, int, float | None, float]]:
        """Yield decoder outputs while the runtime is active."""
        if self.state is not RuntimeState.RUNNING or self._result_queue is None:
            raise RuntimeError("Call start() before stream_results()")
        while self.state is RuntimeState.RUNNING:
            try:
                item = await self._result_queue.get()
                self._result_queue.task_done()
                yield item
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Stop all tasks and release the underlying source."""
        if self.state is RuntimeState.STOPPED:
            return
        self.state = RuntimeState.DRAINING
        for task in self._agent_tasks:
            task.cancel()
        for task in self._agent_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
            except Exception:
                self.logger.exception("Runtime task failed during shutdown")
        self._agent_tasks.clear()
        try:
            await self.driver.stop()
        finally:
            self.state = RuntimeState.STOPPED

    async def _start_agents(self, duration: Optional[float] = None) -> None:
        """Deprecated compatibility wrapper for :meth:`start`."""
        warnings.warn(
            "_start_agents() is deprecated; use start()",
            DeprecationWarning,
            stacklevel=2,
        )
        if duration is not None:
            self.duration = duration
        await self.start()
