"""Multi-modal orchestrator for neurOS."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import numpy as np

from neuros.agents.base_agent import BaseAgent
from neuros.agents.device_agent import DeviceAgent
from neuros.agents.fusion_agent import FusionAgent
from neuros.agents.model_agent import ModelAgent
from neuros.agents.processing_agent import ProcessingAgent
from neuros.processing.adaptation import AdaptiveThreshold
from neuros.processing.feature_extraction import BandPowerExtractor
from neuros.processing.filters import SmoothingFilter
from neuros.runtime import OverflowPolicy, QueueStats


class MultiModalOrchestrator(BaseAgent):
    """Coordinate multiple sources, processing chains, fusion, and a decoder."""

    def __init__(
        self,
        drivers: List[Any],
        model: Any,
        *,
        extractors: Optional[List[Any]] = None,
        fs_list: Optional[List[Optional[float]]] = None,
        filters_list: Optional[List[Optional[List[Any]]]] = None,
        adaptation: bool = True,
        duration: Optional[float] = None,
        processing_agent_classes: Optional[List[Optional[type]]] = None,
        processing_kwargs_list: Optional[List[Optional[Dict[str, Any]]]] = None,
        monitor: Optional[Any] = None,
        name: Optional[str] = None,
        queue_capacity: int = 100,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> None:
        super().__init__(name=name or "MultiModalOrchestrator")
        if not drivers:
            raise ValueError("At least one driver must be provided")
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        self.drivers = drivers
        self.model = model
        self.extractors = extractors or [None] * len(drivers)
        self.fs_list = fs_list or [None] * len(drivers)
        self.filters_list = filters_list or [None] * len(drivers)
        self.adaptation_enabled = adaptation
        self.duration = duration
        self.processing_agent_classes = processing_agent_classes or [None] * len(drivers)
        self.processing_kwargs_list = processing_kwargs_list or [None] * len(drivers)
        self.monitor = monitor
        self.queue_capacity = queue_capacity
        self.overflow_policy = overflow_policy
        self.latencies: List[float] = []
        self.sample_count = 0
        self._tasks: List[asyncio.Task] = []
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

    async def run(self) -> Dict[str, Any]:
        feat_queues: List[asyncio.Queue] = []
        tasks: List[asyncio.Task] = []
        for idx, driver in enumerate(self.drivers):
            raw_q = asyncio.Queue(maxsize=self.queue_capacity)
            feat_q = asyncio.Queue(maxsize=self.queue_capacity)
            feat_queues.append(feat_q)
            raw_stats = QueueStats()
            feat_stats = QueueStats()
            self._queue_stats[f"raw_{idx}"] = raw_stats
            self._queue_stats[f"features_{idx}"] = feat_stats
            device_agent = DeviceAgent(
                driver,
                raw_q,
                overflow_policy=self.overflow_policy,
                queue_stats=raw_stats,
            )
            tasks.append(asyncio.create_task(device_agent.run()))
            custom_cls = self.processing_agent_classes[idx] if idx < len(self.processing_agent_classes) else None
            custom_kwargs = (
                self.processing_kwargs_list[idx]
                if idx < len(self.processing_kwargs_list) and self.processing_kwargs_list[idx] is not None
                else {}
            )
            if custom_cls is not None:
                processing_agent = custom_cls(raw_q, feat_q, **custom_kwargs)
            else:
                filters = (
                    self.filters_list[idx]
                    if idx < len(self.filters_list) and self.filters_list[idx] is not None
                    else []
                )
                if not any(isinstance(f, SmoothingFilter) for f in filters):
                    filters = filters + [SmoothingFilter(window_size=5)]
                extractor = self.extractors[idx] if idx < len(self.extractors) else None
                if extractor is None:
                    fs = (
                        self.fs_list[idx]
                        if idx < len(self.fs_list) and self.fs_list[idx] is not None
                        else getattr(driver, "sampling_rate", 250.0)
                    )
                    extractor = BandPowerExtractor(fs=fs)
                processing_agent = ProcessingAgent(
                    raw_q,
                    feat_q,
                    filters=filters,
                    extractor=extractor,
                    monitor=self.monitor,
                    overflow_policy=self.overflow_policy,
                    queue_stats=feat_stats,
                )
            tasks.append(asyncio.create_task(processing_agent.run()))

        fused_queue = asyncio.Queue(maxsize=self.queue_capacity)
        fusion_stats = QueueStats()
        result_stats = QueueStats()
        self._queue_stats["fused"] = fusion_stats
        self._queue_stats["results"] = result_stats
        fusion_agent = FusionAgent(
            feat_queues,
            fused_queue,
            overflow_policy=self.overflow_policy,
            queue_stats=fusion_stats,
        )
        tasks.append(asyncio.create_task(fusion_agent.run()))
        result_queue = asyncio.Queue(maxsize=self.queue_capacity)
        adapt_obj = AdaptiveThreshold(window_size=50) if self.adaptation_enabled else None
        model_agent = ModelAgent(
            fused_queue,
            result_queue,
            model=self.model,
            adaptation=adapt_obj,
            callback=self._on_result,
            overflow_policy=self.overflow_policy,
            queue_stats=result_stats,
        )
        tasks.append(asyncio.create_task(model_agent.run()))
        self._tasks = tasks

        started = time.perf_counter()
        try:
            if self.duration is not None:
                await asyncio.sleep(self.duration)
            else:
                while True:
                    await asyncio.sleep(1)
        finally:
            for task in tasks:
                task.cancel()
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            for driver in self.drivers:
                await driver.stop()

        runtime = max(0.0, time.perf_counter() - started)
        metrics: Dict[str, Any] = {
            "duration": runtime,
            "samples": self.sample_count,
            "throughput": self.sample_count / runtime if runtime else 0.0,
            "mean_latency": float(np.mean(self.latencies)) if self.latencies else 0.0,
            "model": self.model.__class__.__name__,
            "driver": "+".join(driver.__class__.__name__ for driver in self.drivers),
        }
        for name, stats in self._queue_stats.items():
            metrics[f"{name}_queue_dropped"] = stats.dropped
            metrics[f"{name}_queue_high_water_mark"] = stats.high_water_mark
        return metrics
