"""
Orchestrator for neurOS.

The orchestrator composes multiple agents into a functioning pipeline.  It
creates queues for inter‑agent communication, spawns agent tasks and
collects metrics such as latency and throughput.  The orchestrator exposes a
simple API to run the pipeline for a specified duration or until
manually stopped.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import numpy as np

from neuros.drivers.base_driver import BaseDriver
from neuros.models.base_model import BaseModel
from neuros.processing.adaptation import AdaptiveThreshold
from neuros.processing.filters import BandpassFilter, SmoothingFilter
from neuros.processing.feature_extraction import BandPowerExtractor
from neuros.agents.device_agent import DeviceAgent
from neuros.agents.processing_agent import ProcessingAgent
from neuros.agents.model_agent import ModelAgent
from neuros.agents.base_agent import BaseAgent


class Orchestrator(BaseAgent):
    """Coordinate drivers, processors and models into a pipeline.

    The orchestrator composes a device agent, a processing agent and a
    model agent to form a complete data processing pipeline.  It
    supports custom processing agent classes for handling non‑EEG
    modalities (e.g. video, motion sensors).  Metrics such as
    throughput and mean latency are collected during runs.
    """

    def __init__(
        self,
        driver: BaseDriver,
        model: BaseModel,
        *,
        fs: float,
        duration: Optional[float] = None,
        bands: Optional[Dict[str, tuple[float, float]]] = None,
        adaptation: bool = True,
        filters: Optional[List[object]] = None,
        processing_agent_class: type[BaseAgent] | None = None,
        processing_kwargs: Optional[Dict[str, Any]] = None,
        monitor: Optional[object] = None,
    ) -> None:
        """
        Parameters
        ----------
        driver : BaseDriver
            The data source driver.
        model : BaseModel
            The trained classifier or regressor used for prediction.
        fs : float
            Sampling frequency of the driver (in Hz).  If using a
            non‑EEG modality, this may represent a frame rate.
        duration : float, optional
            If provided, the pipeline runs for the specified number of
            seconds; otherwise it runs until stopped.
        bands : dict[str, tuple[float, float]], optional
            Frequency bands used for band‑power extraction when using the
            default processing agent.
        adaptation : bool, optional
            If True, an adaptive threshold is applied to classification
            confidence values.
        filters : list[object], optional
            Filters to apply to the raw signal (used by the default
            processing agent).  Additional smoothing is always applied.
        processing_agent_class : type[BaseAgent], optional
            Custom processing agent class.  If provided, the
            orchestrator instantiates this class instead of the default
            :class:`ProcessingAgent`.  The class must accept
            ``(input_queue, output_queue, **processing_kwargs)`` in its
            constructor and implement a ``run`` coroutine.
        processing_kwargs : dict, optional
            Keyword arguments to pass to the custom processing agent.
        """
        super().__init__(name="Orchestrator")
        self.driver = driver
        self.model = model
        self.fs = fs
        self.duration = duration
        self.bands = bands
        self.filters = filters or []
        self.adaptation_enabled = adaptation
        self.processing_agent_class = processing_agent_class
        self.processing_kwargs = processing_kwargs or {}
        # optional quality monitor used by the processing agent
        self.monitor = monitor
        # metrics
        self.latencies: List[float] = []
        self.sample_count: int = 0
        # internal state
        self._result_queue: Optional[asyncio.Queue] = None
        self._agent_tasks: List[asyncio.Task] = []

    def _on_result(self, timestamp: float, features: np.ndarray, latency: float, label: int, confidence: float) -> None:
        self.latencies.append(latency)
        self.sample_count += 1

    async def run(self) -> Dict[str, float]:
        """Run the orchestrated pipeline.

        Returns
        -------
        dict
            A dictionary of metrics collected during the run.
        """
        # create queues
        raw_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        feat_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        result_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        # instantiate agents
        device_agent = DeviceAgent(self.driver, raw_queue)
        # choose processing agent
        if self.processing_agent_class is not None:
            # custom agent must accept (input_queue, output_queue, **kwargs)
            processing_agent = self.processing_agent_class(
                raw_queue, feat_queue, **self.processing_kwargs
            )
        else:
            # default EEG processing: band‑pass filters and band‑power extraction
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
            )
        # model agent with optional adaptation
        adapt_obj = AdaptiveThreshold(window_size=50) if self.adaptation_enabled else None
        model_agent = ModelAgent(
            feat_queue,
            result_queue,
            model=self.model,
            adaptation=adapt_obj,
            callback=self._on_result,
        )
        # schedule tasks
        tasks = [
            asyncio.create_task(device_agent.run()),
            asyncio.create_task(processing_agent.run()),
            asyncio.create_task(model_agent.run()),
        ]
        self._tasks = tasks
        # run for specified duration or until cancelled
        start_time = time.time()
        try:
            if self.duration is not None:
                await asyncio.sleep(self.duration)
            else:
                # if no duration, run until externally cancelled
                while True:
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # stop all agents
            for task in tasks:
                task.cancel()
            # ensure driver stops streaming
            await self.driver.stop()
            # wait for tasks to finish
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # compute metrics
        runtime = time.time() - start_time
        throughput = self.sample_count / runtime if runtime > 0 else 0.0
        mean_latency = float(np.mean(self.latencies)) if self.latencies else 0.0
        # base metrics
        metrics: Dict[str, float] = {
            "duration": runtime,
            "samples": self.sample_count,
            "throughput": throughput,
            "mean_latency": mean_latency,
        }
        # include quality metrics if available
        if self.monitor is not None:
            try:
                qm = self.monitor.result()
                metrics.update(qm)
            except Exception:
                pass
        # include model and driver names for downstream analytics
        metrics["model"] = self.model.__class__.__name__
        metrics["driver"] = self.driver.__class__.__name__
        return metrics

    async def _start_agents(
        self, duration: Optional[float] = None
    ) -> None:
        """
        Internal helper to start agents without waiting for completion.

        When called, it starts the device, processing and model agents and stores
        the result queue for streaming.  It does not return metrics; use
        :meth:`run` for that.  After starting the agents you can await
        :meth:`stream_results` to consume results until stopped.
        """
        # create queues
        raw_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        feat_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        result_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._result_queue = result_queue
        # instantiate agents
        device_agent = DeviceAgent(self.driver, raw_queue)
        # choose processing agent
        if self.processing_agent_class is not None:
            processing_agent = self.processing_agent_class(
                raw_queue, feat_queue, **self.processing_kwargs
            )
        else:
            filters = self.filters.copy()
            if not any(isinstance(f, SmoothingFilter) for f in filters):
                filters.append(SmoothingFilter(window_size=5))
            extractor = BandPowerExtractor(fs=self.fs, bands=self.bands)
            processing_agent = ProcessingAgent(raw_queue, feat_queue, filters=filters, extractor=extractor)
        adapt_obj = AdaptiveThreshold(window_size=50) if self.adaptation_enabled else None
        model_agent = ModelAgent(
            feat_queue,
            result_queue,
            model=self.model,
            adaptation=adapt_obj,
            callback=self._on_result,
        )
        # schedule agent tasks
        tasks = [
            asyncio.create_task(device_agent.run()),
            asyncio.create_task(processing_agent.run()),
            asyncio.create_task(model_agent.run()),
        ]
        self._agent_tasks = tasks

    async def stream_results(self) -> AsyncIterator[tuple[float, int, float, float]]:
        """
        Asynchronously yield classification results in real time.

        Before calling this method you must invoke :meth:`_start_agents` to
        initialise the pipeline.  Each yielded item is a tuple
        ``(timestamp, label, confidence, latency)``.  When the orchestrator is
        stopped the generator terminates.
        """
        if self._result_queue is None:
            raise RuntimeError("stream_results called before _start_agents")
        while True:
            try:
                item = await self._result_queue.get()
                yield item
            except asyncio.CancelledError:
                break

    async def stop(self) -> None:
        """Stop all running agent tasks and the underlying driver."""
        # cancel tasks
        for task in getattr(self, "_agent_tasks", []):
            task.cancel()
        # stop driver
        await self.driver.stop()
        # wait for tasks to finish
        for task in getattr(self, "_agent_tasks", []):
            try:
                await task
            except asyncio.CancelledError:
                pass
