"""
Multi‑modal orchestrator for neurOS.

This orchestrator coordinates multiple drivers, processing agents and a
model to form a unified pipeline capable of ingesting and fusing
signals from many modalities.  Each driver is paired with its own
processing chain to clean and extract features.  A fusion agent
concatenates the feature vectors from all modalities and forwards the
result to a single model agent for prediction.  Latencies and
throughput are tracked across the entire pipeline.

The architecture mirrors the existing single‑modality orchestrator
(:class:`~neuros.agents.orchestrator_agent.Orchestrator`) but extends
it to handle multiple streams concurrently.  It preserves the
asynchronous design so that fast and slow modalities can be fused
without blocking each other.  The orchestrator collects metrics such
as mean latency and sample throughput, enabling comparison across
different multi‑modal configurations.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

from neuros.drivers.base_driver import BaseDriver
from neuros.models.base_model import BaseModel
from neuros.processing.adaptation import AdaptiveThreshold
from neuros.processing.filters import SmoothingFilter
from neuros.processing.feature_extraction import BandPowerExtractor
from neuros.agents.device_agent import DeviceAgent
from neuros.agents.processing_agent import ProcessingAgent
from neuros.agents.model_agent import ModelAgent
from neuros.agents.base_agent import BaseAgent
from neuros.agents.fusion_agent import FusionAgent


class MultiModalOrchestrator(BaseAgent):
    """Coordinate multiple drivers, processing chains and a model.

    Parameters
    ----------
    drivers : list of BaseDriver
        Data source drivers for each modality.
    model : BaseModel
        Model used for prediction on fused features.
    extractors : list of objects or None
        List of feature extractor objects corresponding to each
        driver.  Each extractor must implement an ``extract`` method
        returning a 1‑D NumPy array.  If an element is None, a
        default :class:`BandPowerExtractor` will be used with the
        sampling rate of the corresponding driver.  When a
        ``processing_agent_classes`` entry is provided the extractor
        is ignored and delegated to the custom processing agent.
    fs_list : list of float or None, optional
        Sampling rates for each driver.  If None for an entry, the
        driver's ``sampling_rate`` attribute is used.  Band power
        extraction depends on this value when a default extractor is
        used.
    filters_list : list of list, optional
        A list containing a list of filter objects for each
        modality.  These filters are applied before feature
        extraction.  If a sublist is empty or omitted, only a
        smoothing filter is used.  Use ``None`` for modalities that
        bypass filtering.
    adaptation : bool, optional
        If True (default), apply an adaptive threshold to model
        confidences via :class:`AdaptiveThreshold`.
    duration : float or None, optional
        Duration in seconds to run the pipeline.  If None, the
        orchestrator runs until externally cancelled.
    processing_agent_classes : list of type or None, optional
        Custom processing agent classes for each modality.  If an
        entry is not None, the orchestrator instantiates that class
        instead of the default :class:`ProcessingAgent` and passes the
        corresponding ``processing_kwargs_list`` element as keyword
        arguments.  Custom agents must accept ``(input_queue,
        output_queue, **kwargs)`` in their constructor and implement a
        ``run`` coroutine.
    processing_kwargs_list : list of dict, optional
        Keyword arguments passed to each custom processing agent.
    monitor : optional
        Quality monitor object used by processing agents.  If
        provided, it is passed to all default processing agents.  For
        custom agents the monitor must be handled within the custom
        implementation.
    """

    def __init__(
        self,
        drivers: List[BaseDriver],
        model: BaseModel,
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
    ) -> None:
        super().__init__(name=name or "MultiModalOrchestrator")
        if not drivers:
            raise ValueError("At least one driver must be provided.")
        self.drivers = drivers
        self.model = model
        self.extractors = extractors or [None] * len(drivers)
        if fs_list is None:
            self.fs_list = [None] * len(drivers)
        else:
            self.fs_list = fs_list
        self.filters_list = filters_list or [None] * len(drivers)
        self.adaptation_enabled = adaptation
        self.duration = duration
        self.processing_agent_classes = processing_agent_classes or [None] * len(drivers)
        self.processing_kwargs_list = processing_kwargs_list or [None] * len(drivers)
        self.monitor = monitor
        # metrics
        self.latencies: List[float] = []
        self.sample_count: int = 0
        # internal state
        self._tasks: List[asyncio.Task] = []

    def _on_result(self, timestamp: float, features: np.ndarray, latency: float, label: int, confidence: float) -> None:
        self.latencies.append(latency)
        self.sample_count += 1

    async def run(self) -> Dict[str, float]:
        # create queues and agents per modality
        raw_queues: List[asyncio.Queue] = []
        feat_queues: List[asyncio.Queue] = []
        device_agents: List[DeviceAgent] = []
        processing_agents: List[BaseAgent] = []
        tasks: List[asyncio.Task] = []

        for idx, driver in enumerate(self.drivers):
            raw_q: asyncio.Queue = asyncio.Queue(maxsize=100)
            feat_q: asyncio.Queue = asyncio.Queue(maxsize=100)
            raw_queues.append(raw_q)
            feat_queues.append(feat_q)
            # instantiate device agent
            d_agent = DeviceAgent(driver, raw_q)
            device_agents.append(d_agent)
            tasks.append(asyncio.create_task(d_agent.run()))
            # determine processing agent
            custom_cls = self.processing_agent_classes[idx] if idx < len(self.processing_agent_classes) else None
            custom_kwargs = (
                self.processing_kwargs_list[idx] if idx < len(self.processing_kwargs_list) and self.processing_kwargs_list[idx] is not None else {}
            )
            if custom_cls is not None:
                # custom processing agent handles extraction internally
                p_agent = custom_cls(raw_q, feat_q, **custom_kwargs)
            else:
                # default processing: optional filters and extractor
                filters = self.filters_list[idx] if idx < len(self.filters_list) and self.filters_list[idx] is not None else []
                # ensure smoothing filter is present
                if not any(isinstance(f, SmoothingFilter) for f in filters):
                    filters = filters + [SmoothingFilter(window_size=5)]
                # choose extractor
                extractor = self.extractors[idx] if idx < len(self.extractors) else None
                # if no extractor, fall back to BandPowerExtractor for EEG‑like data
                if extractor is None:
                    # determine sampling rate for band power
                    fs = self.fs_list[idx] if idx < len(self.fs_list) and self.fs_list[idx] is not None else getattr(driver, "sampling_rate", 250.0)
                    extractor = BandPowerExtractor(fs=fs)
                p_agent = ProcessingAgent(raw_q, feat_q, filters=filters, extractor=extractor, monitor=self.monitor)
            processing_agents.append(p_agent)
            tasks.append(asyncio.create_task(p_agent.run()))
        # fusion agent
        fused_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        fusion_agent = FusionAgent(feat_queues, fused_queue)
        tasks.append(asyncio.create_task(fusion_agent.run()))
        # model agent
        result_queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        adapt_obj = AdaptiveThreshold(window_size=50) if self.adaptation_enabled else None
        model_agent = ModelAgent(fused_queue, result_queue, model=self.model, adaptation=adapt_obj, callback=self._on_result)
        tasks.append(asyncio.create_task(model_agent.run()))
        self._tasks = tasks
        # run until duration or external cancellation
        start_time = time.time()
        try:
            if self.duration is not None:
                await asyncio.sleep(self.duration)
            else:
                while True:
                    await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass
        finally:
            # stop all tasks
            for task in tasks:
                task.cancel()
            # stop drivers
            for driver in self.drivers:
                await driver.stop()
            # await tasks to finish
            for task in tasks:
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        # compute metrics
        runtime = time.time() - start_time
        throughput = self.sample_count / runtime if runtime > 0 else 0.0
        mean_latency = float(np.mean(self.latencies)) if self.latencies else 0.0
        metrics: Dict[str, float] = {
            "duration": runtime,
            "samples": self.sample_count,
            "throughput": throughput,
            "mean_latency": mean_latency,
        }
        # include model and drivers
        metrics["model"] = self.model.__class__.__name__
        # join driver names separated by '+'
        metrics["driver"] = "+".join([d.__class__.__name__ for d in self.drivers])
        return metrics