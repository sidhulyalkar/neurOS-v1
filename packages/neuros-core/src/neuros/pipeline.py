"""High-level pipeline wrappers for neurOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from neuros.agents.orchestrator_agent import Orchestrator
from neuros.processing.health_monitor import QualityMonitor
from neuros.runtime import OverflowPolicy


@dataclass
class Pipeline:
    """Configurable single-stream neurOS pipeline.

    Concrete drivers and decoders are injected by callers. This keeps
    ``neuros-core`` independent from ``neuros-drivers`` and ``neuros-models``.
    """

    driver: Any
    model: Any
    fs: float = 250.0
    filters: List[object] = field(default_factory=list)
    bands: Optional[Dict[str, tuple[float, float]]] = None
    adaptation: bool = True
    processing_agent_class: Optional[type] = None
    processing_kwargs: Dict[str, object] = field(default_factory=dict)
    monitor: Optional[object] = None
    queue_capacity: int = 100
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST

    def __post_init__(self) -> None:
        if self.fs <= 0:
            raise ValueError("fs must be positive")
        if self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.train(X, y)

    async def run(self, duration: Optional[float] = None) -> Dict[str, Any]:
        run_duration = duration
        if run_duration is None and hasattr(self.driver, "get_duration"):
            try:
                run_duration = float(self.driver.get_duration())
            except (TypeError, ValueError):
                run_duration = None
        if self.monitor is None:
            try:
                self.monitor = QualityMonitor()
            except Exception:
                self.monitor = None
        orchestrator = Orchestrator(
            driver=self.driver,
            model=self.model,
            fs=self.fs,
            duration=run_duration,
            bands=self.bands,
            adaptation=self.adaptation,
            filters=self.filters,
            processing_agent_class=self.processing_agent_class,
            processing_kwargs=self.processing_kwargs,
            monitor=self.monitor,
            queue_capacity=self.queue_capacity,
            overflow_policy=self.overflow_policy,
        )
        return await orchestrator.run()


@dataclass
class MultiModalPipeline:
    """High-level wrapper around the multi-modal runtime."""

    drivers: List[Any]
    model: Any
    extractors: List[object] | None = None
    fs_list: List[Optional[float]] | None = None
    filters_list: List[Optional[List[object]]] | None = None
    adaptation: bool = True
    processing_agent_classes: List[Optional[type]] | None = None
    processing_kwargs_list: List[Optional[Dict[str, object]]] | None = None
    monitor: Optional[object] = None
    queue_capacity: int = 100
    overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST

    def __post_init__(self) -> None:
        if not self.drivers:
            raise ValueError("drivers must contain at least one source")
        if self.queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.train(X, y)

    async def run(self, duration: Optional[float] = None) -> Dict[str, Any]:
        from neuros.agents.multimodal_orchestrator import MultiModalOrchestrator

        run_duration = duration
        if run_duration is None:
            for driver in self.drivers:
                if hasattr(driver, "get_duration"):
                    try:
                        run_duration = float(driver.get_duration())
                        break
                    except (TypeError, ValueError):
                        pass
        if self.monitor is None:
            try:
                self.monitor = QualityMonitor()
            except Exception:
                self.monitor = None
        orchestrator = MultiModalOrchestrator(
            drivers=self.drivers,
            model=self.model,
            extractors=self.extractors or [None] * len(self.drivers),
            fs_list=self.fs_list or [None] * len(self.drivers),
            filters_list=self.filters_list or [None] * len(self.drivers),
            adaptation=self.adaptation,
            duration=run_duration,
            processing_agent_classes=self.processing_agent_classes or [None] * len(self.drivers),
            processing_kwargs_list=self.processing_kwargs_list or [None] * len(self.drivers),
            monitor=self.monitor,
            queue_capacity=self.queue_capacity,
            overflow_policy=self.overflow_policy,
        )
        return await orchestrator.run()
