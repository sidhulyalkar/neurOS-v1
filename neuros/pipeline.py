"""
Pipeline wrapper for neurOS.

The :class:`Pipeline` class bundles together a driver, processing chain,
model and orchestrator.  It provides a simple interface to train a model
offline and then run the pipeline in real time.  A pipeline can be
configured with custom filters, frequency bands and adaptation settings.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .drivers.base_driver import BaseDriver
from .drivers.mock_driver import MockDriver
from .models.base_model import BaseModel
from .models.simple_classifier import SimpleClassifier
from .processing.filters import BandpassFilter, SmoothingFilter
from .processing.feature_extraction import BandPowerExtractor
from .agents.orchestrator_agent import Orchestrator
from .processing.health_monitor import QualityMonitor


@dataclass
class Pipeline:
    """Configurable wrapper for a neurOS processing pipeline."""

    driver: BaseDriver = field(default_factory=lambda: MockDriver())
    model: BaseModel = field(default_factory=lambda: SimpleClassifier())
    fs: float = 250.0
    filters: List[object] = field(default_factory=list)
    bands: Optional[Dict[str, tuple[float, float]]] = None
    adaptation: bool = True
    # optional custom processing agent for non‑EEG modalities
    processing_agent_class: Optional[type] = None
    processing_kwargs: Dict[str, object] = field(default_factory=dict)
    # optional quality monitor to compute data quality metrics
    monitor: Optional[object] = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the underlying model using feature vectors and labels."""
        # model may require training; we assume features are already computed
        self.model.train(X, y)

    async def run(self, duration: Optional[float] = None) -> Dict[str, float]:
        """Run the pipeline for a given duration and return metrics.

        If ``duration`` is None and the driver defines a ``get_duration``
        method, that duration is used automatically.  A quality monitor is
        created if none was provided, enabling pipeline health metrics to be
        reported.  Metrics include throughput, mean latency, quality
        statistics, model/driver names and any other custom values
        collected by the orchestrator.
        """
        # automatically derive duration from the driver if not provided
        run_duration = duration
        if run_duration is None and hasattr(self.driver, "get_duration"):
            try:
                run_duration = float(self.driver.get_duration())
            except Exception:
                run_duration = None
        # create default quality monitor if none provided
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
        )
        metrics = await orchestrator.run()
        return metrics