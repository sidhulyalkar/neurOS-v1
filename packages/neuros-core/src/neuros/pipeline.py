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

from neuros.drivers.base_driver import BaseDriver
from neuros.drivers.mock_driver import MockDriver
from neuros.models.base_model import BaseModel
from neuros.models.simple_classifier import SimpleClassifier
from neuros.processing.filters import BandpassFilter, SmoothingFilter
from neuros.processing.feature_extraction import BandPowerExtractor
from neuros.agents.orchestrator_agent import Orchestrator
from neuros.processing.health_monitor import QualityMonitor


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


# ---------------------------------------------------------------------------
# Multi‑modal pipeline wrapper

@dataclass
class MultiModalPipeline:
    """Wrapper for a multi‑modal neurOS pipeline.

    This class bundles together multiple drivers, per‑modality feature
    extractors and filters, a model and the multi‑modal orchestrator.
    It exposes a simple interface to train a model offline and then
    run the pipeline in real time.  The orchestrator fuses features
    from all modalities before passing them to the model.

    Parameters
    ----------
    drivers : list of BaseDriver
        Data source drivers for each modality.
    model : BaseModel
        Model used for predicting on fused features.
    extractors : list of objects, optional
        List of feature extractor objects for each modality.  If an
        entry is ``None``, a default extractor (e.g. band power) will
        be used for that modality.
    fs_list : list of float, optional
        Sampling rates corresponding to each driver.  Used when
        constructing default extractors.  If omitted, each driver’s
        ``sampling_rate`` attribute is used.
    filters_list : list of list, optional
        List of filter lists for each modality.  Each sub‑list
        contains filter objects applied before feature extraction.  If
        an element is ``None`` or empty, a smoothing filter is used by
        default.
    adaptation : bool, optional
        If True (default), apply adaptive thresholding to model
        outputs.
    processing_agent_classes : list of type, optional
        Custom processing agent classes for each modality.  If a class
        is provided at a given index, that class is used instead of
        the default :class:`ProcessingAgent` for the corresponding
        modality.
    processing_kwargs_list : list of dict, optional
        Keyword arguments passed to each custom processing agent.
    monitor : optional
        Quality monitor passed to processing agents if supported.
    """

    drivers: List[BaseDriver]
    model: BaseModel
    extractors: List[object] | None = None
    fs_list: List[Optional[float]] | None = None
    filters_list: List[Optional[List[object]]] | None = None
    adaptation: bool = True
    processing_agent_classes: List[Optional[type]] | None = None
    processing_kwargs_list: List[Optional[Dict[str, object]]] | None = None
    monitor: Optional[object] = None

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train the underlying model on fused features.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features), where
            ``n_features`` is the sum of all modality feature lengths.
        y : np.ndarray
            Target labels.
        """
        self.model.train(X, y)

    async def run(self, duration: Optional[float] = None) -> Dict[str, float]:
        """Run the multi‑modal pipeline for a specified duration.

        Metrics such as throughput and mean latency are returned upon
        completion.  If ``duration`` is ``None`` and any driver
        provides a ``get_duration`` method, that value is used instead.
        """
        # determine run duration
        run_duration = duration
        if run_duration is None:
            # try to derive a common duration from drivers
            for driver in self.drivers:
                if hasattr(driver, "get_duration"):
                    try:
                        run_duration = float(driver.get_duration())
                        break
                    except Exception:
                        pass
        # ensure monitor exists if not provided
        if self.monitor is None:
            from .processing.health_monitor import QualityMonitor
            try:
                self.monitor = QualityMonitor()
            except Exception:
                self.monitor = None
        # instantiate orchestrator
        from .agents.multimodal_orchestrator import MultiModalOrchestrator
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
        )
        metrics = await orchestrator.run()
        return metrics