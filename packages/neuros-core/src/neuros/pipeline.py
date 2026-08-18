"""High-level compatibility wrappers over the typed neurOS runtime."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, List, Optional

import numpy as np

from neuros.contracts import DecoderCapabilities, DecoderOutput
from neuros.processing.adaptation import AdaptiveThreshold
from neuros.processing.feature_extraction import BandPowerExtractor
from neuros.processing.filters import SmoothingFilter
from neuros.processing.health_monitor import QualityMonitor
from neuros.processing.operators import FeatureTransform
from neuros.runtime import (
    NodeKind,
    OverflowPolicy,
    RuntimeEdge,
    RuntimeExecutor,
    RuntimeGraph,
    RuntimeNode,
)


class _DecoderAdapter:
    """Normalize legacy train/predict models to the Decoder contract."""

    def __init__(self, model: Any, *, adaptation: bool = False) -> None:
        self.model = model
        self.adaptation = AdaptiveThreshold(window_size=50) if adaptation else None

    @property
    def capabilities(self) -> DecoderCapabilities:
        value = getattr(self.model, "capabilities", None)
        return value if isinstance(value, DecoderCapabilities) else DecoderCapabilities()

    def infer(self, X: np.ndarray) -> DecoderOutput:
        if hasattr(self.model, "infer"):
            output = self.model.infer(X)
        elif hasattr(self.model, "predict"):
            prediction = np.asarray(self.model.predict(X))
            output = DecoderOutput(
                prediction=prediction[0] if prediction.size == 1 else prediction,
                confidence=None,
                model_id=type(self.model).__name__,
            )
        else:
            raise TypeError("model must provide infer(X) or predict(X)")
        if not isinstance(output, DecoderOutput):
            output = DecoderOutput(prediction=output, confidence=None)
        if self.adaptation is None or output.confidence is None:
            return output
        self.adaptation.update(output.confidence)
        return replace(
            output,
            metadata={
                **dict(output.metadata),
                "adaptive_threshold": float(self.adaptation.threshold),
                "adaptive_trigger": bool(self.adaptation.should_trigger(output.confidence)),
            },
        )


def _with_default_smoothing(filters: List[object]) -> list[object]:
    result = list(filters)
    if not any(isinstance(item, SmoothingFilter) for item in result):
        result.append(SmoothingFilter(window_size=5))
    return result


def _compat_metrics(
    snapshot: dict[str, Any],
    *,
    decoder_node: str,
    model: Any,
    driver_name: str,
    monitor: Any | None,
) -> dict[str, Any]:
    node_metrics = snapshot["nodes"].get(decoder_node, {})
    samples = int(node_metrics.get("processed", 0))
    runtime_s = float(snapshot.get("runtime_seconds", 0.0))
    stage_mean_ms = sum(
        float(metrics.get("mean_latency_ms", 0.0))
        for metrics in snapshot.get("nodes", {}).values()
    )
    result: dict[str, Any] = {
        "duration": runtime_s,
        "samples": samples,
        "throughput": samples / runtime_s if runtime_s > 0 else 0.0,
        "mean_latency": stage_mean_ms / 1000.0,
        "model": type(model).__name__,
        "driver": driver_name,
        "runtime": snapshot,
        "dropped": sum(
            int(edge.get("dropped", 0)) for edge in snapshot.get("edges", {}).values()
        ),
    }
    if monitor is not None and hasattr(monitor, "result"):
        result.update(monitor.result())
    return result


@dataclass
class Pipeline:
    """Single-stream facade that compiles to a :class:`RuntimeGraph`.

    Custom legacy ``processing_agent_class`` implementations temporarily use the
    old orchestrator.  Standard pipelines execute exclusively through the native
    graph runtime.
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

    def to_graph(self) -> RuntimeGraph:
        if self.processing_agent_class is not None:
            raise RuntimeError(
                "Custom processing agents are legacy-only; migrate them to a Transform "
                "before compiling to RuntimeGraph"
            )
        monitor = self.monitor
        if monitor is None:
            monitor = QualityMonitor()
            self.monitor = monitor
        transform = FeatureTransform(
            filters=_with_default_smoothing(self.filters),
            extractor=BandPowerExtractor(fs=self.fs, bands=self.bands),
        )
        graph = RuntimeGraph()
        graph.add_node(RuntimeNode("source:primary", NodeKind.SOURCE, self.driver))
        graph.add_node(RuntimeNode("transform:features", NodeKind.TRANSFORM, transform))
        graph.add_node(
            RuntimeNode(
                "decoder:primary",
                NodeKind.DECODER,
                _DecoderAdapter(self.model, adaptation=self.adaptation),
            )
        )
        if monitor is not None:
            graph.add_node(RuntimeNode("monitor:quality", NodeKind.MONITOR, monitor))
        graph.connect(
            RuntimeEdge(
                "source:primary",
                "transform:features",
                capacity=self.queue_capacity,
                overflow=self.overflow_policy.value,
            )
        )
        graph.connect(
            RuntimeEdge(
                "transform:features",
                "decoder:primary",
                capacity=self.queue_capacity,
                overflow=self.overflow_policy.value,
            )
        )
        graph.validate()
        return graph

    async def run(self, duration: Optional[float] = None) -> Dict[str, Any]:
        if self.processing_agent_class is not None:
            from neuros.agents.orchestrator_agent import Orchestrator

            orchestrator = Orchestrator(
                driver=self.driver,
                model=self.model,
                fs=self.fs,
                duration=duration,
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

        run_duration = duration
        if run_duration is None and hasattr(self.driver, "get_duration"):
            try:
                candidate = float(self.driver.get_duration())
                run_duration = candidate if candidate > 0 else None
            except (TypeError, ValueError):
                pass
        executor = RuntimeExecutor(self.to_graph())
        snapshot = (
            await executor.run_for(run_duration)
            if run_duration is not None
            else await executor.run()
        )
        return _compat_metrics(
            snapshot,
            decoder_node="decoder:primary",
            model=self.model,
            driver_name=type(self.driver).__name__,
            monitor=self.monitor,
        )


@dataclass
class MultiModalPipeline:
    """Multi-stream facade using the same RuntimeExecutor as Pipeline."""

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

    @property
    def _has_legacy_agents(self) -> bool:
        return any(item is not None for item in (self.processing_agent_classes or []))

    def to_graph(self) -> RuntimeGraph:
        if self._has_legacy_agents:
            raise RuntimeError("Custom processing agents must be migrated before graph compilation")
        monitor = self.monitor
        if monitor is None:
            monitor = QualityMonitor()
            self.monitor = monitor
        extractors = self.extractors or [None] * len(self.drivers)
        fs_list = self.fs_list or [None] * len(self.drivers)
        filters_list = self.filters_list or [None] * len(self.drivers)
        graph = RuntimeGraph()
        tails: list[str] = []
        for index, driver in enumerate(self.drivers):
            source_id = f"source:{index}"
            transform_id = f"transform:{index}"
            fs = fs_list[index] or getattr(driver, "sampling_rate", 250.0)
            extractor = extractors[index] or BandPowerExtractor(fs=fs)
            filters = _with_default_smoothing(filters_list[index] or [])
            graph.add_node(RuntimeNode(source_id, NodeKind.SOURCE, driver))
            graph.add_node(
                RuntimeNode(
                    transform_id,
                    NodeKind.TRANSFORM,
                    FeatureTransform(filters=filters, extractor=extractor),
                )
            )
            graph.connect(
                RuntimeEdge(
                    source_id,
                    transform_id,
                    capacity=self.queue_capacity,
                    overflow=self.overflow_policy.value,
                )
            )
            tails.append(transform_id)
        if len(tails) == 1:
            upstream = tails[0]
        else:
            upstream = "fusion:primary"
            graph.add_node(RuntimeNode(upstream, NodeKind.FUSION, None))
            for tail in tails:
                graph.connect(
                    RuntimeEdge(
                        tail,
                        upstream,
                        capacity=self.queue_capacity,
                        overflow=self.overflow_policy.value,
                    )
                )
        graph.add_node(
            RuntimeNode(
                "decoder:primary",
                NodeKind.DECODER,
                _DecoderAdapter(self.model, adaptation=self.adaptation),
            )
        )
        graph.connect(
            RuntimeEdge(
                upstream,
                "decoder:primary",
                capacity=self.queue_capacity,
                overflow=self.overflow_policy.value,
            )
        )
        if monitor is not None:
            graph.add_node(RuntimeNode("monitor:quality", NodeKind.MONITOR, monitor))
        graph.validate()
        return graph

    async def run(self, duration: Optional[float] = None) -> Dict[str, Any]:
        if self._has_legacy_agents:
            from neuros.agents.multimodal_orchestrator import MultiModalOrchestrator

            orchestrator = MultiModalOrchestrator(
                drivers=self.drivers,
                model=self.model,
                extractors=self.extractors or [None] * len(self.drivers),
                fs_list=self.fs_list or [None] * len(self.drivers),
                filters_list=self.filters_list or [None] * len(self.drivers),
                adaptation=self.adaptation,
                duration=duration,
                processing_agent_classes=self.processing_agent_classes or [None] * len(self.drivers),
                processing_kwargs_list=self.processing_kwargs_list or [None] * len(self.drivers),
                monitor=self.monitor,
                queue_capacity=self.queue_capacity,
                overflow_policy=self.overflow_policy,
            )
            return await orchestrator.run()
        executor = RuntimeExecutor(self.to_graph())
        snapshot = await executor.run_for(duration) if duration is not None else await executor.run()
        return _compat_metrics(
            snapshot,
            decoder_node="decoder:primary",
            model=self.model,
            driver_name="+".join(type(driver).__name__ for driver in self.drivers),
            monitor=self.monitor,
        )
