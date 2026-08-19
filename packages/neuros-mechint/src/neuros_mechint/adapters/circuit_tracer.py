"""circuit-tracer attribution normalization without upgrading attribution to causation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True, slots=True)
class AttributionGraphSummary:
    """Direct feature-to-target attribution scores extracted from a graph."""

    feature_scores: Mapping[str, float]
    feature_count: int
    logit_target_count: int
    graph_node_count: int
    aggregation: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "feature_scores",
            MappingProxyType({str(key): float(value) for key, value in self.feature_scores.items()}),
        )

    def ranked_features(self, k: int | None = None) -> tuple[tuple[str, float], ...]:
        ranked = tuple(
            sorted(self.feature_scores.items(), key=lambda item: abs(item[1]), reverse=True)
        )
        return ranked if k is None else ranked[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_scores": dict(self.feature_scores),
            "feature_count": self.feature_count,
            "logit_target_count": self.logit_target_count,
            "graph_node_count": self.graph_node_count,
            "aggregation": self.aggregation,
        }


class CircuitTracerAdapter:
    """Thin bridge to the official circuit-tracer attribution graph API.

    ``attribute`` remains attribution. The adapter intentionally emits an
    ``AttributionGraphSummary`` rather than a causal-effect record. Stronger
    claims require intervention-based faithfulness testing on the nominated
    features or components.
    """

    def attribute(self, prompt: Any, replacement_model: Any, **kwargs: Any) -> Any:
        try:
            from circuit_tracer import attribute
        except ImportError as exc:
            raise ImportError(
                "circuit-tracer is not installed; install the official circuit-tracer package"
            ) from exc
        return attribute(prompt, replacement_model, **kwargs)

    def summarize_graph(self, graph: Any) -> AttributionGraphSummary:
        active_features = getattr(graph, "active_features", None)
        adjacency = getattr(graph, "adjacency_matrix", None)
        probabilities = getattr(graph, "logit_probabilities", None)
        if not isinstance(active_features, torch.Tensor):
            raise TypeError("circuit-tracer Graph.active_features must be a tensor")
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("circuit-tracer Graph.adjacency_matrix must be a tensor")
        if not isinstance(probabilities, torch.Tensor):
            raise TypeError("circuit-tracer Graph.logit_probabilities must be a tensor")
        if active_features.ndim != 2 or active_features.shape[1] < 3:
            raise ValueError("active_features must have shape [n_features, >=3]")
        if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
            raise ValueError("adjacency_matrix must be square")

        n_features = int(active_features.shape[0])
        n_logits = int(probabilities.numel())
        if n_features == 0:
            raise ValueError("attribution graph contains no active features")
        if n_logits == 0:
            raise ValueError("attribution graph contains no logit targets")
        if adjacency.shape[0] < n_features + n_logits:
            raise ValueError("adjacency_matrix is too small for graph feature/logit nodes")

        direct = adjacency[-n_logits:, :n_features].detach().to(dtype=torch.float64, device="cpu")
        weights = probabilities.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
        normalizer = float(weights.abs().sum())
        if normalizer <= 1e-12:
            weights = torch.full_like(weights, 1.0 / n_logits)
        else:
            weights = weights / normalizer
        aggregate = torch.matmul(weights, direct)

        triples = active_features.detach().to(device="cpu").numpy()
        scores = {}
        for index, triple in enumerate(triples):
            layer, position, feature = (int(triple[0]), int(triple[1]), int(triple[2]))
            key = f"feature:L{layer}:P{position}:F{feature}"
            if key in scores:
                raise ValueError(f"duplicate active feature identity {key!r}")
            scores[key] = float(aggregate[index].item())

        return AttributionGraphSummary(
            feature_scores=scores,
            feature_count=n_features,
            logit_target_count=n_logits,
            graph_node_count=int(adjacency.shape[0]),
            aggregation="probability-weighted-direct-feature-to-logit-edge",
        )

    def candidate(self, graph: Any, *, k: int, name: str = "circuit-tracer-top-features") -> Any:
        """Convert top attribution features into a faithfulness candidate.

        This local import avoids making the adapter layer depend on the benchmark
        package during import. The returned candidate still requires a separate
        feature intervention implementation before it can be called faithful.
        """

        if k <= 0:
            raise ValueError("k must be positive")
        from neuros_mechint.benchmarks.faithfulness import CircuitCandidate

        summary = self.summarize_graph(graph)
        ranked = summary.ranked_features(k)
        return CircuitCandidate(
            name=name,
            targets=tuple(key for key, _ in ranked),
            scores=dict(ranked),
            source="circuit-tracer-attribution",
        )


def feature_identity(layer: int, position: int, feature: int) -> str:
    """Canonical circuit-tracer feature key used by the adapter."""

    values = np.asarray([layer, position, feature], dtype=np.int64)
    if np.any(values < 0):
        raise ValueError("feature identity indices must be non-negative")
    return f"feature:L{int(layer)}:P{int(position)}:F{int(feature)}"
