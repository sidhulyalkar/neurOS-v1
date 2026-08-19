"""Scientifically explicit module-pruning circuit discovery.

This module retains the historic ``AutomatedCircuitDiscovery`` name for source
compatibility, but the implementation is intentionally labeled as
ACDC-inspired module-output pruning. Canonical ACDC is edge-level and requires
a model graph with addressable sender/receiver edges. Treating a whole module
output as an edge would overstate the evidence.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Edge:
    source: str
    target: str
    importance: float = 0.0

    def with_importance(self, value: float) -> Edge:
        return Edge(self.source, self.target, float(value))


@dataclass
class Circuit:
    edges: set[Edge] = field(default_factory=set)
    nodes: set[str] = field(default_factory=set)
    performance: float = 0.0
    sparsity: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_edge(self, edge: Edge) -> None:
        self.edges.add(edge)
        self.nodes.update((edge.source, edge.target))

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": sorted((edge.source, edge.target, edge.importance) for edge in self.edges),
            "nodes": sorted(self.nodes),
            "performance": self.performance,
            "sparsity": self.sparsity,
            "metadata": dict(self.metadata),
        }


def _replace_output(output: Any, mode: str) -> Any:
    if isinstance(output, torch.Tensor):
        if mode == "zero":
            return torch.zeros_like(output)
        if mode == "mean":
            return torch.full_like(output, output.mean())
        raise ValueError(f"unsupported ablation mode: {mode!r}")
    if isinstance(output, tuple):
        values = list(output)
        for index, value in enumerate(values):
            if isinstance(value, torch.Tensor):
                values[index] = _replace_output(value, mode)
                return tuple(values)
    raise TypeError(f"cannot ablate module output of type {type(output).__name__}")


class ModuleCircuitDiscovery:
    """Rank leaf modules by necessity and evaluate the retained subnetwork.

    This is useful as a coarse localization baseline and as a teaching method.
    It is not presented as faithful edge-level ACDC.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float = 0.01,
        metric: Callable[[torch.Tensor, torch.Tensor], float] | None = None,
        ablation_method: str = "zero",
        device: str | None = None,
        verbose: bool = False,
        importance_threshold: float | None = None,
    ) -> None:
        if importance_threshold is not None:
            threshold = importance_threshold
        self.model = model
        self.threshold = float(threshold)
        self.metric = metric or self._default_metric
        self.ablation_method = ablation_method
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

    def _candidate_modules(self) -> list[str]:
        candidates = []
        for name, module in self.model.named_modules():
            if not name or len(list(module.children())) != 0:
                continue
            if any(parameter.numel() for parameter in module.parameters(recurse=False)):
                candidates.append(name)
        return candidates

    @staticmethod
    def _default_metric(output: torch.Tensor, target: torch.Tensor) -> float:
        if (
            output.ndim >= 2
            and target.ndim == output.ndim - 1
            and target.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8}
        ):
            return float(-F.cross_entropy(output, target).detach().cpu().item())
        if output.shape != target.shape:
            raise ValueError(
                "default metric requires regression targets with the same shape as output "
                "or integer class-index targets"
            )
        return float(-F.mse_loss(output, target).detach().cpu().item())

    @contextmanager
    def _ablated(self, names: Sequence[str]) -> Iterator[None]:
        modules = dict(self.model.named_modules())
        handles = []
        try:
            for name in names:
                if name not in modules:
                    raise KeyError(f"unknown module: {name}")

                def _hook(module, args, output, *, _mode: str = self.ablation_method):
                    del module, args
                    return _replace_output(output, _mode)

                handles.append(modules[name].register_forward_hook(_hook))
            yield
        finally:
            for handle in handles:
                handle.remove()

    def _score(self, inputs: torch.Tensor, targets: torch.Tensor) -> float:
        with torch.no_grad():
            return float(self.metric(self.model(inputs), targets))

    def discover_circuit(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        max_iterations: int | None = None,
    ) -> Circuit:
        candidates = self._candidate_modules()
        if max_iterations is not None:
            evaluated = candidates[:max_iterations]
            unevaluated = candidates[max_iterations:]
        else:
            evaluated = candidates
            unevaluated = []

        baseline = self._score(inputs, targets)
        importances: dict[str, float] = {}
        removed: list[str] = []
        kept: list[str] = list(unevaluated)

        for name in evaluated:
            with self._ablated([name]):
                ablated_score = self._score(inputs, targets)
            importance = baseline - ablated_score
            importances[name] = importance
            if importance >= self.threshold:
                kept.append(name)
            else:
                removed.append(name)

        # Crucially, final performance is measured with all rejected modules
        # ablated together. The legacy implementation removed hooks before
        # measuring this value, which accidentally reported full-model quality.
        with self._ablated(removed):
            final_performance = self._score(inputs, targets)

        ordered_kept = [name for name in candidates if name in kept]
        circuit = Circuit(
            performance=final_performance,
            sparsity=(len(ordered_kept) / len(candidates)) if candidates else 0.0,
            metadata={
                "algorithm": "acdc_inspired_module_pruning",
                "faithful_acdc": False,
                "threshold": self.threshold,
                "ablation_method": self.ablation_method,
                "baseline_performance": baseline,
                "removed_modules": removed,
                "evaluated_modules": evaluated,
                "unevaluated_modules": unevaluated,
            },
        )
        previous = "input"
        for name in ordered_kept:
            edge = Edge(previous, name, importances.get(name, float("nan")))
            circuit.add_edge(edge)
            previous = name
        if ordered_kept:
            circuit.add_edge(Edge(previous, "output", 0.0))
        return circuit

    @staticmethod
    def compare_circuits(circuit1: Circuit, circuit2: Circuit) -> dict[str, Any]:
        edges1 = {(edge.source, edge.target) for edge in circuit1.edges}
        edges2 = {(edge.source, edge.target) for edge in circuit2.edges}
        union = edges1 | edges2
        node_union = circuit1.nodes | circuit2.nodes
        return {
            "edge_overlap": len(edges1 & edges2) / len(union) if union else 0.0,
            "node_overlap": len(circuit1.nodes & circuit2.nodes) / len(node_union) if node_union else 0.0,
            "performance_difference": abs(circuit1.performance - circuit2.performance),
            "shared_edges": edges1 & edges2,
        }


class AutomatedCircuitDiscovery(ModuleCircuitDiscovery):
    """Compatibility name for the ACDC-inspired module-pruning baseline."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        warnings.warn(
            "AutomatedCircuitDiscovery is currently an ACDC-inspired module-output "
            "pruning baseline, not canonical edge-level ACDC. Use ModuleCircuitDiscovery "
            "for explicit semantics.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(*args, **kwargs)


__all__ = ["AutomatedCircuitDiscovery", "Circuit", "Edge", "ModuleCircuitDiscovery"]
