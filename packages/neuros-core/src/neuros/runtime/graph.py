"""Typed runtime graph specifications for neurOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

from .queues import OverflowPolicy


class NodeKind(str, Enum):
    SOURCE = "source"
    TRANSFORM = "transform"
    FUSION = "fusion"
    DECODER = "decoder"
    SINK = "sink"
    MONITOR = "monitor"


@dataclass(frozen=True, slots=True)
class RuntimeNode:
    node_id: str
    kind: NodeKind
    operator: Any
    executor: str = "inline"
    latency_budget_ms: float | None = None
    execution_timeout_s: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if self.latency_budget_ms is not None and self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
        if self.execution_timeout_s is not None and self.execution_timeout_s <= 0:
            raise ValueError("execution_timeout_s must be positive")
        if self.executor not in {"inline", "thread", "process", "gpu"}:
            raise ValueError(f"Unsupported executor: {self.executor}")
        if self.kind is NodeKind.SOURCE and self.executor != "inline":
            raise ValueError(
                "Source nodes currently require executor='inline'; source "
                "lifecycle/stream isolation is not yet implemented"
            )
        if self.execution_timeout_s is not None and self.executor != "process":
            raise ValueError(
                "execution_timeout_s is only authoritative for executor='process'"
            )
        if self.executor == "process" and self.execution_timeout_s is None:
            raise ValueError(
                "executor='process' requires an explicit execution_timeout_s; "
                "latency_budget_ms is an SLO and is not termination authority"
            )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class RuntimeEdge:
    source: str
    target: str
    capacity: int = 8
    overflow: str = "drop_oldest"

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        OverflowPolicy(self.overflow)
        if self.source == self.target:
            raise ValueError("self edges are not allowed")


@dataclass(slots=True)
class RuntimeGraph:
    """A validated directed acyclic graph of runtime operators."""

    nodes: dict[str, RuntimeNode] = field(default_factory=dict)
    edges: list[RuntimeEdge] = field(default_factory=list)

    def add_node(self, node: RuntimeNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node

    def connect(self, edge: RuntimeEdge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Both edge endpoints must be registered nodes")
        if any(
            existing.source == edge.source and existing.target == edge.target
            for existing in self.edges
        ):
            raise ValueError(f"Duplicate edge: {edge.source} -> {edge.target}")
        self.edges.append(edge)

    def incoming(self, node_id: str) -> tuple[RuntimeEdge, ...]:
        return tuple(edge for edge in self.edges if edge.target == node_id)

    def outgoing(self, node_id: str) -> tuple[RuntimeEdge, ...]:
        return tuple(edge for edge in self.edges if edge.source == node_id)

    def topological_order(self) -> tuple[str, ...]:
        indegree = {node_id: 0 for node_id in self.nodes}
        adjacency: dict[str, list[str]] = {node_id: [] for node_id in self.nodes}
        for edge in self.edges:
            indegree[edge.target] += 1
            adjacency[edge.source].append(edge.target)
        ready = sorted(node_id for node_id, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            node_id = ready.pop(0)
            order.append(node_id)
            for target in adjacency[node_id]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    ready.append(target)
                    ready.sort()
        if len(order) != len(self.nodes):
            raise ValueError("RuntimeGraph must be acyclic")
        return tuple(order)

    def validate(self) -> None:
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"Invalid edge: {edge.source} -> {edge.target}")
        self.topological_order()

        for node_id, node in self.nodes.items():
            incoming = self.incoming(node_id)
            outgoing = self.outgoing(node_id)
            if node.kind is NodeKind.SOURCE and incoming:
                raise ValueError(f"Source node {node_id} cannot have incoming edges")
            if node.kind is NodeKind.FUSION and len(incoming) < 2:
                raise ValueError(f"Fusion node {node_id} requires at least two inputs")
            if node.kind in (NodeKind.TRANSFORM, NodeKind.DECODER, NodeKind.SINK):
                if len(incoming) != 1:
                    raise ValueError(
                        f"{node.kind.value} node {node_id} requires exactly one input"
                    )
            if node.kind is NodeKind.SINK and outgoing:
                raise ValueError(f"Sink node {node_id} cannot have outgoing edges")
            if node.kind is NodeKind.MONITOR and (incoming or outgoing):
                raise ValueError(
                    f"Monitor node {node_id} is observational and must not own data edges"
                )
