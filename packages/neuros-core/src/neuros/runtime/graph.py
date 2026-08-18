"""Typed runtime graph specifications for neurOS."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


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
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.node_id:
            raise ValueError("node_id must be non-empty")
        if self.latency_budget_ms is not None and self.latency_budget_ms <= 0:
            raise ValueError("latency_budget_ms must be positive")
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


@dataclass(slots=True)
class RuntimeGraph:
    """A validated directed graph of runtime operators."""

    nodes: dict[str, RuntimeNode] = field(default_factory=dict)
    edges: list[RuntimeEdge] = field(default_factory=list)

    def add_node(self, node: RuntimeNode) -> None:
        if node.node_id in self.nodes:
            raise ValueError(f"Duplicate node_id: {node.node_id}")
        self.nodes[node.node_id] = node

    def connect(self, edge: RuntimeEdge) -> None:
        if edge.source not in self.nodes or edge.target not in self.nodes:
            raise ValueError("Both edge endpoints must be registered nodes")
        self.edges.append(edge)

    def validate(self) -> None:
        for edge in self.edges:
            if edge.source not in self.nodes or edge.target not in self.nodes:
                raise ValueError(f"Invalid edge: {edge.source} -> {edge.target}")
