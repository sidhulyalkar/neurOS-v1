"""Deterministic execution-authority evidence for runtime snapshots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .graph import NodeKind, RuntimeNode

_EXECUTION_DOMAINS = {
    "inline": "event_loop",
    "gpu": "event_loop",
    "thread": "worker_thread",
    "process": "persistent_process",
}

_SCHEDULING_MODES = {
    NodeKind.SOURCE: "source_task",
    NodeKind.TRANSFORM: "unary_task",
    NodeKind.FUSION: "fusion_task",
    NodeKind.DECODER: "unary_task",
    NodeKind.SINK: "unary_task",
    NodeKind.MONITOR: "observation_callback",
}


@dataclass(frozen=True, slots=True)
class ExecutionAuthority:
    """Execution policy accepted for one node at executor construction."""

    node_id: str
    kind: str
    requested_executor: str
    execution_domain: str
    scheduling_mode: str
    execution_timeout_s: float | None
    process_transport: str | None
    process_request_capacity_bytes: int | None
    process_response_capacity_bytes: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "requested_executor": self.requested_executor,
            "execution_domain": self.execution_domain,
            "scheduling_mode": self.scheduling_mode,
            "execution_timeout_s": self.execution_timeout_s,
            "process_transport": self.process_transport,
            "process_request_capacity_bytes": self.process_request_capacity_bytes,
            "process_response_capacity_bytes": self.process_response_capacity_bytes,
        }


def _effective_domain(node: RuntimeNode) -> str:
    # Runtime-owned concatenate-latest fusion does not dispatch through the
    # node executor unless a custom fuse() operator exists. Preserve that truth
    # in evidence even for direct programmatic graphs. #140 owns making such
    # declarations fail closed or fully authoritative at the runtime boundary.
    if node.kind is NodeKind.FUSION and not hasattr(node.operator, "fuse"):
        return "event_loop"
    return _EXECUTION_DOMAINS[node.executor]


def capture_execution_authority(
    nodes: Mapping[str, RuntimeNode],
) -> tuple[ExecutionAuthority, ...]:
    """Capture deterministic execution authority from an already validated graph."""

    authority: list[ExecutionAuthority] = []
    for node_id, node in sorted(nodes.items()):
        is_process = node.executor == "process"
        authority.append(
            ExecutionAuthority(
                node_id=node_id,
                kind=node.kind.value,
                requested_executor=node.executor,
                execution_domain=_effective_domain(node),
                scheduling_mode=_SCHEDULING_MODES[node.kind],
                execution_timeout_s=node.execution_timeout_s if is_process else None,
                process_transport=node.process_transport if is_process else None,
                process_request_capacity_bytes=(
                    node.process_request_capacity_bytes if is_process else None
                ),
                process_response_capacity_bytes=(
                    node.process_response_capacity_bytes if is_process else None
                ),
            )
        )
    return tuple(authority)


def execution_authority_snapshot(
    authority: tuple[ExecutionAuthority, ...],
) -> dict[str, dict[str, Any]]:
    """Return a fresh JSON-compatible mapping for one captured authority set."""

    return {item.node_id: item.as_dict() for item in authority}
