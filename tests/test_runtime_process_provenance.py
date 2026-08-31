from __future__ import annotations

from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode


class IdentityTransform:
    def transform(self, item):
        return item


def test_snapshot_records_pickle_and_shared_memory_process_authority_before_start():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, object()))
    graph.add_node(
        RuntimeNode(
            "pickle-node",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            executor="process",
            execution_timeout_s=2.5,
        )
    )
    graph.add_node(
        RuntimeNode(
            "shared-node",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            executor="process",
            execution_timeout_s=1.25,
            process_transport="shared_memory",
            process_request_capacity_bytes=8192,
            process_response_capacity_bytes=16384,
        )
    )
    graph.connect(RuntimeEdge("source", "pickle-node"))
    graph.connect(RuntimeEdge("source", "shared-node"))

    snapshot = RuntimeExecutor(graph).snapshot()

    assert snapshot["process_execution"] == {
        "pickle-node": {
            "transport": "pickle",
            "execution_timeout_s": 2.5,
            "request_capacity_bytes": None,
            "response_capacity_bytes": None,
        },
        "shared-node": {
            "transport": "shared_memory",
            "execution_timeout_s": 1.25,
            "request_capacity_bytes": 8192,
            "response_capacity_bytes": 16384,
        },
    }
    assert snapshot["process_receipts"] == {
        "pickle-node": [],
        "shared-node": [],
    }
