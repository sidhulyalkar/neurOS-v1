from __future__ import annotations

import asyncio

import pytest

from neuros.config import PipelineConfig, resolve_config
from neuros.plugins import PluginKind, PluginRegistry
from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode


class FiniteSource:
    def __init__(self, values=(1,)):
        self.values = tuple(values)

    async def start(self):
        return None

    async def stop(self):
        return None

    async def frames(self):
        for value in self.values:
            await asyncio.sleep(0)
            yield value


class IdentityTransform:
    def transform(self, item):
        return item


class FailingTransform:
    def transform(self, item):
        raise LookupError(f"evidence failure {item}")


class IdentityDecoder:
    def infer(self, item):
        return item


class Sink:
    def __init__(self):
        self.items = []

    def write(self, item):
        self.items.append(item)


class Monitor:
    def update(self, payload):
        return None


def _manifest_graph(*, transform=IdentityTransform()):
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, FiniteSource()))
    graph.add_node(
        RuntimeNode("transform", NodeKind.TRANSFORM, transform, executor="thread")
    )
    graph.add_node(
        RuntimeNode("decoder", NodeKind.DECODER, IdentityDecoder(), executor="gpu")
    )
    graph.add_node(
        RuntimeNode(
            "sink",
            NodeKind.SINK,
            Sink(),
            executor="process",
            execution_timeout_s=2.5,
        )
    )
    graph.add_node(
        RuntimeNode(
            "monitor",
            NodeKind.MONITOR,
            Monitor(),
            executor="process",
            execution_timeout_s=1.25,
            process_transport="shared_memory",
            process_request_capacity_bytes=8192,
            process_response_capacity_bytes=16384,
        )
    )
    graph.connect(RuntimeEdge("source", "transform", overflow="block"))
    graph.connect(RuntimeEdge("transform", "decoder", overflow="block"))
    graph.connect(RuntimeEdge("decoder", "sink", overflow="block"))
    return graph


def test_snapshot_exposes_all_node_execution_authority_before_start():
    snapshot = RuntimeExecutor(_manifest_graph()).snapshot()

    assert snapshot["execution"] == {
        "decoder": {
            "kind": "decoder",
            "requested_executor": "gpu",
            "execution_domain": "event_loop",
            "scheduling_mode": "unary_task",
            "execution_timeout_s": None,
            "process_transport": None,
            "process_request_capacity_bytes": None,
            "process_response_capacity_bytes": None,
        },
        "monitor": {
            "kind": "monitor",
            "requested_executor": "process",
            "execution_domain": "persistent_process",
            "scheduling_mode": "observation_callback",
            "execution_timeout_s": 1.25,
            "process_transport": "shared_memory",
            "process_request_capacity_bytes": 8192,
            "process_response_capacity_bytes": 16384,
        },
        "sink": {
            "kind": "sink",
            "requested_executor": "process",
            "execution_domain": "persistent_process",
            "scheduling_mode": "unary_task",
            "execution_timeout_s": 2.5,
            "process_transport": "pickle",
            "process_request_capacity_bytes": None,
            "process_response_capacity_bytes": None,
        },
        "source": {
            "kind": "source",
            "requested_executor": "inline",
            "execution_domain": "event_loop",
            "scheduling_mode": "source_task",
            "execution_timeout_s": None,
            "process_transport": None,
            "process_request_capacity_bytes": None,
            "process_response_capacity_bytes": None,
        },
        "transform": {
            "kind": "transform",
            "requested_executor": "thread",
            "execution_domain": "worker_thread",
            "scheduling_mode": "unary_task",
            "execution_timeout_s": None,
            "process_transport": None,
            "process_request_capacity_bytes": None,
            "process_response_capacity_bytes": None,
        },
    }


def test_execution_snapshot_preserves_existing_process_execution_surface():
    snapshot = RuntimeExecutor(_manifest_graph()).snapshot()
    assert snapshot["process_execution"] == {
        "monitor": {
            "transport": "shared_memory",
            "execution_timeout_s": 1.25,
            "request_capacity_bytes": 8192,
            "response_capacity_bytes": 16384,
        },
        "sink": {
            "transport": "pickle",
            "execution_timeout_s": 2.5,
            "request_capacity_bytes": None,
            "response_capacity_bytes": None,
        },
    }


def test_snapshot_returns_fresh_execution_mapping_without_mutating_authority():
    executor = RuntimeExecutor(_manifest_graph())
    first = executor.snapshot()["execution"]
    first["decoder"]["execution_domain"] = "invented"
    first["extra"] = {"fake": True}

    second = executor.snapshot()["execution"]
    assert "extra" not in second
    assert second["decoder"]["execution_domain"] == "event_loop"


@pytest.mark.asyncio
async def test_execution_authority_is_stable_across_successful_lifecycle():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, FiniteSource((3, 4))))
    graph.add_node(
        RuntimeNode(
            "transform", NodeKind.TRANSFORM, IdentityTransform(), executor="thread"
        )
    )
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, Sink(), executor="gpu"))
    graph.connect(RuntimeEdge("source", "transform", overflow="block"))
    graph.connect(RuntimeEdge("transform", "sink", overflow="block"))

    executor = RuntimeExecutor(graph)
    before = executor.snapshot()["execution"]
    await executor.run()
    after = executor.snapshot()["execution"]

    assert after == before
    assert after["sink"]["requested_executor"] == "gpu"
    assert after["sink"]["execution_domain"] == "event_loop"


@pytest.mark.asyncio
async def test_execution_authority_is_stable_after_runtime_failure():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, FiniteSource((9,))))
    graph.add_node(RuntimeNode("transform", NodeKind.TRANSFORM, FailingTransform()))
    graph.connect(RuntimeEdge("source", "transform", overflow="block"))

    executor = RuntimeExecutor(graph)
    before = executor.snapshot()["execution"]
    with pytest.raises(RuntimeError, match="evidence failure 9"):
        await executor.run()
    assert executor.snapshot()["execution"] == before


def _registry() -> PluginRegistry:
    registry = PluginRegistry()
    registry.register(name="source", kind=PluginKind.SOURCE, factory=FiniteSource)
    registry.register(name="transform", kind=PluginKind.TRANSFORM, factory=IdentityTransform)
    registry.register(name="decoder", kind=PluginKind.DECODER, factory=IdentityDecoder)
    registry.register(name="sink", kind=PluginKind.SINK, factory=Sink)
    registry.register(name="monitor", kind=PluginKind.MONITOR, factory=Monitor)
    return registry


def test_schema_v2_resolved_policy_survives_into_runtime_snapshot():
    config = PipelineConfig.from_mapping(
        {
            "schema_version": 2,
            "streams": [
                {
                    "id": "eeg",
                    "source": {"plugin": "source"},
                    "transforms": [
                        {"plugin": "transform", "execution": {"executor": "thread"}}
                    ],
                }
            ],
            "decoder": {"plugin": "decoder", "execution": {"executor": "gpu"}},
            "sinks": [
                {
                    "plugin": "sink",
                    "execution": {
                        "executor": "process",
                        "execution_timeout_s": 2.0,
                    },
                }
            ],
            "monitors": [
                {"plugin": "monitor", "execution": {"executor": "thread"}}
            ],
        }
    )
    resolved = resolve_config(config, registry=_registry())
    execution = RuntimeExecutor(resolved.graph).snapshot()["execution"]

    assert execution["source:eeg"]["execution_domain"] == "event_loop"
    assert execution["transform:eeg:0"]["execution_domain"] == "worker_thread"
    assert execution["decoder:primary"] == {
        "kind": "decoder",
        "requested_executor": "gpu",
        "execution_domain": "event_loop",
        "scheduling_mode": "unary_task",
        "execution_timeout_s": None,
        "process_transport": None,
        "process_request_capacity_bytes": None,
        "process_response_capacity_bytes": None,
    }
    assert execution["sink:0"]["execution_domain"] == "persistent_process"
    assert execution["sink:0"]["execution_timeout_s"] == 2.0
    assert execution["monitor:0"]["requested_executor"] == "thread"
    assert execution["monitor:0"]["scheduling_mode"] == "observation_callback"


def test_builtin_fusion_evidence_does_not_claim_ignored_process_domain():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("a", NodeKind.SOURCE, FiniteSource()))
    graph.add_node(RuntimeNode("b", NodeKind.SOURCE, FiniteSource()))
    graph.add_node(
        RuntimeNode(
            "fusion",
            NodeKind.FUSION,
            None,
            executor="process",
            execution_timeout_s=1.0,
        )
    )
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, Sink()))
    graph.connect(RuntimeEdge("a", "fusion"))
    graph.connect(RuntimeEdge("b", "fusion"))
    graph.connect(RuntimeEdge("fusion", "sink"))

    evidence = RuntimeExecutor(graph).snapshot()["execution"]["fusion"]
    assert evidence["requested_executor"] == "process"
    assert evidence["execution_domain"] == "event_loop"
    assert evidence["scheduling_mode"] == "fusion_task"
