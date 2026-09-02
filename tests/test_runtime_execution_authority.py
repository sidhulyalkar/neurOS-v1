from __future__ import annotations

import asyncio
import threading

import pytest

from neuros.runtime import (
    NodeKind,
    RuntimeEdge,
    RuntimeExecutor,
    RuntimeGraph,
    RuntimeNode,
)


class FiniteSource:
    def __init__(self, values):
        self.values = tuple(values)

    async def start(self):
        return None

    async def stop(self):
        return None

    async def frames(self):
        for value in self.values:
            await asyncio.sleep(0)
            yield value


class CollectingSink:
    def __init__(self):
        self.items = []

    async def write(self, item):
        self.items.append(item)


class SyncThreadTransform:
    def transform(self, item):
        return threading.get_ident(), item


class AsyncThreadTransform:
    async def transform(self, item):
        await asyncio.sleep(0)
        return threading.get_ident(), item


class AsyncThreadSink:
    def __init__(self):
        self.thread_ids = []
        self.items = []

    async def write(self, item):
        await asyncio.sleep(0)
        self.thread_ids.append(threading.get_ident())
        self.items.append(item)


class AsyncThreadMonitor:
    def __init__(self):
        self.thread_ids = []
        self.nodes = []

    async def update(self, payload):
        await asyncio.sleep(0)
        self.thread_ids.append(threading.get_ident())
        self.nodes.append(payload["node_id"])


class FailingAsyncThreadMonitor:
    async def update(self, payload):
        await asyncio.sleep(0)
        raise LookupError("thread monitor rejected " + payload["node_id"])


def source_sink_graph(source, sink, *, sink_executor="inline"):
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, source))
    graph.add_node(
        RuntimeNode("sink", NodeKind.SINK, sink, executor=sink_executor)
    )
    graph.connect(RuntimeEdge("source", "sink", overflow="block"))
    return graph


def source_transform_sink_graph(transform, *, executor):
    sink = CollectingSink()
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, FiniteSource([7])))
    graph.add_node(
        RuntimeNode("transform", NodeKind.TRANSFORM, transform, executor=executor)
    )
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, sink))
    graph.connect(RuntimeEdge("source", "transform", overflow="block"))
    graph.connect(RuntimeEdge("transform", "sink", overflow="block"))
    return graph, sink


def test_source_non_inline_executor_is_rejected_instead_of_ignored():
    with pytest.raises(ValueError, match="Source nodes currently require"):
        RuntimeNode("source", NodeKind.SOURCE, object(), executor="thread")


def test_process_executor_requires_explicit_hard_timeout():
    with pytest.raises(ValueError, match="execution_timeout_s"):
        RuntimeNode("transform", NodeKind.TRANSFORM, object(), executor="process")
    node = RuntimeNode(
        "transform",
        NodeKind.TRANSFORM,
        object(),
        executor="process",
        execution_timeout_s=1.0,
    )
    assert node.execution_timeout_s == 1.0


@pytest.mark.asyncio
async def test_sync_transform_thread_executor_runs_off_event_loop_thread():
    main_thread = threading.get_ident()
    graph, sink = source_transform_sink_graph(SyncThreadTransform(), executor="thread")
    await RuntimeExecutor(graph).run()
    thread_id, value = sink.items[0]
    assert value == 7
    assert thread_id != main_thread


@pytest.mark.asyncio
async def test_async_transform_thread_executor_runs_coroutine_in_worker_thread():
    main_thread = threading.get_ident()
    graph, sink = source_transform_sink_graph(AsyncThreadTransform(), executor="thread")
    await RuntimeExecutor(graph).run()
    thread_id, value = sink.items[0]
    assert value == 7
    assert thread_id != main_thread


@pytest.mark.asyncio
async def test_async_sink_honors_thread_executor():
    main_thread = threading.get_ident()
    sink = AsyncThreadSink()
    executor = RuntimeExecutor(
        source_sink_graph(FiniteSource([3]), sink, sink_executor="thread")
    )
    await executor.run()
    assert sink.items == [3]
    assert len(sink.thread_ids) == 1
    assert sink.thread_ids[0] != main_thread


@pytest.mark.asyncio
async def test_async_monitor_honors_thread_executor_and_records_success():
    main_thread = threading.get_ident()
    monitor = AsyncThreadMonitor()
    graph = source_sink_graph(FiniteSource([5]), CollectingSink())
    graph.add_node(
        RuntimeNode("monitor", NodeKind.MONITOR, monitor, executor="thread")
    )
    executor = RuntimeExecutor(graph)
    await executor.run()
    assert monitor.nodes == ["source"]
    assert len(monitor.thread_ids) == 1
    assert monitor.thread_ids[0] != main_thread
    assert executor.snapshot()["nodes"]["monitor"]["processed"] == 1


@pytest.mark.asyncio
async def test_thread_monitor_failure_preserves_monitor_culprit():
    graph = source_sink_graph(FiniteSource([11]), CollectingSink())
    graph.add_node(
        RuntimeNode(
            "monitor",
            NodeKind.MONITOR,
            FailingAsyncThreadMonitor(),
            executor="thread",
        )
    )
    executor = RuntimeExecutor(graph)
    with pytest.raises(RuntimeError, match="thread monitor rejected source"):
        await executor.run()
    assert executor.snapshot()["failure"] == {
        "node_id": "monitor",
        "error_type": "LookupError",
        "message": "thread monitor rejected source",
    }
