from __future__ import annotations

import asyncio

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
        self.started = False
        self.stopped = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False
        self.stopped = True

    async def frames(self):
        for value in self.values:
            await asyncio.sleep(0)
            yield value


class InfiniteSource:
    def __init__(self):
        self.started = False
        self.stopped = False
        self.sequence = 0

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False
        self.stopped = True

    async def frames(self):
        while True:
            await asyncio.sleep(0)
            self.sequence += 1
            yield self.sequence


class ParkedSource:
    """Emit one item, then remain live until cancellation or release."""

    def __init__(self):
        self.started = False
        self.stopped = False
        self.release = asyncio.Event()

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False
        self.stopped = True

    async def frames(self):
        yield 1
        await self.release.wait()


class CollectingSink:
    def __init__(self, target: int = 1):
        self.items = []
        self.target = target
        self.reached = asyncio.Event()

    async def write(self, item):
        self.items.append(item)
        if len(self.items) >= self.target:
            self.reached.set()


class HangingSink:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def write(self, item):
        self.entered.set()
        await self.release.wait()


class IdentityTransform:
    def transform(self, item):
        return item


class SlowTransform:
    async def transform(self, item):
        await asyncio.sleep(0.02)
        return item


class DelayedFailTransform:
    async def transform(self, item):
        await asyncio.sleep(0.02)
        raise ValueError("delayed qualified transform failure")


class FailTransform:
    def transform(self, item):
        raise ValueError("qualified transform failure")


class FailingMonitor:
    def update(self, payload):
        raise LookupError(f"monitor rejected {payload['node_id']}")


class FakeProcessWorker:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


def source_sink_graph(source, sink, *, capacity=2, overflow="block"):
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, source))
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, sink))
    graph.connect(
        RuntimeEdge(
            "source",
            "sink",
            capacity=capacity,
            overflow=overflow,
        )
    )
    return graph


def source_transform_sink_graph(
    source,
    transform,
    sink,
    *,
    source_capacity=2,
    source_overflow="block",
):
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, source))
    graph.add_node(RuntimeNode("transform", NodeKind.TRANSFORM, transform))
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, sink))
    graph.connect(
        RuntimeEdge(
            "source",
            "transform",
            capacity=source_capacity,
            overflow=source_overflow,
        )
    )
    graph.connect(RuntimeEdge("transform", "sink", capacity=2, overflow="block"))
    return graph


def assert_executor_tasks_terminal(executor: RuntimeExecutor) -> None:
    assert executor._completion_task is not None
    assert executor._completion_task.done()
    assert all(task.done() for task in executor._tasks.values())


async def collect_events(executor: RuntimeExecutor):
    return [event async for event in executor.events()]


@pytest.mark.asyncio
async def test_graceful_external_stop_drains_to_stopped_without_failure():
    source = InfiniteSource()
    sink = CollectingSink(target=3)
    executor = RuntimeExecutor(
        source_sink_graph(source, sink, capacity=8, overflow="block"),
        drain_timeout_s=0.5,
    )

    await executor.start()
    await asyncio.wait_for(sink.reached.wait(), timeout=1.0)
    await executor.stop()
    await executor.wait()

    snapshot = executor.snapshot()
    assert snapshot["state"] == "stopped"
    assert snapshot["failure"] is None
    assert source.stopped is True
    assert_executor_tasks_terminal(executor)
    events = await collect_events(executor)
    assert any(event.event == "runtime_draining" for event in events)
    assert events[-1].event == "runtime_stopped"
    assert not any(event.event == "runtime_drain_timeout" for event in events)


@pytest.mark.asyncio
async def test_drain_timeout_is_failed_with_pending_nodes_not_silent_stopped():
    source = InfiniteSource()
    sink = HangingSink()
    executor = RuntimeExecutor(
        source_sink_graph(source, sink, capacity=1, overflow="block"),
        drain_timeout_s=0.02,
    )

    await executor.start()
    await asyncio.wait_for(sink.entered.wait(), timeout=1.0)
    await executor.stop()

    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"]["node_id"] == "runtime"
    assert snapshot["failure"]["error_type"] == "RuntimeDrainTimeoutError"
    assert "runtime drain exceeded" in snapshot["failure"]["message"]
    assert "sink" in snapshot["failure"]["message"]
    assert source.stopped is True
    assert_executor_tasks_terminal(executor)

    with pytest.raises(RuntimeError, match="RuntimeDrainTimeoutError"):
        await executor.wait()

    events = await collect_events(executor)
    timeout_events = [event for event in events if event.event == "runtime_drain_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0].state.value == "failed"
    assert "sink" in timeout_events[0].metadata["pending_node_ids"]
    assert not any(event.event == "runtime_stopped" for event in events)


@pytest.mark.asyncio
async def test_operator_failure_preserves_culprit_and_cancels_peer_tasks():
    executor = RuntimeExecutor(
        source_transform_sink_graph(
            FiniteSource([1]),
            FailTransform(),
            CollectingSink(),
        )
    )

    with pytest.raises(RuntimeError, match="qualified transform failure"):
        await executor.run()

    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"] == {
        "node_id": "transform",
        "error_type": "ValueError",
        "message": "qualified transform failure",
    }
    assert snapshot["nodes"]["transform"]["failed"] == 1
    assert snapshot["nodes"]["source"]["failed"] == 0
    assert_executor_tasks_terminal(executor)
    events = await collect_events(executor)
    assert sum(event.event == "node_failed" for event in events) == 1
    assert not any(event.event == "runtime_stopped" for event in events)


@pytest.mark.asyncio
async def test_monitor_failure_is_attributed_to_observational_monitor():
    graph = source_sink_graph(FiniteSource([1]), CollectingSink())
    graph.add_node(RuntimeNode("monitor", NodeKind.MONITOR, FailingMonitor()))
    executor = RuntimeExecutor(graph)

    with pytest.raises(RuntimeError, match="monitor rejected source"):
        await executor.run()

    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"] == {
        "node_id": "monitor",
        "error_type": "LookupError",
        "message": "monitor rejected source",
    }
    assert snapshot["nodes"]["monitor"]["failed"] == 1
    assert snapshot["nodes"]["source"]["failed"] == 0
    assert_executor_tasks_terminal(executor)


@pytest.mark.asyncio
async def test_fail_overflow_is_runtime_failure_at_emitting_source():
    executor = RuntimeExecutor(
        source_transform_sink_graph(
            FiniteSource(range(8)),
            SlowTransform(),
            CollectingSink(),
            source_capacity=1,
            source_overflow="fail",
        )
    )

    with pytest.raises(RuntimeError, match="QueueFull"):
        await asyncio.wait_for(executor.run(), timeout=1.0)

    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"]["node_id"] == "source"
    assert snapshot["failure"]["error_type"] == "QueueFull"
    assert snapshot["edges"]["source->transform"]["overflow_policy"] == "fail"
    assert_executor_tasks_terminal(executor)


@pytest.mark.asyncio
async def test_executor_owned_process_workers_are_closed_on_failure_path():
    executor = RuntimeExecutor(
        source_transform_sink_graph(
            FiniteSource([1]),
            FailTransform(),
            CollectingSink(),
        )
    )
    fake_worker = FakeProcessWorker()
    executor._process_workers["owned-test-worker"] = fake_worker

    with pytest.raises(RuntimeError, match="qualified transform failure"):
        await executor.run()

    assert fake_worker.close_calls == 1
    assert_executor_tasks_terminal(executor)


@pytest.mark.asyncio
async def test_run_for_surfaces_early_failure_without_sleeping_full_duration():
    executor = RuntimeExecutor(
        source_transform_sink_graph(
            FiniteSource([1]),
            FailTransform(),
            CollectingSink(),
        )
    )

    with pytest.raises(RuntimeError, match="qualified transform failure"):
        await asyncio.wait_for(executor.run_for(30.0), timeout=1.0)

    assert executor.snapshot()["state"] == "failed"
    assert_executor_tasks_terminal(executor)
    leaked = [
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("neuros:")
    ]
    assert leaked == []


@pytest.mark.asyncio
async def test_unexpected_node_cancellation_is_failed_not_silent_stopped():
    source = ParkedSource()
    sink = CollectingSink(target=1)
    executor = RuntimeExecutor(
        source_sink_graph(source, sink, capacity=4, overflow="block"),
        drain_timeout_s=0.5,
    )

    await executor.start()
    await asyncio.wait_for(sink.reached.wait(), timeout=1.0)
    executor._tasks["sink"].cancel()

    with pytest.raises(RuntimeError, match="RuntimeUnexpectedCancellationError"):
        await asyncio.wait_for(executor.wait(), timeout=1.0)

    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"] == {
        "node_id": "sink",
        "error_type": "RuntimeUnexpectedCancellationError",
        "message": "runtime node task 'sink' was cancelled without shutdown authority",
    }
    assert snapshot["nodes"]["sink"]["failed"] == 1
    assert source.stopped is True
    assert_executor_tasks_terminal(executor)
    events = await collect_events(executor)
    assert sum(event.event == "node_cancelled" for event in events) == 1
    assert not any(event.event == "runtime_stopped" for event in events)


@pytest.mark.asyncio
async def test_failure_peer_cancellation_cannot_deadlock_on_saturated_stop_queue():
    source = InfiniteSource()
    executor = RuntimeExecutor(
        source_transform_sink_graph(
            source,
            DelayedFailTransform(),
            CollectingSink(),
            source_capacity=1,
            source_overflow="block",
        ),
        drain_timeout_s=0.5,
    )

    with pytest.raises(RuntimeError, match="delayed qualified transform failure"):
        await asyncio.wait_for(executor.run(), timeout=1.0)

    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"]["node_id"] == "transform"
    assert snapshot["failure"]["error_type"] == "ValueError"
    assert source.stopped is True
    assert_executor_tasks_terminal(executor)
    leaked = [
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("neuros:")
    ]
    assert leaked == []
