from __future__ import annotations

import asyncio

import pytest

from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode


class StopBlockingSource:
    """Source whose first cancellation enters a deliberately stalled stop()."""

    def __init__(self):
        self.started = asyncio.Event()
        self.stop_entered = asyncio.Event()
        self.frame_release = asyncio.Event()
        self.stop_release = asyncio.Event()

    async def start(self):
        self.started.set()

    async def stop(self):
        self.stop_entered.set()
        await self.stop_release.wait()

    async def frames(self):
        await self.frame_release.wait()
        yield 1


class Sink:
    async def write(self, item):
        return None


@pytest.mark.asyncio
async def test_cancelled_fusion_reaps_owned_queue_get_tasks():
    left = StopBlockingSource()
    right = StopBlockingSource()

    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("left", NodeKind.SOURCE, left))
    graph.add_node(RuntimeNode("right", NodeKind.SOURCE, right))
    graph.add_node(RuntimeNode("fusion", NodeKind.FUSION, None))
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, Sink()))
    graph.connect(RuntimeEdge("left", "fusion", capacity=1, overflow="block"))
    graph.connect(RuntimeEdge("right", "fusion", capacity=1, overflow="block"))
    graph.connect(RuntimeEdge("fusion", "sink", capacity=1, overflow="block"))

    executor = RuntimeExecutor(graph, drain_timeout_s=0.02)
    await executor.start()
    await asyncio.wait_for(left.started.wait(), timeout=1.0)
    await asyncio.wait_for(right.started.wait(), timeout=1.0)

    # Let the fusion node establish its temporary queue.get() children before
    # source cancellation stalls inside stop().
    await asyncio.sleep(0.01)
    await executor.stop()

    assert left.stop_entered.is_set()
    assert right.stop_entered.is_set()
    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"]["error_type"] == "RuntimeDrainTimeoutError"
    assert "fusion" in snapshot["failure"]["message"]

    with pytest.raises(RuntimeError, match="RuntimeDrainTimeoutError"):
        await executor.wait()

    assert executor._completion_task is not None
    assert executor._completion_task.done()
    assert all(task.done() for task in executor._tasks.values())

    leaked = [
        task.get_name()
        for task in asyncio.all_tasks()
        if task is not asyncio.current_task()
        and not task.done()
        and task.get_name().startswith("neuros:")
    ]
    assert leaked == []
