from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import pickle
import threading
import time
from pathlib import Path

import pytest

from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode
from neuros.runtime.process_worker import PersistentProcessWorker, ProcessWorkerProtocolError


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


class CounterTransform:
    def __init__(self):
        self.calls = 0

    def transform(self, item):
        self.calls += 1
        return self.calls, item, os.getpid()


class AsyncPidTransform:
    async def transform(self, item):
        await asyncio.sleep(0)
        return os.getpid(), item


class FailingTransform:
    def transform(self, item):
        raise LookupError(f"process rejected {item}")


class SleepTransform:
    def __init__(self, pid_path: Path, delay_s: float = 10.0):
        self.pid_path = pid_path
        self.delay_s = delay_s

    def transform(self, item):
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(self.delay_s)
        return item


class DelayedFailTransform:
    def __init__(self, pid_path: Path, delay_s: float = 0.2):
        self.pid_path = pid_path
        self.delay_s = delay_s

    def transform(self, item):
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(self.delay_s)
        raise RuntimeError("peer process failed")


class CrashDecoder:
    def __init__(self, pid_path: Path):
        self.pid_path = pid_path

    def infer(self, item):
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        os._exit(17)


class SleepDecoder:
    def __init__(self, pid_path: Path):
        self.pid_path = pid_path

    def infer(self, item):
        self.pid_path.write_text(str(os.getpid()), encoding="utf-8")
        time.sleep(10.0)
        return 1


class IdentityTransform:
    def transform(self, item):
        return item


class UnpicklableOperator:
    def __init__(self):
        self.lock = threading.Lock()

    def transform(self, item):
        return item


class UnpicklableResultTransform:
    def transform(self, item):
        return lambda value: (item, value)


class FileSink:
    def __init__(self, path: Path):
        self.path = path

    def write(self, item):
        self.path.write_text(f"{os.getpid()}:{item}", encoding="utf-8")


class FileMonitor:
    def __init__(self, path: Path):
        self.path = path

    def update(self, payload):
        self.path.write_text(
            f"{os.getpid()}:{payload['node_id']}", encoding="utf-8"
        )


def _source_transform_sink(transform, *, timeout_s=2.0, values=(1,)):
    sink = CollectingSink()
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, FiniteSource(values)))
    graph.add_node(
        RuntimeNode(
            "transform",
            NodeKind.TRANSFORM,
            transform,
            executor="process",
            execution_timeout_s=timeout_s,
        )
    )
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, sink))
    graph.connect(RuntimeEdge("source", "transform", overflow="block"))
    graph.connect(RuntimeEdge("transform", "sink", overflow="block"))
    return graph, sink


def _source_decoder(decoder, *, timeout_s):
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, FiniteSource((1,))))
    graph.add_node(
        RuntimeNode(
            "decoder",
            NodeKind.DECODER,
            decoder,
            executor="process",
            execution_timeout_s=timeout_s,
        )
    )
    graph.connect(RuntimeEdge("source", "decoder", overflow="block"))
    return graph


def _active_child_pids():
    return {process.pid for process in mp.active_children() if process.pid is not None}


async def _wait_for_path(path: Path, timeout_s: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while not path.exists():
        if asyncio.get_running_loop().time() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        await asyncio.sleep(0.01)


def test_process_timeout_is_explicit_and_separate_from_latency_slo():
    with pytest.raises(ValueError, match="execution_timeout_s"):
        RuntimeNode(
            "transform",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            executor="process",
            latency_budget_ms=0.001,
        )

    node = RuntimeNode(
        "transform",
        NodeKind.TRANSFORM,
        IdentityTransform(),
        executor="process",
        latency_budget_ms=0.001,
        execution_timeout_s=0.5,
    )
    assert node.latency_budget_ms == pytest.approx(0.001)
    assert node.execution_timeout_s == pytest.approx(0.5)

    with pytest.raises(ValueError, match="only authoritative"):
        RuntimeNode(
            "transform",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            executor="thread",
            execution_timeout_s=0.5,
        )


@pytest.mark.asyncio
async def test_process_worker_preserves_operator_state_and_semantic_receipts():
    graph, sink = _source_transform_sink(CounterTransform(), values=(10, 20, 30))
    snapshot = await RuntimeExecutor(graph).run()

    assert [(calls, item) for calls, item, _ in sink.items] == [
        (1, 10),
        (2, 20),
        (3, 30),
    ]
    worker_pids = {pid for _, _, pid in sink.items}
    assert len(worker_pids) == 1
    assert os.getpid() not in worker_pids
    assert worker_pids.isdisjoint(_active_child_pids())
    assert snapshot["process_receipts"]["transform"] == [
        {
            "node_id": "transform",
            "generation": 0,
            "request_id": request_id,
            "outcome": "success",
            "remote_error_type": None,
        }
        for request_id in (1, 2, 3)
    ]


@pytest.mark.asyncio
async def test_async_transform_runs_coroutine_inside_persistent_process():
    graph, sink = _source_transform_sink(AsyncPidTransform())
    await RuntimeExecutor(graph).run()
    pid, item = sink.items[0]
    assert item == 1
    assert pid != os.getpid()


@pytest.mark.asyncio
async def test_process_sink_and_monitor_honor_declared_domain(tmp_path):
    sink_path = tmp_path / "sink.txt"
    monitor_path = tmp_path / "monitor.txt"
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, FiniteSource((7,))))
    graph.add_node(
        RuntimeNode(
            "sink",
            NodeKind.SINK,
            FileSink(sink_path),
            executor="process",
            execution_timeout_s=2.0,
        )
    )
    graph.add_node(
        RuntimeNode(
            "monitor",
            NodeKind.MONITOR,
            FileMonitor(monitor_path),
            executor="process",
            execution_timeout_s=2.0,
        )
    )
    graph.connect(RuntimeEdge("source", "sink", overflow="block"))

    snapshot = await RuntimeExecutor(graph).run()
    sink_pid, sink_item = sink_path.read_text(encoding="utf-8").split(":", 1)
    monitor_pid, monitor_node = monitor_path.read_text(encoding="utf-8").split(":", 1)
    assert int(sink_pid) != os.getpid()
    assert int(monitor_pid) != os.getpid()
    assert sink_item == "7"
    assert monitor_node == "source"
    assert snapshot["process_receipts"]["sink"][0]["outcome"] == "success"
    assert snapshot["process_receipts"]["monitor"][0]["outcome"] == "success"


@pytest.mark.asyncio
async def test_remote_exception_preserves_exact_node_and_remote_error_type():
    graph, _ = _source_transform_sink(FailingTransform())
    executor = RuntimeExecutor(graph)
    with pytest.raises(RuntimeError, match="LookupError: process rejected 1"):
        await executor.run()

    assert executor.snapshot()["failure"] == {
        "node_id": "transform",
        "error_type": "LookupError",
        "message": "process rejected 1",
    }
    receipt = executor.snapshot()["process_receipts"]["transform"][-1]
    assert receipt["outcome"] == "error"
    assert receipt["remote_error_type"] == "LookupError"


@pytest.mark.asyncio
async def test_hard_timeout_terminates_direct_child_and_admits_no_output(tmp_path):
    pid_path = tmp_path / "timeout.pid"
    executor = RuntimeExecutor(_source_decoder(SleepDecoder(pid_path), timeout_s=0.1))
    await executor.start()
    with pytest.raises(RuntimeError, match="ProcessWorkerTimeoutError"):
        await executor.wait()
    await _wait_for_path(pid_path)
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    assert child_pid not in _active_child_pids()
    assert [item async for item in executor.outputs()] == []
    assert executor.snapshot()["process_receipts"]["decoder"][-1]["outcome"] == "timeout"


@pytest.mark.asyncio
async def test_asyncio_cancellation_terminates_direct_child(tmp_path):
    pid_path = tmp_path / "cancel.pid"
    graph, _ = _source_transform_sink(SleepTransform(pid_path), timeout_s=5.0)
    executor = RuntimeExecutor(graph)
    await executor.start()
    await _wait_for_path(pid_path)
    child_pid = int(pid_path.read_text(encoding="utf-8"))
    executor._tasks["transform"].cancel()

    with pytest.raises(RuntimeError, match="RuntimeUnexpectedCancellationError"):
        await executor.wait()
    assert child_pid not in _active_child_pids()
    assert executor.snapshot()["process_receipts"]["transform"][-1]["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_child_crash_becomes_explicit_failure_and_admits_no_output(tmp_path):
    pid_path = tmp_path / "crash.pid"
    executor = RuntimeExecutor(_source_decoder(CrashDecoder(pid_path), timeout_s=2.0))
    with pytest.raises(RuntimeError, match="ProcessWorkerCrashedError"):
        await executor.run()

    child_pid = int(pid_path.read_text(encoding="utf-8"))
    assert child_pid not in _active_child_pids()
    assert [item async for item in executor.outputs()] == []
    assert executor.snapshot()["process_receipts"]["decoder"][-1]["outcome"] == "crashed"


@pytest.mark.asyncio
async def test_stale_response_identity_is_rejected_and_worker_terminated():
    worker = PersistentProcessWorker(
        "transform", CounterTransform(), execution_timeout_s=1.0
    )
    await worker.heartbeat()

    def stale_response(timeout_s, request_id):
        return {
            "protocol": 1,
            "kind": "result",
            "node_id": "transform",
            "generation": 0,
            "request_id": request_id - 1,
            "result": pickle.dumps("stale"),
        }

    worker._recv = stale_response
    with pytest.raises(ProcessWorkerProtocolError, match="stale or mismatched"):
        await worker.invoke("transform", 1)
    assert not worker.is_alive
    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "protocol_error"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operator", "values", "expected"),
    [
        (UnpicklableOperator(), (1,), "operator/start serialization failed"),
        (IdentityTransform(), (lambda value: value,), "input/request serialization failed"),
        (UnpicklableResultTransform(), (1,), "result serialization failed"),
    ],
)
async def test_process_serialization_failures_fail_closed(operator, values, expected):
    graph, _ = _source_transform_sink(operator, values=values)
    executor = RuntimeExecutor(graph)
    with pytest.raises(RuntimeError, match="ProcessWorkerSerializationError"):
        await executor.run()
    failure = executor.snapshot()["failure"]
    assert failure["node_id"] == "transform"
    assert expected in failure["message"]
    assert executor.snapshot()["process_receipts"]["transform"][-1]["outcome"] == "serialization_error"


@pytest.mark.asyncio
async def test_runtime_failure_terminates_every_executor_owned_worker(tmp_path):
    slow_pid_path = tmp_path / "slow.pid"
    fail_pid_path = tmp_path / "fail.pid"
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, FiniteSource((1,))))
    graph.add_node(
        RuntimeNode(
            "slow",
            NodeKind.TRANSFORM,
            SleepTransform(slow_pid_path),
            executor="process",
            execution_timeout_s=5.0,
        )
    )
    graph.add_node(
        RuntimeNode(
            "fail",
            NodeKind.TRANSFORM,
            DelayedFailTransform(fail_pid_path),
            executor="process",
            execution_timeout_s=5.0,
        )
    )
    graph.add_node(RuntimeNode("slow_sink", NodeKind.SINK, CollectingSink()))
    graph.add_node(RuntimeNode("fail_sink", NodeKind.SINK, CollectingSink()))
    graph.connect(RuntimeEdge("source", "slow", overflow="block"))
    graph.connect(RuntimeEdge("source", "fail", overflow="block"))
    graph.connect(RuntimeEdge("slow", "slow_sink", overflow="block"))
    graph.connect(RuntimeEdge("fail", "fail_sink", overflow="block"))

    executor = RuntimeExecutor(graph)
    with pytest.raises(RuntimeError, match="peer process failed"):
        await executor.run()

    await _wait_for_path(slow_pid_path)
    await _wait_for_path(fail_pid_path)
    child_pids = {
        int(slow_pid_path.read_text(encoding="utf-8")),
        int(fail_pid_path.read_text(encoding="utf-8")),
    }
    assert child_pids.isdisjoint(_active_child_pids())
    assert executor.snapshot()["failure"]["node_id"] == "fail"
