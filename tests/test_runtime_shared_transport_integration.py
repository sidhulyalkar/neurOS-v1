from __future__ import annotations

import asyncio
import time
from dataclasses import replace

import numpy as np
import pytest

from neuros.contracts import DecoderOutput, NeuralWindow, SignalFrame, TransformEmission
from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode
from neuros.runtime.shared_process_worker import SharedMemoryProcessWorker


class ItemsSource:
    def __init__(self, items):
        self.items = list(items)
        self.started = False

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    async def frames(self):
        for item in self.items:
            await asyncio.sleep(0)
            yield item


class CollectSink:
    def __init__(self):
        self.items = []

    async def write(self, item):
        self.items.append(item)


class FrameScaleTransform:
    def __init__(self, factor: float):
        self.factor = float(factor)

    def transform(self, frame: SignalFrame):
        return replace(frame, data=np.asarray(frame.data) * self.factor)


class BatchShapeDecoder:
    def infer(self, X):
        array = np.asarray(X)
        return DecoderOutput(
            prediction=int(float(array.mean()) >= 0.0),
            confidence=0.9,
            metadata={
                "observed_shape": tuple(int(value) for value in array.shape),
                "observed_sum": float(array.sum()),
            },
        )


class PairEmitter:
    def transform(self, frame: SignalFrame):
        return TransformEmission(
            (
                replace(frame, data=np.asarray(frame.data) + 1.0),
                replace(frame, data=np.asarray(frame.data) + 2.0),
            )
        )


class FileSink:
    def __init__(self, path: str):
        self.path = path

    def write(self, item):
        if isinstance(item, SignalFrame):
            value = float(np.asarray(item.data).reshape(-1)[0])
            line = f"{item.stream_id}:{item.sequence_id}:{value}\n"
        else:
            line = f"{type(item).__name__}\n"
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)


class FileMonitor:
    def __init__(self, path: str):
        self.path = path

    def update(self, payload):
        line = (
            f"{payload['node_id']}:{payload['kind']}:"
            f"{type(payload['item']).__name__}\n"
        )
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(line)


class IdentityTransform:
    def transform(self, item):
        return item


class DelayedFailTransform:
    def transform(self, item):
        time.sleep(0.5)
        raise ValueError("shared runtime synthetic failure")


class SlowTransform:
    def transform(self, item):
        time.sleep(5.0)
        return item


def _frame(sequence_id: int = 0, value: float = 1.0) -> SignalFrame:
    return SignalFrame(
        stream_id="eeg",
        sequence_id=sequence_id,
        data=np.array([value, value + 0.5], dtype=np.float32),
        sample_rate_hz=250.0,
        host_receive_time_ns=1_000 + sequence_id,
        metadata={"channel_names": ("C3", "C4")},
    )


def _window() -> NeuralWindow:
    return NeuralWindow(
        stream_id="eeg",
        window_id=17,
        data=np.arange(8, dtype=np.float32).reshape(2, 4),
        sample_rate_hz=250.0,
        start_time_ns=1_000,
        end_time_ns=16_001_000,
        channel_names=("C3", "C4"),
        source_sequence_ids=(10, 11),
        metadata={"trial": "shared-runtime"},
    )


def _shared_node(
    node_id: str,
    kind: NodeKind,
    operator,
    *,
    request_capacity: int = 64 * 1024,
    response_capacity: int = 64 * 1024,
    timeout_s: float = 3.0,
) -> RuntimeNode:
    return RuntimeNode(
        node_id,
        kind,
        operator,
        executor="process",
        execution_timeout_s=timeout_s,
        process_transport="shared_memory",
        process_request_capacity_bytes=request_capacity,
        process_response_capacity_bytes=response_capacity,
    )


def test_shared_memory_runtime_configuration_is_explicit_and_fail_closed():
    with pytest.raises(ValueError, match="positive process_request_capacity_bytes"):
        RuntimeNode(
            "missing-capacity",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            executor="process",
            execution_timeout_s=1.0,
            process_transport="shared_memory",
            process_response_capacity_bytes=4096,
        )

    with pytest.raises(ValueError, match="only valid for executor='process'"):
        RuntimeNode(
            "inline-shared",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            process_transport="shared_memory",
            process_request_capacity_bytes=4096,
            process_response_capacity_bytes=4096,
        )

    with pytest.raises(ValueError, match="only valid for process_transport='shared_memory'"):
        RuntimeNode(
            "pickle-capacity",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            executor="process",
            execution_timeout_s=1.0,
            process_request_capacity_bytes=4096,
        )


@pytest.mark.asyncio
async def test_runtime_shared_transform_round_trips_signal_frames_and_closes_worker():
    sink = CollectSink()
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, ItemsSource([_frame(0, 1.0), _frame(1, 2.0)])))
    graph.add_node(_shared_node("transform", NodeKind.TRANSFORM, FrameScaleTransform(3.0)))
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, sink))
    graph.connect(RuntimeEdge("source", "transform", overflow="block"))
    graph.connect(RuntimeEdge("transform", "sink", overflow="block"))

    executor = RuntimeExecutor(graph)
    snapshot = await executor.run()

    assert [item.sequence_id for item in sink.items] == [0, 1]
    assert np.allclose(sink.items[0].data, np.array([3.0, 4.5], dtype=np.float32))
    assert np.allclose(sink.items[1].data, np.array([6.0, 7.5], dtype=np.float32))
    assert snapshot["state"] == "stopped"
    assert [receipt["outcome"] for receipt in snapshot["process_receipts"]["transform"]] == [
        "success",
        "success",
    ]
    worker = executor._process_workers["transform"]
    assert isinstance(worker, SharedMemoryProcessWorker)
    assert not worker.is_alive
    assert worker.shared_memory_names is None


@pytest.mark.asyncio
async def test_runtime_shared_decoder_preserves_neural_window_batching_and_provenance():
    window = _window()
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, ItemsSource([window])))
    graph.add_node(_shared_node("decoder", NodeKind.DECODER, BatchShapeDecoder()))
    graph.connect(RuntimeEdge("source", "decoder", overflow="block"))

    executor = RuntimeExecutor(graph)
    await executor.start()
    outputs = [output async for output in executor.outputs()]
    await executor.wait()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.metadata["observed_shape"] == (1, 2, 4)
    assert output.metadata["observed_sum"] == pytest.approx(float(window.data.sum()))
    assert output.metadata["neuros_stream_id"] == "eeg"
    assert output.metadata["neuros_window_id"] == 17
    assert output.metadata["window_channel_names"] == ("C3", "C4")
    assert output.metadata["source_sequence_ids"] == (10, 11)
    assert executor.snapshot()["process_receipts"]["decoder"][0]["outcome"] == "success"


@pytest.mark.asyncio
async def test_runtime_shared_transform_emission_preserves_fanout_semantics():
    sink = CollectSink()
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, ItemsSource([_frame()])))
    graph.add_node(_shared_node("emit", NodeKind.TRANSFORM, PairEmitter()))
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, sink))
    graph.connect(RuntimeEdge("source", "emit", overflow="block"))
    graph.connect(RuntimeEdge("emit", "sink", overflow="block"))

    snapshot = await RuntimeExecutor(graph).run()

    assert snapshot["state"] == "stopped"
    assert len(sink.items) == 2
    assert np.allclose(sink.items[0].data, np.array([2.0, 2.5], dtype=np.float32))
    assert np.allclose(sink.items[1].data, np.array([3.0, 3.5], dtype=np.float32))


@pytest.mark.asyncio
async def test_runtime_shared_process_sink_executes_side_effect_and_closes(tmp_path):
    destination = tmp_path / "sink.txt"
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, ItemsSource([_frame(3, 4.0)])))
    graph.add_node(_shared_node("sink", NodeKind.SINK, FileSink(str(destination))))
    graph.connect(RuntimeEdge("source", "sink", overflow="block"))

    executor = RuntimeExecutor(graph)
    snapshot = await executor.run()

    assert destination.read_text(encoding="utf-8") == "eeg:3:4.0\n"
    assert snapshot["process_receipts"]["sink"][0]["outcome"] == "success"
    assert not executor._process_workers["sink"].is_alive
    assert executor._process_workers["sink"].shared_memory_names is None


@pytest.mark.asyncio
async def test_runtime_shared_process_monitor_observes_canonical_payload(tmp_path):
    destination = tmp_path / "monitor.txt"
    sink = CollectSink()
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, ItemsSource([_frame()])))
    graph.add_node(RuntimeNode("sink", NodeKind.SINK, sink))
    graph.add_node(_shared_node("monitor", NodeKind.MONITOR, FileMonitor(str(destination))))
    graph.connect(RuntimeEdge("source", "sink", overflow="block"))

    executor = RuntimeExecutor(graph)
    snapshot = await executor.run()

    assert destination.read_text(encoding="utf-8") == "source:source:SignalFrame\n"
    assert snapshot["process_receipts"]["monitor"][0]["outcome"] == "success"
    assert not executor._process_workers["monitor"].is_alive
    assert executor._process_workers["monitor"].shared_memory_names is None


@pytest.mark.asyncio
async def test_runtime_attributes_shared_request_capacity_failure_to_owning_node():
    graph = RuntimeGraph()
    graph.add_node(
        RuntimeNode(
            "source",
            NodeKind.SOURCE,
            ItemsSource([np.arange(100, dtype=np.float64)]),
        )
    )
    graph.add_node(
        _shared_node(
            "transform",
            NodeKind.TRANSFORM,
            IdentityTransform(),
            request_capacity=64,
            response_capacity=4096,
        )
    )
    graph.connect(RuntimeEdge("source", "transform", overflow="block"))

    executor = RuntimeExecutor(graph)
    with pytest.raises(RuntimeError, match="ProcessWorkerTransportError"):
        await executor.run()

    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"]["node_id"] == "transform"
    assert snapshot["failure"]["error_type"] == "ProcessWorkerTransportError"
    assert snapshot["process_receipts"]["transform"][-1]["outcome"] == "transport_error"
    worker = executor._process_workers["transform"]
    assert not worker.is_alive
    assert worker.shared_memory_names is None


@pytest.mark.asyncio
async def test_runtime_shared_failure_cancels_peer_worker_without_overwriting_culprit():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, ItemsSource([np.array([1.0], dtype=np.float32)])))
    graph.add_node(
        _shared_node(
            "fail",
            NodeKind.TRANSFORM,
            DelayedFailTransform(),
            timeout_s=3.0,
        )
    )
    graph.add_node(
        _shared_node(
            "slow",
            NodeKind.TRANSFORM,
            SlowTransform(),
            timeout_s=10.0,
        )
    )
    graph.connect(RuntimeEdge("source", "fail", overflow="block"))
    graph.connect(RuntimeEdge("source", "slow", overflow="block"))

    executor = RuntimeExecutor(graph)
    with pytest.raises(RuntimeError, match="shared runtime synthetic failure"):
        await executor.run()

    snapshot = executor.snapshot()
    assert snapshot["failure"]["node_id"] == "fail"
    assert snapshot["failure"]["error_type"] == "ValueError"
    assert snapshot["process_receipts"]["fail"][-1]["outcome"] == "error"
    assert snapshot["process_receipts"]["fail"][-1]["remote_error_type"] == "ValueError"
    assert snapshot["process_receipts"]["slow"][-1]["outcome"] == "cancelled"
    for worker in executor._process_workers.values():
        assert not worker.is_alive
        assert worker.shared_memory_names is None
