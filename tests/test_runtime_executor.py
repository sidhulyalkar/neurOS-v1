import asyncio

import numpy as np
import pytest

from neuros.contracts import DecoderCapabilities, DecoderOutput, SignalFrame, StreamDescriptor
from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode


class FiniteSource:
    def __init__(self, stream_id: str, values: list[float]):
        self._descriptor = StreamDescriptor(
            stream_id=stream_id,
            modality="eeg",
            sample_rate_hz=100.0,
            channel_names=("ch0",),
        )
        self.values = values
        self.started = False

    @property
    def descriptor(self):
        return self._descriptor

    async def start(self):
        self.started = True

    async def stop(self):
        self.started = False

    async def frames(self):
        for index, value in enumerate(self.values):
            await asyncio.sleep(0)
            yield SignalFrame(
                stream_id=self._descriptor.stream_id,
                sequence_id=index,
                data=np.array([value], dtype=np.float32),
                sample_rate_hz=100.0,
                host_receive_time_ns=1_000 + index,
            )


class Scale:
    def __init__(self, factor: float):
        self.factor = factor

    def transform(self, frame: SignalFrame):
        from dataclasses import replace

        return replace(frame, data=frame.data * self.factor)


class Decoder:
    @property
    def capabilities(self):
        return DecoderCapabilities(probabilities=True)

    def infer(self, X):
        score = float(np.asarray(X).reshape(-1)[0])
        return DecoderOutput(prediction=int(score >= 2.0), confidence=0.75)


class Sink:
    def __init__(self):
        self.items = []

    async def write(self, item):
        self.items.append(item)


class FailTransform:
    def transform(self, item):
        raise ValueError("synthetic failure")


def build_graph(source, transform, decoder, sink=None):
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, source))
    graph.add_node(RuntimeNode("transform", NodeKind.TRANSFORM, transform))
    graph.add_node(RuntimeNode("decoder", NodeKind.DECODER, decoder))
    graph.connect(RuntimeEdge("source", "transform", capacity=2, overflow="block"))
    graph.connect(RuntimeEdge("transform", "decoder", capacity=2, overflow="block"))
    if sink is not None:
        graph.add_node(RuntimeNode("sink", NodeKind.SINK, sink))
        graph.connect(RuntimeEdge("decoder", "sink", capacity=2, overflow="block"))
    return graph


@pytest.mark.asyncio
async def test_runtime_executor_runs_finite_graph_and_taps_decoder_outputs():
    sink = Sink()
    executor = RuntimeExecutor(
        build_graph(FiniteSource("a", [0.5, 1.0, 2.0]), Scale(2.0), Decoder(), sink)
    )
    await executor.start()
    outputs = [output async for output in executor.outputs()]
    await executor.wait()

    assert [output.prediction for output in outputs] == [0, 1, 1]
    assert len(sink.items) == 3
    snapshot = executor.snapshot()
    assert snapshot["state"] == "stopped"
    assert snapshot["nodes"]["decoder"]["processed"] == 3
    assert snapshot["edges"]["source->transform"]["dropped"] == 0


@pytest.mark.asyncio
async def test_runtime_executor_fuses_multiple_signal_sources():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("a", NodeKind.SOURCE, FiniteSource("a", [1.0, 2.0])))
    graph.add_node(RuntimeNode("b", NodeKind.SOURCE, FiniteSource("b", [3.0, 4.0])))
    graph.add_node(RuntimeNode("fusion", NodeKind.FUSION, None))
    graph.add_node(RuntimeNode("decoder", NodeKind.DECODER, Decoder()))
    graph.connect(RuntimeEdge("a", "fusion", overflow="block"))
    graph.connect(RuntimeEdge("b", "fusion", overflow="block"))
    graph.connect(RuntimeEdge("fusion", "decoder", overflow="block"))

    executor = RuntimeExecutor(graph)
    await executor.start()
    outputs = [output async for output in executor.outputs()]
    await executor.wait()
    assert outputs
    assert executor.snapshot()["nodes"]["fusion"]["processed"] >= 1


@pytest.mark.asyncio
async def test_runtime_executor_propagates_node_failure():
    executor = RuntimeExecutor(
        build_graph(FiniteSource("a", [1.0]), FailTransform(), Decoder())
    )
    with pytest.raises(RuntimeError, match="synthetic failure"):
        await executor.run()
    snapshot = executor.snapshot()
    assert snapshot["state"] == "failed"
    assert snapshot["failure"]["node_id"] == "transform"


def test_runtime_graph_rejects_cycles():
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("a", NodeKind.TRANSFORM, Scale(1.0)))
    graph.add_node(RuntimeNode("b", NodeKind.TRANSFORM, Scale(1.0)))
    # Use direct edge construction so validation catches both the cycle and
    # invalid transform topology in one deterministic place.
    graph.edges.extend([RuntimeEdge("a", "b"), RuntimeEdge("b", "a")])
    with pytest.raises(ValueError, match="acyclic"):
        graph.validate()
