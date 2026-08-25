import asyncio

import numpy as np
import pytest

from neuros.contracts import SignalFrame, StreamDescriptor, WindowSpec
from neuros.models import EEGNetModel
from neuros.processing import SlidingWindowTransform
from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode


class _OneWindowSource:
    def __init__(self, data: np.ndarray, sample_rate_hz: float = 64.0) -> None:
        self.data = np.asarray(data, dtype=np.float32)
        self._descriptor = StreamDescriptor(
            stream_id="eeg",
            modality="eeg",
            sample_rate_hz=sample_rate_hz,
            channel_names=("C3", "C4"),
        )

    @property
    def descriptor(self) -> StreamDescriptor:
        return self._descriptor

    async def start(self) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def frames(self):
        yield SignalFrame(
            stream_id="eeg",
            sequence_id=0,
            data=self.data,
            sample_rate_hz=self._descriptor.sample_rate_hz,
            host_receive_time_ns=1_000_000_000,
            metadata={
                "axis_order": ("sample", "channel"),
                "channel_names": self._descriptor.channel_names,
            },
        )
        await asyncio.sleep(0)


@pytest.mark.asyncio
async def test_eegnet_runs_through_canonical_neural_window_runtime_path():
    rng = np.random.default_rng(7)
    train_x = rng.normal(size=(6, 2, 64)).astype(np.float32)
    train_y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)

    model = EEGNetModel(
        n_channels=2,
        n_classes=2,
        temporal_filters=2,
        depth_multiplier=1,
        separable_filters=4,
        temporal_kernel=15,
        separable_kernel=7,
        n_epochs=1,
        batch_size=2,
        random_state=7,
        device="cpu",
    )
    model.train(train_x, train_y)

    sample_major = rng.normal(size=(64, 2)).astype(np.float32)
    source = _OneWindowSource(sample_major)
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, source))
    graph.add_node(
        RuntimeNode(
            "window",
            NodeKind.TRANSFORM,
            SlidingWindowTransform(WindowSpec(64, 64), descriptor=source.descriptor),
        )
    )
    graph.add_node(RuntimeNode("decoder", NodeKind.DECODER, model))
    graph.connect(RuntimeEdge("source", "window", overflow="block"))
    graph.connect(RuntimeEdge("window", "decoder", overflow="block"))

    executor = RuntimeExecutor(graph)
    await executor.start()
    outputs = [output async for output in executor.outputs()]
    await executor.wait()

    assert len(outputs) == 1
    output = outputs[0]
    assert output.model_id == "EEGNetModel"
    assert output.probabilities is not None
    assert np.asarray(output.probabilities).shape == (2,)
    assert output.metadata["neuros_stream_id"] == "eeg"
    assert output.metadata["neuros_window_id"] == 0
    assert output.metadata["window_channel_names"] == ("C3", "C4")
    assert output.metadata["source_sequence_ids"] == (0,)
