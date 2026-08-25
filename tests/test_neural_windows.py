import asyncio

import numpy as np
import pytest

from neuros.contracts import (
    DecoderCapabilities,
    DecoderOutput,
    NeuralWindow,
    QualityFlag,
    SignalFrame,
    StreamDescriptor,
    TransformEmission,
    WindowSpec,
)
from neuros.processing import DiscontinuityPolicy, SlidingWindowTransform
from neuros.runtime import NodeKind, RuntimeEdge, RuntimeExecutor, RuntimeGraph, RuntimeNode


def _frame(
    sequence_id: int,
    data: np.ndarray,
    *,
    start_time_ns: int,
    sample_rate_hz: float = 10.0,
    quality: QualityFlag = QualityFlag.GOOD,
    axis_order: bool = True,
) -> SignalFrame:
    metadata = {"channel_names": ("C3", "C4")}
    if axis_order and np.asarray(data).ndim == 2:
        metadata["axis_order"] = ("sample", "channel")
    return SignalFrame(
        stream_id="eeg",
        sequence_id=sequence_id,
        data=np.asarray(data, dtype=np.float32),
        sample_rate_hz=sample_rate_hz,
        host_receive_time_ns=start_time_ns,
        quality=quality,
        metadata=metadata,
    )


def test_neural_window_contract_is_channel_major_and_decoder_batchable():
    window = NeuralWindow(
        stream_id="eeg",
        window_id=3,
        data=np.arange(8, dtype=np.float32).reshape(2, 4),
        sample_rate_hz=100.0,
        start_time_ns=1_000,
        end_time_ns=40_001_000,
        channel_names=("C3", "C4"),
        source_sequence_ids=(4, 5),
    )

    assert window.n_channels == 2
    assert window.n_samples == 4
    assert window.as_batch().shape == (1, 2, 4)
    assert window.metadata == {}

    with pytest.raises(ValueError, match="channels, time"):
        NeuralWindow(
            stream_id="eeg",
            window_id=0,
            data=np.ones(4),
            sample_rate_hz=100.0,
            start_time_ns=1,
            end_time_ns=2,
        )

    with pytest.raises(ValueError, match="NaN or infinite"):
        NeuralWindow(
            stream_id="eeg",
            window_id=0,
            data=np.array([[1.0, np.nan]]),
            sample_rate_hz=100.0,
            start_time_ns=1,
            end_time_ns=2,
        )


def test_sliding_window_transform_emits_all_windows_from_chunked_frames():
    descriptor = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=10.0,
        channel_names=("C3", "C4"),
    )
    transform = SlidingWindowTransform(
        WindowSpec(window_samples=4, stride_samples=2),
        descriptor=descriptor,
    )

    first = transform.transform(
        _frame(
            0,
            np.array([[0, 10], [1, 11], [2, 12], [3, 13]]),
            start_time_ns=1_000_000_000,
        )
    )
    assert isinstance(first, TransformEmission)
    assert len(first.items) == 1
    assert first.items[0].data.tolist() == [[0, 1, 2, 3], [10, 11, 12, 13]]
    assert first.items[0].source_sequence_ids == (0,)

    second = transform.transform(
        _frame(
            1,
            np.array([[4, 14], [5, 15], [6, 16], [7, 17]]),
            start_time_ns=1_400_000_000,
        )
    )
    assert isinstance(second, TransformEmission)
    assert len(second.items) == 2
    assert [window.window_id for window in second.items] == [1, 2]
    assert second.items[0].source_sequence_ids == (0, 1)
    assert second.items[1].source_sequence_ids == (1,)
    assert second.items[0].data.tolist() == [[2, 3, 4, 5], [12, 13, 14, 15]]
    assert second.items[1].data.tolist() == [[4, 5, 6, 7], [14, 15, 16, 17]]
    assert transform.pending_samples == 2


def test_window_transform_fails_closed_on_ambiguous_geometry_and_gaps():
    transform = SlidingWindowTransform(WindowSpec(4, 2))
    with pytest.raises(ValueError, match="axis_order"):
        transform.transform(
            _frame(
                0,
                np.ones((2, 2)),
                start_time_ns=1_000_000_000,
                axis_order=False,
            )
        )

    transform = SlidingWindowTransform(WindowSpec(4, 2))
    assert transform.transform(
        _frame(0, np.array([[0, 10], [1, 11]]), start_time_ns=1_000_000_000)
    ) is None
    with pytest.raises(ValueError, match="sequence gap"):
        transform.transform(
            _frame(2, np.array([[2, 12], [3, 13]]), start_time_ns=1_400_000_000)
        )


def test_window_transform_can_reset_explicitly_at_discontinuity():
    transform = SlidingWindowTransform(
        WindowSpec(4, 2), discontinuity=DiscontinuityPolicy.RESET
    )
    assert transform.transform(
        _frame(0, np.array([[0, 10], [1, 11]]), start_time_ns=1_000_000_000)
    ) is None
    assert transform.transform(
        _frame(2, np.array([[2, 12], [3, 13]]), start_time_ns=1_400_000_000)
    ) is None
    emitted = transform.transform(
        _frame(3, np.array([[4, 14], [5, 15]]), start_time_ns=1_600_000_000)
    )
    assert isinstance(emitted, TransformEmission)
    window = emitted.items[0]
    assert window.source_sequence_ids == (2, 3)
    assert window.metadata["discontinuity_count"] == 1
    assert transform.discontinuity_count == 1

    transform = SlidingWindowTransform(
        WindowSpec(2, 2), discontinuity=DiscontinuityPolicy.RESET
    )
    flagged = transform.transform(
        _frame(
            0,
            np.array([[0, 10], [1, 11]]),
            start_time_ns=1_000_000_000,
            quality=QualityFlag.DROPPED_SAMPLES,
        )
    )
    assert isinstance(flagged, TransformEmission)
    assert flagged.items[0].quality & QualityFlag.DROPPED_SAMPLES


class _ChunkSource:
    def __init__(self) -> None:
        self._descriptor = StreamDescriptor(
            stream_id="eeg",
            modality="eeg",
            sample_rate_hz=10.0,
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
        yield _frame(
            0,
            np.array([[0, 10], [1, 11], [2, 12], [3, 13]]),
            start_time_ns=1_000_000_000,
        )
        await asyncio.sleep(0)
        yield _frame(
            1,
            np.array([[4, 14], [5, 15], [6, 16], [7, 17]]),
            start_time_ns=1_400_000_000,
        )


class _WindowDecoder:
    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []

    @property
    def capabilities(self) -> DecoderCapabilities:
        return DecoderCapabilities(probabilities=False)

    def infer(self, X: np.ndarray) -> DecoderOutput:
        self.shapes.append(tuple(np.asarray(X).shape))
        return DecoderOutput(prediction=len(self.shapes) - 1, model_id="window-test")


@pytest.mark.asyncio
async def test_runtime_fans_out_windows_batches_decoder_input_and_binds_provenance():
    source = _ChunkSource()
    decoder = _WindowDecoder()
    graph = RuntimeGraph()
    graph.add_node(RuntimeNode("source", NodeKind.SOURCE, source))
    graph.add_node(
        RuntimeNode(
            "window",
            NodeKind.TRANSFORM,
            SlidingWindowTransform(WindowSpec(4, 2), descriptor=source.descriptor),
        )
    )
    graph.add_node(RuntimeNode("decoder", NodeKind.DECODER, decoder))
    graph.connect(RuntimeEdge("source", "window", overflow="block"))
    graph.connect(RuntimeEdge("window", "decoder", overflow="block"))

    executor = RuntimeExecutor(graph)
    await executor.start()
    outputs = [output async for output in executor.outputs()]
    await executor.wait()

    assert decoder.shapes == [(1, 2, 4), (1, 2, 4), (1, 2, 4)]
    assert [output.prediction for output in outputs] == [0, 1, 2]
    assert [output.metadata["neuros_window_id"] for output in outputs] == [0, 1, 2]
    assert outputs[1].metadata["source_sequence_ids"] == (0, 1)
    assert outputs[0].metadata["window_channel_names"] == ("C3", "C4")
    snapshot = executor.snapshot()
    assert snapshot["nodes"]["window"]["processed"] == 2
    assert snapshot["nodes"]["decoder"]["processed"] == 3
