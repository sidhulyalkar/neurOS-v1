import numpy as np
import pytest

from neuros.contracts import SignalFrame, StreamDescriptor
from neuros.recording import FrameRecorder, ReplaySource


def make_frame(sequence_id: int, timestamp_ns: int) -> SignalFrame:
    return SignalFrame(
        stream_id="eeg",
        sequence_id=sequence_id,
        data=np.full((2, 2), sequence_id),
        sample_rate_hz=250.0,
        host_receive_time_ns=timestamp_ns,
    )


@pytest.mark.asyncio
async def test_replay_preserves_order_and_timestamps():
    descriptor = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=250.0,
    )
    expected = [make_frame(0, 100), make_frame(1, 200), make_frame(2, 300)]
    source = ReplaySource(descriptor, expected)
    await source.start()
    actual = [frame async for frame in source.frames()]
    await source.stop()
    assert [frame.sequence_id for frame in actual] == [0, 1, 2]
    assert [frame.timestamp_ns for frame in actual] == [100, 200, 300]


@pytest.mark.asyncio
async def test_frame_recorder_snapshots_immutable_sequence():
    recorder = FrameRecorder()
    await recorder.write(make_frame(0, 100))
    snapshot = recorder.snapshot()
    await recorder.write(make_frame(1, 200))
    assert len(snapshot) == 1
    assert len(recorder.snapshot()) == 2
