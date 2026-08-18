import json
from pathlib import Path

import numpy as np
import pytest

from neuros.contracts import ClockDomain, QualityFlag, SignalFrame, StreamDescriptor
from neuros.recording import SessionArchiveReader, SessionArchiveWriter


def _frames():
    return [
        SignalFrame(
            stream_id="eeg",
            sequence_id=index,
            data=np.arange(6, dtype=np.float32).reshape(2, 3) + index,
            sample_rate_hz=250.0,
            host_receive_time_ns=1_000_000 + index,
            device_time_ns=2_000_000 + index,
            synchronized_time_ns=3_000_000 + index,
            clock_domain=ClockDomain.SYNCHRONIZED,
            quality=QualityFlag.ARTIFACT_SUSPECTED if index == 1 else QualityFlag.GOOD,
            metadata={"trial": index, "label": "left" if index % 2 == 0 else "right"},
        )
        for index in range(3)
    ]


@pytest.mark.asyncio
async def test_session_archive_round_trips_exact_frame_semantics(tmp_path: Path):
    descriptor = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=250.0,
        channel_names=("C3", "C4"),
        channel_types=("eeg", "eeg"),
        units=("uV", "uV"),
        device="synthetic",
        manufacturer="neurOS",
        clock_domain=ClockDomain.SYNCHRONIZED,
        metadata={"reference": "common-average"},
    )
    root = tmp_path / "session"
    writer = SessionArchiveWriter(
        root,
        session_id="s1",
        config={"schema_version": 1, "test": True},
        metadata={"subject": "anonymous"},
    )
    writer.register_stream(descriptor)
    original = _frames()
    for frame in original:
        await writer.write(frame)
    await writer.close(runtime_metrics={"state": "stopped", "dropped": 0})

    reader = SessionArchiveReader(root)
    assert reader.descriptor("eeg") == descriptor
    restored = list(reader.iter_frames("eeg"))
    assert len(restored) == len(original)
    for expected, actual in zip(original, restored):
        assert actual.sequence_id == expected.sequence_id
        np.testing.assert_array_equal(actual.data, expected.data)
        assert actual.sample_rate_hz == expected.sample_rate_hz
        assert actual.host_receive_time_ns == expected.host_receive_time_ns
        assert actual.device_time_ns == expected.device_time_ns
        assert actual.synchronized_time_ns == expected.synchronized_time_ns
        assert actual.clock_domain == expected.clock_domain
        assert actual.quality == expected.quality
        assert dict(actual.metadata) == dict(expected.metadata)

    summary = reader.summary()
    assert summary["streams"] == {"eeg": 3}
    assert summary["config_hash"]
    assert summary["runtime_metrics"]["dropped"] == 0
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["status"] == "complete"


@pytest.mark.asyncio
async def test_session_archive_detects_corrupted_frame_payload(tmp_path: Path):
    root = tmp_path / "session"
    writer = SessionArchiveWriter(root, session_id="s1")
    writer.register_stream(
        StreamDescriptor(stream_id="eeg", modality="eeg", sample_rate_hz=250.0)
    )
    await writer.write(_frames()[0])
    await writer.close()

    payload = next((root / "streams" / "eeg" / "frames").glob("*.npy"))
    raw = bytearray(payload.read_bytes())
    raw[-1] ^= 0xFF
    payload.write_bytes(bytes(raw))

    reader = SessionArchiveReader(root, verify_hashes=True)
    with pytest.raises(IOError, match="hash mismatch"):
        list(reader.iter_frames("eeg"))
