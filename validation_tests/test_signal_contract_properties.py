from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from neuros.contracts import (
    ClockDomain,
    SignalFrame,
    StreamDescriptor,
    validate_frame_against_descriptor,
)
from neuros.recording import SessionArchiveReader, SessionArchiveWriter


@st.composite
def finite_signal_arrays(draw):
    channels = draw(st.integers(min_value=1, max_value=8))
    samples = draw(st.integers(min_value=1, max_value=24))
    values = draw(
        st.lists(
            st.floats(
                min_value=-1_000.0,
                max_value=1_000.0,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=channels * samples,
            max_size=channels * samples,
        )
    )
    return np.asarray(values, dtype=np.float32).reshape(channels, samples)


@given(data=finite_signal_arrays())
@settings(max_examples=24, deadline=None)
def test_signal_contract_archive_roundtrip_and_payload_tamper_fail_closed(data: np.ndarray):
    channels = int(data.shape[0])
    names = tuple(f"ch{index}" for index in range(channels))
    descriptor = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=250.0,
        channel_names=names,
        clock_domain=ClockDomain.HOST_MONOTONIC,
        metadata={"fixture": "hypothesis"},
    )
    caller_owned = data.copy()
    frame = SignalFrame(
        stream_id="eeg",
        sequence_id=0,
        data=caller_owned,
        sample_rate_hz=250.0,
        host_receive_time_ns=123,
        clock_domain=ClockDomain.HOST_MONOTONIC,
        metadata={
            "axis_order": ("channel", "time"),
            "channel_names": names,
            "modality": "eeg",
        },
    )
    validate_frame_against_descriptor(descriptor, frame)

    caller_owned[...] = 0
    assert np.array_equal(frame.data, data)
    assert frame.data.flags.writeable is False

    async def write_archive(root: Path) -> None:
        writer = SessionArchiveWriter(root, session_id="property")
        writer.register_stream(descriptor)
        await writer.write(frame)
        await writer.close(runtime_metrics={"state": "stopped"})

    with tempfile.TemporaryDirectory(prefix="neuros-property-") as raw_root:
        root = Path(raw_root) / "archive"
        asyncio.run(write_archive(root))

        reader = SessionArchiveReader(root, verify_hashes=True)
        restored = list(reader.iter_frames("eeg"))
        assert len(restored) == 1
        assert np.array_equal(restored[0].data, data)
        assert restored[0].data.dtype == data.dtype
        assert reader.descriptor("eeg").fingerprint() == descriptor.fingerprint()

        payload = next((root / "streams" / "eeg" / "frames").glob("*.npy"))
        mutated = bytearray(payload.read_bytes())
        mutated[len(mutated) // 2] ^= 0x01
        payload.write_bytes(mutated)

        with pytest.raises(IOError, match="Data hash mismatch"):
            list(SessionArchiveReader(root, verify_hashes=True).iter_frames("eeg"))


@given(left=st.integers(), right=st.integers(), nested=st.integers())
@settings(max_examples=50, deadline=None)
def test_stream_descriptor_fingerprint_is_mapping_order_invariant(left: int, right: int, nested: int):
    first = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=256.0,
        metadata={"left": left, "right": right, "nested": {"value": nested, "kind": "x"}},
    )
    second = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=256.0,
        metadata={"nested": {"kind": "x", "value": nested}, "right": right, "left": left},
    )
    assert first.fingerprint() == second.fingerprint()


@given(
    bad_rate=st.one_of(
        st.just(0.0),
        st.floats(max_value=-1e-12, allow_nan=False, allow_infinity=False),
        st.just(float("inf")),
        st.just(float("-inf")),
        st.just(float("nan")),
    )
)
@settings(max_examples=20, deadline=None)
def test_signal_contract_rejects_nonphysical_declared_sample_rates(bad_rate: float):
    with pytest.raises((TypeError, ValueError)):
        StreamDescriptor(stream_id="eeg", modality="eeg", sample_rate_hz=bad_rate)

    with pytest.raises((TypeError, ValueError)):
        SignalFrame(
            stream_id="eeg",
            sequence_id=0,
            data=np.ones(2, dtype=np.float32),
            sample_rate_hz=bad_rate,
            host_receive_time_ns=1,
        )


@given(channels=st.integers(min_value=1, max_value=8), delta=st.integers(min_value=1, max_value=4))
@settings(max_examples=24, deadline=None)
def test_descriptor_frame_channel_geometry_mismatch_is_never_silently_accepted(channels: int, delta: int):
    descriptor = StreamDescriptor(
        stream_id="eeg",
        modality="eeg",
        sample_rate_hz=250.0,
        channel_names=tuple(f"ch{index}" for index in range(channels)),
    )
    frame = SignalFrame(
        stream_id="eeg",
        sequence_id=0,
        data=np.ones(channels + delta, dtype=np.float32),
        sample_rate_hz=250.0,
        host_receive_time_ns=1,
    )
    with pytest.raises(ValueError, match="channel geometry"):
        validate_frame_against_descriptor(descriptor, frame)
