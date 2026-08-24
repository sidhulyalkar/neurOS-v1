from datetime import datetime, timezone

import mne
import numpy as np
import pytest

from neuros.contracts import ClockDomain, SignalFrame
from neuros.interop.mne import frames_from_raw, raw_from_signal_frames, stream_descriptor_from_raw


def _raw(*, n_times: int = 10, sfreq: float = 100.0):
    info = mne.create_info(
        ch_names=["Fz", "Cz"],
        sfreq=sfreq,
        ch_types=["eeg", "eeg"],
        verbose=False,
    )
    data = np.vstack(
        [
            np.linspace(-1e-6, 1e-6, n_times),
            np.linspace(2e-6, -2e-6, n_times),
        ]
    )
    return mne.io.RawArray(data, info, verbose=False), data


def test_descriptor_preserves_mne_geometry_without_preprocessing():
    raw, _ = _raw()

    descriptor = stream_descriptor_from_raw(raw, stream_id="subject-01-eeg")

    assert descriptor.stream_id == "subject-01-eeg"
    assert descriptor.modality == "eeg"
    assert descriptor.sample_rate_hz == 100.0
    assert descriptor.channel_names == ("Fz", "Cz")
    assert descriptor.channel_types == ("eeg", "eeg")
    assert descriptor.metadata["interop"] == "mne"
    assert descriptor.metadata["mne_n_times"] == 10


def test_raw_to_signalframe_chunks_are_explicit_sample_by_channel():
    raw, original = _raw()

    frames = tuple(
        frames_from_raw(
            raw,
            stream_id="subject-01-eeg",
            chunk_samples=4,
            start_sequence=7,
        )
    )

    assert [frame.sequence_id for frame in frames] == [7, 8, 9]
    assert [frame.data.shape for frame in frames] == [(4, 2), (4, 2), (2, 2)]
    assert all(frame.metadata["axis_order"] == ("sample", "channel") for frame in frames)
    assert all(frame.clock_domain is ClockDomain.UNKNOWN for frame in frames)
    np.testing.assert_allclose(np.concatenate([frame.data for frame in frames]).T, original)


def test_measurement_date_produces_absolute_synchronized_frame_time():
    raw, _ = _raw()
    raw.set_meas_date(datetime(2026, 8, 24, 12, 0, tzinfo=timezone.utc))

    frame = next(frames_from_raw(raw, chunk_samples=5))

    assert frame.clock_domain is ClockDomain.SYNCHRONIZED
    assert frame.synchronized_time_ns is not None
    assert frame.metadata["measurement_time_available"] is True


def test_mne_roundtrip_preserves_samples_rate_and_channel_identity():
    raw, original = _raw(n_times=11, sfreq=128.0)
    descriptor = stream_descriptor_from_raw(raw, stream_id="roundtrip")
    frames = tuple(frames_from_raw(raw, stream_id="roundtrip", chunk_samples=3))

    reconstructed = raw_from_signal_frames(frames, descriptor=descriptor)

    assert reconstructed.info["sfreq"] == 128.0
    assert reconstructed.ch_names == ["Fz", "Cz"]
    assert reconstructed.get_channel_types() == ["eeg", "eeg"]
    np.testing.assert_allclose(reconstructed.get_data(), original)
    assert "Converted from neurOS stream roundtrip" in reconstructed.info["description"]


def test_single_sample_live_style_frames_can_be_exported_to_mne():
    frames = [
        SignalFrame(
            stream_id="live",
            sequence_id=index,
            data=np.asarray([index, index + 0.5], dtype=np.float64),
            sample_rate_hz=250.0,
            host_receive_time_ns=index + 1,
            metadata={"channel_names": ("C3", "C4"), "channel_types": ("eeg", "eeg")},
        )
        for index in range(3)
    ]

    raw = raw_from_signal_frames(frames)

    assert raw.ch_names == ["C3", "C4"]
    np.testing.assert_allclose(raw.get_data(), [[0.0, 1.0, 2.0], [0.5, 1.5, 2.5]])


def test_ambiguous_two_dimensional_frame_fails_closed():
    frame = SignalFrame(
        stream_id="ambiguous",
        sequence_id=0,
        data=np.ones((4, 2)),
        sample_rate_hz=100.0,
        host_receive_time_ns=1,
    )

    with pytest.raises(ValueError, match="axis_order"):
        raw_from_signal_frames([frame])


def test_roundtrip_rejects_mixed_streams_and_non_monotonic_sequence():
    first = SignalFrame(
        stream_id="a",
        sequence_id=1,
        data=np.asarray([1.0, 2.0]),
        sample_rate_hz=100.0,
        host_receive_time_ns=1,
    )
    mixed = SignalFrame(
        stream_id="b",
        sequence_id=2,
        data=np.asarray([1.0, 2.0]),
        sample_rate_hz=100.0,
        host_receive_time_ns=2,
    )
    duplicate = SignalFrame(
        stream_id="a",
        sequence_id=1,
        data=np.asarray([3.0, 4.0]),
        sample_rate_hz=100.0,
        host_receive_time_ns=3,
    )

    with pytest.raises(ValueError, match="stream_id"):
        raw_from_signal_frames([first, mixed])
    with pytest.raises(ValueError, match="strictly increasing"):
        raw_from_signal_frames([first, duplicate])
