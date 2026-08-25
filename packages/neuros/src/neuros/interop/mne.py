"""MNE-Python interoperability for canonical neurOS signal contracts.

The adapter is intentionally small and explicit. MNE remains authoritative for
its object model and preprocessing ecosystem; neurOS only translates the
information needed to cross the runtime boundary without inventing hidden
preprocessing or resampling.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from datetime import datetime, timezone
from typing import Any

import numpy as np

from neuros.contracts import ClockDomain, QualityFlag, SignalFrame, StreamDescriptor


def _require_mne() -> Any:
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - exercised in minimal installs
        raise ImportError(
            "MNE interoperability requires the optional MNE dependency. "
            'Install it with `pip install "neuros[interop-mne]"`.'
        ) from exc
    return mne


def _measurement_date(raw: Any) -> datetime | None:
    value = raw.info.get("meas_date")
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    return None


def _device_metadata(raw: Any) -> tuple[str | None, str | None]:
    device_info = raw.info.get("device_info") or {}
    if not isinstance(device_info, dict):
        return None, None
    device = device_info.get("model") or device_info.get("type")
    manufacturer = device_info.get("manufacturer")
    return (
        str(device) if device is not None else None,
        str(manufacturer) if manufacturer is not None else None,
    )


def stream_descriptor_from_raw(
    raw: Any,
    *,
    stream_id: str = "mne-raw",
) -> StreamDescriptor:
    """Translate an MNE ``BaseRaw`` object into a stable stream descriptor.

    No preprocessing occurs. The descriptor records channel names/types,
    sampling rate, and a small provenance envelope describing the source Raw
    object.
    """

    _require_mne()
    sample_rate = float(raw.info["sfreq"])
    if not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError("MNE Raw sampling rate must be finite and positive")

    channel_names = tuple(str(name) for name in raw.ch_names)
    if not channel_names:
        raise ValueError("MNE Raw object must contain at least one channel")

    channel_types = tuple(str(kind) for kind in raw.get_channel_types())
    unique_types = tuple(dict.fromkeys(channel_types))
    modality = unique_types[0] if len(unique_types) == 1 else "mixed"
    device, manufacturer = _device_metadata(raw)
    meas_date = _measurement_date(raw)

    return StreamDescriptor(
        stream_id=stream_id,
        modality=modality,
        sample_rate_hz=sample_rate,
        channel_names=channel_names,
        channel_types=channel_types,
        device=device,
        manufacturer=manufacturer,
        clock_domain=ClockDomain.UNKNOWN,
        metadata={
            "interop": "mne",
            "mne_class": f"{raw.__class__.__module__}.{raw.__class__.__qualname__}",
            "mne_first_samp": int(raw.first_samp),
            "mne_n_times": int(raw.n_times),
            "mne_highpass_hz": float(raw.info.get("highpass", 0.0)),
            "mne_lowpass_hz": float(raw.info.get("lowpass", sample_rate / 2.0)),
            "mne_meas_date": meas_date.isoformat() if meas_date is not None else None,
        },
    )


def frames_from_raw(
    raw: Any,
    *,
    stream_id: str = "mne-raw",
    chunk_samples: int = 256,
    start: int = 0,
    stop: int | None = None,
    start_sequence: int = 0,
) -> Iterator[SignalFrame]:
    """Yield sample-by-channel ``SignalFrame`` chunks from an MNE Raw object.

    MNE stores data as ``channel x sample``. neurOS emits two-dimensional
    chunks as ``sample x channel`` and writes the axis order into immutable
    metadata so the conversion is never inferred later.

    If ``raw.info['meas_date']`` is present, frames receive an absolute
    synchronized timestamp derived from that measurement origin and the sample
    index. Otherwise the recording-relative position is retained in metadata
    and neurOS does not pretend that an absolute clock is available.
    """

    descriptor = stream_descriptor_from_raw(raw, stream_id=stream_id)
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    if start < 0:
        raise ValueError("start must be >= 0")
    resolved_stop = int(raw.n_times) if stop is None else int(stop)
    if resolved_stop < start or resolved_stop > int(raw.n_times):
        raise ValueError("stop must satisfy start <= stop <= raw.n_times")
    if start_sequence < 0:
        raise ValueError("start_sequence must be >= 0")

    meas_date = _measurement_date(raw)
    meas_origin_ns = int(round(meas_date.timestamp() * 1_000_000_000)) if meas_date else None
    sfreq = descriptor.sample_rate_hz
    sequence = start_sequence

    for chunk_start in range(start, resolved_stop, chunk_samples):
        chunk_stop = min(chunk_start + chunk_samples, resolved_stop)
        data = np.asarray(raw.get_data(start=chunk_start, stop=chunk_stop), dtype=np.float64).T
        if data.shape != (chunk_stop - chunk_start, len(descriptor.channel_names)):
            raise RuntimeError(
                "MNE returned unexpected data geometry: "
                f"expected {(chunk_stop - chunk_start, len(descriptor.channel_names))}, "
                f"received {data.shape}"
            )
        if not np.isfinite(data).all():
            raise ValueError("MNE Raw chunk contains NaN or infinite samples")

        relative_seconds = (int(raw.first_samp) + chunk_start) / sfreq
        synchronized_time_ns = (
            meas_origin_ns + int(round(relative_seconds * 1_000_000_000))
            if meas_origin_ns is not None
            else None
        )

        yield SignalFrame(
            stream_id=stream_id,
            sequence_id=sequence,
            data=data,
            sample_rate_hz=sfreq,
            host_receive_time_ns=time.monotonic_ns(),
            synchronized_time_ns=synchronized_time_ns,
            clock_domain=(
                ClockDomain.SYNCHRONIZED if synchronized_time_ns is not None else ClockDomain.UNKNOWN
            ),
            quality=QualityFlag.GOOD,
            metadata={
                "interop": "mne",
                "axis_order": ("sample", "channel"),
                "channel_names": descriptor.channel_names,
                "channel_types": descriptor.channel_types,
                "mne_start_sample": chunk_start,
                "mne_stop_sample": chunk_stop,
                "recording_relative_start_seconds": relative_seconds,
                "measurement_time_available": meas_origin_ns is not None,
            },
        )
        sequence += 1


def _frame_matrix(frame: SignalFrame) -> np.ndarray:
    data = np.asarray(frame.data)
    if data.ndim == 1:
        return data[np.newaxis, :]
    if data.ndim == 2:
        axis_order = tuple(frame.metadata.get("axis_order", ()))
        if axis_order != ("sample", "channel"):
            raise ValueError(
                "Two-dimensional SignalFrames require metadata "
                "axis_order=('sample', 'channel') for MNE conversion"
            )
        return data
    raise ValueError("MNE conversion supports one- or two-dimensional SignalFrames only")


def _mne_chunk_bounds(frame: SignalFrame, sample_count: int) -> tuple[int, int] | None:
    start = frame.metadata.get("mne_start_sample")
    stop = frame.metadata.get("mne_stop_sample")
    if start is None and stop is None:
        return None
    if start is None or stop is None:
        raise ValueError("MNE-derived SignalFrames must provide both start and stop sample metadata")
    start_i, stop_i = int(start), int(stop)
    if start_i < 0 or stop_i <= start_i:
        raise ValueError("Invalid MNE sample bounds in SignalFrame metadata")
    if stop_i - start_i != sample_count:
        raise ValueError("MNE sample bounds do not match SignalFrame sample count")
    return start_i, stop_i


def _restore_measurement_date(
    raw: Any,
    first: SignalFrame,
    descriptor: StreamDescriptor | None,
) -> None:
    date: datetime | None = None
    if descriptor is not None:
        encoded = descriptor.metadata.get("mne_meas_date")
        if isinstance(encoded, str) and encoded:
            date = datetime.fromisoformat(encoded)
            if date.tzinfo is None:
                date = date.replace(tzinfo=timezone.utc)

    if date is None and first.synchronized_time_ns is not None:
        relative = first.metadata.get("recording_relative_start_seconds")
        if relative is not None:
            origin_seconds = first.synchronized_time_ns / 1_000_000_000.0 - float(relative)
            date = datetime.fromtimestamp(origin_seconds, tz=timezone.utc)

    if date is not None:
        raw.set_meas_date(date)


def raw_from_signal_frames(
    frames: Iterable[SignalFrame],
    *,
    descriptor: StreamDescriptor | None = None,
) -> Any:
    """Construct an MNE ``RawArray`` from compatible SignalFrames.

    Frames must share stream identity, sampling rate, channel geometry, and a
    contiguous sequence. MNE-derived chunks must also have contiguous sample
    bounds. Two-dimensional frames are accepted only when their axis order is
    explicit. The adapter never resamples, pads, reorders, or silently repairs
    missing data.
    """

    mne = _require_mne()
    materialized = tuple(frames)
    if not materialized:
        raise ValueError("At least one SignalFrame is required")

    first = materialized[0]
    matrices: list[np.ndarray] = []
    channel_count: int | None = None
    previous_sequence: int | None = None
    previous_mne_stop: int | None = None

    for frame in materialized:
        if frame.stream_id != first.stream_id:
            raise ValueError("All SignalFrames must share the same stream_id")
        if not np.isclose(frame.sample_rate_hz, first.sample_rate_hz, rtol=0.0, atol=1e-12):
            raise ValueError("All SignalFrames must share the same sample_rate_hz")
        if previous_sequence is not None and frame.sequence_id != previous_sequence + 1:
            raise ValueError("SignalFrames must have contiguous, strictly increasing sequence_id")
        previous_sequence = frame.sequence_id

        matrix = np.asarray(_frame_matrix(frame), dtype=np.float64)
        if not np.isfinite(matrix).all():
            raise ValueError("SignalFrame contains NaN or infinite samples")
        if channel_count is None:
            channel_count = int(matrix.shape[1])
        elif matrix.shape[1] != channel_count:
            raise ValueError("SignalFrames have inconsistent channel geometry")

        bounds = _mne_chunk_bounds(frame, matrix.shape[0])
        if bounds is not None:
            start_i, stop_i = bounds
            if previous_mne_stop is not None and start_i != previous_mne_stop:
                raise ValueError("MNE-derived SignalFrames contain a sample gap or overlap")
            previous_mne_stop = stop_i
        elif previous_mne_stop is not None:
            raise ValueError("Cannot mix MNE-derived and unbounded SignalFrames in one conversion")

        matrices.append(matrix)

    assert channel_count is not None
    if descriptor is not None:
        if descriptor.stream_id != first.stream_id:
            raise ValueError("StreamDescriptor stream_id does not match SignalFrames")
        if not np.isclose(descriptor.sample_rate_hz, first.sample_rate_hz, rtol=0.0, atol=1e-12):
            raise ValueError("StreamDescriptor sampling rate does not match SignalFrames")
        if descriptor.channel_names and len(descriptor.channel_names) != channel_count:
            raise ValueError("StreamDescriptor channel geometry does not match SignalFrames")

    metadata_names = tuple(first.metadata.get("channel_names", ()))
    metadata_types = tuple(first.metadata.get("channel_types", ()))
    channel_names = (
        descriptor.channel_names
        if descriptor is not None and descriptor.channel_names
        else metadata_names
        if metadata_names
        else tuple(f"ch{index}" for index in range(channel_count))
    )
    channel_types = (
        descriptor.channel_types
        if descriptor is not None and descriptor.channel_types
        else metadata_types
        if metadata_types
        else tuple("eeg" for _ in range(channel_count))
    )
    if len(channel_names) != channel_count or len(channel_types) != channel_count:
        raise ValueError("Channel names/types do not match SignalFrame geometry")

    first_samp = 0
    if descriptor is not None:
        first_samp = int(descriptor.metadata.get("mne_first_samp", 0))
    first_chunk_start = first.metadata.get("mne_start_sample")
    if first_chunk_start is not None:
        first_samp += int(first_chunk_start)

    sample_by_channel = np.concatenate(matrices, axis=0)
    info = mne.create_info(
        ch_names=list(channel_names),
        sfreq=float(first.sample_rate_hz),
        ch_types=list(channel_types),
        verbose=False,
    )
    raw = mne.io.RawArray(
        sample_by_channel.T,
        info,
        first_samp=first_samp,
        verbose=False,
    )
    _restore_measurement_date(raw, first, descriptor)
    raw.info["description"] = (
        f"Converted from neurOS stream {first.stream_id}; "
        f"frames={len(materialized)}; sequence={materialized[0].sequence_id}-"
        f"{materialized[-1].sequence_id}"
    )
    return raw
