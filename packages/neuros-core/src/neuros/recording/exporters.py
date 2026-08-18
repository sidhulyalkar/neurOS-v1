"""Optional NWB and Zarr exports from the canonical neurOS session archive."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .archive import SessionArchiveReader, _descriptor_to_dict, _jsonable


def _frames(reader: SessionArchiveReader, stream_id: str):
    frames = list(reader.iter_frames(stream_id))
    if not frames:
        raise ValueError(f"Cannot export empty stream: {stream_id}")
    shapes = {tuple(np.asarray(frame.data).shape) for frame in frames}
    if len(shapes) != 1:
        raise ValueError(
            f"NWB/Zarr dense export requires a stable frame shape for {stream_id}; got {sorted(shapes)}"
        )
    return frames


def export_zarr(archive: str | Path, destination: str | Path) -> Path:
    """Export a neurOS archive to a dense Zarr hierarchy.

    Exact per-frame neurOS metadata is retained as JSON in group attributes;
    the canonical archive remains the authoritative lossless replay format.
    """

    try:
        import zarr
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Zarr export requires `pip install neuros-core[zarr]`") from exc

    reader = SessionArchiveReader(archive)
    destination = Path(destination)
    root = zarr.open_group(str(destination), mode="w")
    root.attrs["neuros_manifest_json"] = json.dumps(reader.manifest, sort_keys=True, default=str)

    for stream_id in reader.stream_ids:
        descriptor = reader.descriptor(stream_id)
        frames = _frames(reader, stream_id)
        group = root.create_group(stream_id)
        data = np.stack([np.asarray(frame.data) for frame in frames])
        group.create_dataset("data", data=data, chunks=(1, *data.shape[1:]))
        group.create_dataset(
            "sequence_id", data=np.asarray([frame.sequence_id for frame in frames], dtype=np.int64)
        )
        group.create_dataset(
            "host_receive_time_ns",
            data=np.asarray([frame.host_receive_time_ns for frame in frames], dtype=np.int64),
        )
        group.create_dataset(
            "device_time_ns",
            data=np.asarray([
                -1 if frame.device_time_ns is None else frame.device_time_ns for frame in frames
            ], dtype=np.int64),
        )
        group.create_dataset(
            "synchronized_time_ns",
            data=np.asarray([
                -1 if frame.synchronized_time_ns is None else frame.synchronized_time_ns
                for frame in frames
            ], dtype=np.int64),
        )
        group.create_dataset(
            "quality", data=np.asarray([int(frame.quality) for frame in frames], dtype=np.int64)
        )
        group.attrs["descriptor_json"] = json.dumps(
            _descriptor_to_dict(descriptor), sort_keys=True, default=str
        )
        group.attrs["frame_metadata_json"] = json.dumps(
            [_jsonable(dict(frame.metadata)) for frame in frames], sort_keys=True, default=str
        )
    return destination


def export_nwb(archive: str | Path, destination: str | Path) -> Path:
    """Export a neurOS archive to NWB for ecosystem interoperability."""

    try:
        from pynwb import NWBHDF5IO, NWBFile, TimeSeries
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("NWB export requires `pip install neuros-core[nwb]`") from exc

    reader = SessionArchiveReader(archive)
    destination = Path(destination)
    created = datetime.fromisoformat(reader.manifest["created_at"])
    nwbfile = NWBFile(
        session_description=str(reader.manifest.get("metadata", {}).get("description", "neurOS session")),
        identifier=str(reader.manifest["session_id"]),
        session_start_time=created,
    )

    exact_metadata: dict[str, Any] = {
        "manifest": reader.manifest,
        "streams": {},
    }
    for stream_id in reader.stream_ids:
        descriptor = reader.descriptor(stream_id)
        frames = _frames(reader, stream_id)
        data = np.stack([np.asarray(frame.data) for frame in frames])
        timestamps = np.asarray([frame.timestamp_ns / 1_000_000_000.0 for frame in frames])
        unit = descriptor.units[0] if descriptor.units else "a.u."
        series = TimeSeries(
            name=stream_id,
            data=data,
            unit=unit,
            timestamps=timestamps,
            description=(
                f"neurOS {descriptor.modality} stream; exact clock domains, sequence IDs, "
                "quality flags, and frame metadata are stored in neuros_exact_metadata"
            ),
        )
        nwbfile.add_acquisition(series)
        exact_metadata["streams"][stream_id] = {
            "descriptor": _descriptor_to_dict(descriptor),
            "frames": [
                {
                    "sequence_id": frame.sequence_id,
                    "sample_rate_hz": frame.sample_rate_hz,
                    "host_receive_time_ns": frame.host_receive_time_ns,
                    "device_time_ns": frame.device_time_ns,
                    "synchronized_time_ns": frame.synchronized_time_ns,
                    "clock_domain": frame.clock_domain.value,
                    "quality": int(frame.quality),
                    "metadata": _jsonable(dict(frame.metadata)),
                }
                for frame in frames
            ],
        }

    nwbfile.add_scratch(
        json.dumps(exact_metadata, sort_keys=True, default=str),
        name="neuros_exact_metadata",
        description="Lossless neurOS frame/timing/provenance metadata for round-trip reconstruction.",
    )
    with NWBHDF5IO(str(destination), "w") as io:
        io.write(nwbfile)
    return destination
