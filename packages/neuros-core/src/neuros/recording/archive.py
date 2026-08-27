"""Lossless persistent session archives for neurOS SignalFrame streams."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, AsyncIterator, Iterable, Mapping

import numpy as np

from neuros.contracts import (
    ClockDomain,
    QualityFlag,
    SignalFrame,
    StreamDescriptor,
    frame_channel_count,
    validate_frame_against_descriptor,
)


ARCHIVE_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (ClockDomain, QualityFlag)):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(item) for item in value]
    return value


def canonical_hash(value: Any) -> str:
    payload = json.dumps(
        _jsonable(value), sort_keys=True, separators=(",", ":"), default=str
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("neuros", "neuros-core", "neuros-drivers", "neuros-models", "neuros-orion"):
        try:
            result[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return result


def _descriptor_to_dict(descriptor: StreamDescriptor) -> dict[str, Any]:
    return {
        "stream_id": descriptor.stream_id,
        "modality": descriptor.modality,
        "sample_rate_hz": descriptor.sample_rate_hz,
        "channel_names": list(descriptor.channel_names),
        "channel_types": list(descriptor.channel_types),
        "units": list(descriptor.units),
        "device": descriptor.device,
        "manufacturer": descriptor.manufacturer,
        "clock_domain": descriptor.clock_domain.value,
        "metadata": _jsonable(dict(descriptor.metadata)),
        "fingerprint_sha256": descriptor.fingerprint(),
    }


def _descriptor_from_dict(raw: Mapping[str, Any]) -> StreamDescriptor:
    descriptor = StreamDescriptor(
        stream_id=str(raw["stream_id"]),
        modality=str(raw["modality"]),
        sample_rate_hz=float(raw["sample_rate_hz"]),
        channel_names=tuple(raw.get("channel_names", [])),
        channel_types=tuple(raw.get("channel_types", [])),
        units=tuple(raw.get("units", [])),
        device=raw.get("device"),
        manufacturer=raw.get("manufacturer"),
        clock_domain=ClockDomain(raw.get("clock_domain", ClockDomain.UNKNOWN.value)),
        metadata=raw.get("metadata", {}),
    )
    expected = raw.get("fingerprint_sha256")
    if expected is not None and str(expected) != descriptor.fingerprint():
        raise IOError("StreamDescriptor fingerprint mismatch")
    return descriptor


@dataclass(slots=True)
class SessionManifest:
    session_id: str
    created_at: str
    schema_version: int = ARCHIVE_SCHEMA_VERSION
    status: str = "recording"
    git_sha: str | None = None
    config_hash: str | None = None
    package_versions: dict[str, str] = field(default_factory=dict)
    host: dict[str, Any] = field(default_factory=dict)
    streams: dict[str, dict[str, Any]] = field(default_factory=dict)
    runtime_metrics: dict[str, Any] = field(default_factory=dict)
    model_artifacts: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _jsonable(asdict(self))


class SessionArchiveWriter:
    """Append SignalFrames to a lossless, dependency-free directory archive.

    Data arrays are stored as individual NPY payloads so arbitrary frame shapes
    are preserved. An NDJSON index records every timing/provenance field and a
    SHA-256 hash of each data payload. This is the canonical replay source; NWB
    and Zarr are export formats rather than the authority for neurOS semantics.

    Once a stream is registered, every frame is validated against that exact
    descriptor before bytes are written. The archive therefore cannot silently
    mix stream IDs, sample rates, clock domains, or channel geometries under one
    stream identity.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        session_id: str,
        config: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        model_artifacts: Iterable[Mapping[str, Any]] = (),
        overwrite: bool = False,
    ) -> None:
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()) and not overwrite:
            raise FileExistsError(f"Archive is not empty: {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "streams").mkdir(exist_ok=True)
        self._lock = asyncio.Lock()
        self._closed = False
        self._descriptors: dict[str, StreamDescriptor] = {}
        self._counts: dict[str, int] = {}
        self.manifest = SessionManifest(
            session_id=session_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_sha=_git_sha(),
            config_hash=canonical_hash(config) if config is not None else None,
            package_versions=_package_versions(),
            host={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            model_artifacts=[_jsonable(dict(item)) for item in model_artifacts],
            metadata=_jsonable(dict(metadata or {})),
        )
        if config is not None:
            self._write_json_atomic(self.root / "config.json", _jsonable(config))
        self._flush_manifest()

    def _write_json_atomic(self, path: Path, value: Any) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str), encoding="utf-8"
        )
        temporary.replace(path)

    def _flush_manifest(self) -> None:
        self._write_json_atomic(self.root / "manifest.json", self.manifest.to_dict())

    def register_stream(self, descriptor: StreamDescriptor) -> None:
        if not isinstance(descriptor, StreamDescriptor):
            raise TypeError("register_stream requires a StreamDescriptor")
        existing = self._descriptors.get(descriptor.stream_id)
        if existing is not None and existing.fingerprint() != descriptor.fingerprint():
            raise ValueError(f"Conflicting descriptor for stream {descriptor.stream_id}")
        if existing is not None:
            return
        self._descriptors[descriptor.stream_id] = descriptor
        self._counts[descriptor.stream_id] = 0
        stream_root = self.root / "streams" / descriptor.stream_id
        (stream_root / "frames").mkdir(parents=True, exist_ok=True)
        encoded = _descriptor_to_dict(descriptor)
        self._write_json_atomic(stream_root / "descriptor.json", encoded)
        self.manifest.streams[descriptor.stream_id] = {
            "descriptor": encoded,
            "descriptor_fingerprint_sha256": descriptor.fingerprint(),
            "frame_count": 0,
        }
        self._flush_manifest()

    def _descriptor_from_unregistered_frame(self, item: SignalFrame) -> StreamDescriptor:
        channels = frame_channel_count(item)
        metadata_names = tuple(item.metadata.get("channel_names", ()))
        metadata_types = tuple(item.metadata.get("channel_types", ()))
        metadata_units = tuple(item.metadata.get("units", ()))
        if metadata_names and len(metadata_names) != channels:
            raise ValueError("SignalFrame channel_names do not match its channel axis")
        if metadata_types and len(metadata_types) != channels:
            raise ValueError("SignalFrame channel_types do not match its channel axis")
        if metadata_units and len(metadata_units) != channels:
            raise ValueError("SignalFrame units do not match its channel axis")
        return StreamDescriptor(
            stream_id=item.stream_id,
            modality=str(item.metadata.get("modality", "unknown")),
            sample_rate_hz=item.sample_rate_hz,
            channel_names=metadata_names or tuple(f"ch{i}" for i in range(channels)),
            channel_types=metadata_types,
            units=metadata_units,
            clock_domain=item.clock_domain,
            metadata={"auto_registered_from_frame": True},
        )

    async def write(self, item: SignalFrame) -> None:
        if self._closed:
            raise RuntimeError("SessionArchiveWriter is closed")
        if not isinstance(item, SignalFrame):
            raise TypeError("SessionArchiveWriter accepts SignalFrame objects")
        async with self._lock:
            if item.stream_id not in self._descriptors:
                self.register_stream(self._descriptor_from_unregistered_frame(item))
            descriptor = self._descriptors[item.stream_id]
            validate_frame_against_descriptor(descriptor, item)

            index = self._counts[item.stream_id]
            stream_root = self.root / "streams" / item.stream_id
            relative_data = Path("frames") / f"{index:012d}.npy"
            data_path = stream_root / relative_data
            temporary = data_path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                np.save(handle, item.data, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.replace(data_path)
            payload_hash = hashlib.sha256(data_path.read_bytes()).hexdigest()
            row = {
                "index": index,
                "sequence_id": item.sequence_id,
                "data": str(relative_data),
                "data_sha256": payload_hash,
                "sample_rate_hz": item.sample_rate_hz,
                "host_receive_time_ns": item.host_receive_time_ns,
                "device_time_ns": item.device_time_ns,
                "synchronized_time_ns": item.synchronized_time_ns,
                "clock_domain": item.clock_domain.value,
                "quality": int(item.quality),
                "metadata": _jsonable(dict(item.metadata)),
            }
            with (stream_root / "index.ndjson").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True, default=str) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            self._counts[item.stream_id] += 1
            self.manifest.streams[item.stream_id]["frame_count"] = self._counts[item.stream_id]

    async def close(self, *, runtime_metrics: Mapping[str, Any] | None = None) -> None:
        if self._closed:
            return
        async with self._lock:
            self.manifest.status = "complete"
            self.manifest.runtime_metrics = _jsonable(dict(runtime_metrics or {}))
            self._flush_manifest()
            self._closed = True


class SessionArchiveReader:
    def __init__(self, root: str | Path, *, verify_hashes: bool = True) -> None:
        self.root = Path(root)
        raw = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if int(raw.get("schema_version", -1)) != ARCHIVE_SCHEMA_VERSION:
            raise ValueError("Unsupported neurOS archive schema")
        self.manifest = raw
        self.verify_hashes = verify_hashes

    @property
    def stream_ids(self) -> tuple[str, ...]:
        return tuple(self.manifest.get("streams", {}).keys())

    def descriptor(self, stream_id: str) -> StreamDescriptor:
        raw = json.loads(
            (self.root / "streams" / stream_id / "descriptor.json").read_text(encoding="utf-8")
        )
        return _descriptor_from_dict(raw)

    def iter_frames(self, stream_id: str) -> Iterable[SignalFrame]:
        stream_root = self.root / "streams" / stream_id
        index_path = stream_root / "index.ndjson"
        if not index_path.exists():
            return
        descriptor = self.descriptor(stream_id)
        for line in index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            data_path = stream_root / row["data"]
            if self.verify_hashes:
                actual = hashlib.sha256(data_path.read_bytes()).hexdigest()
                if actual != row["data_sha256"]:
                    raise IOError(f"Data hash mismatch: {data_path}")
            with data_path.open("rb") as handle:
                data = np.load(handle, allow_pickle=False)
            frame = SignalFrame(
                stream_id=stream_id,
                sequence_id=int(row["sequence_id"]),
                data=data,
                sample_rate_hz=float(row["sample_rate_hz"]),
                host_receive_time_ns=int(row["host_receive_time_ns"]),
                device_time_ns=None if row["device_time_ns"] is None else int(row["device_time_ns"]),
                synchronized_time_ns=None
                if row["synchronized_time_ns"] is None
                else int(row["synchronized_time_ns"]),
                clock_domain=ClockDomain(row["clock_domain"]),
                quality=QualityFlag(int(row["quality"])),
                metadata=row.get("metadata", {}),
            )
            validate_frame_against_descriptor(descriptor, frame)
            yield frame

    def summary(self) -> dict[str, Any]:
        return {
            "session_id": self.manifest["session_id"],
            "status": self.manifest["status"],
            "created_at": self.manifest["created_at"],
            "git_sha": self.manifest.get("git_sha"),
            "config_hash": self.manifest.get("config_hash"),
            "streams": {
                stream_id: self.manifest["streams"][stream_id]["frame_count"]
                for stream_id in self.stream_ids
            },
            "runtime_metrics": self.manifest.get("runtime_metrics", {}),
        }


class ArchiveReplaySource:
    """Stream frames lazily from a persistent archive."""

    def __init__(
        self,
        reader: SessionArchiveReader,
        stream_id: str,
        *,
        realtime: bool = False,
        speed: float = 1.0,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self.reader = reader
        self.stream_id = stream_id
        self.realtime = realtime
        self.speed = speed
        self._running = False

    @property
    def descriptor(self) -> StreamDescriptor:
        return self.reader.descriptor(self.stream_id)

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def frames(self) -> AsyncIterator[SignalFrame]:
        previous: int | None = None
        for frame in self.reader.iter_frames(self.stream_id):
            if not self._running:
                return
            if self.realtime and previous is not None:
                delay = max(0, frame.timestamp_ns - previous) / 1_000_000_000.0
                await asyncio.sleep(delay / self.speed)
            yield frame
            previous = frame.timestamp_ns
            await asyncio.sleep(0)


class RecordingSource:
    """Source decorator that records exact frames before forwarding them."""

    def __init__(self, source: Any, writer: SessionArchiveWriter) -> None:
        self.source = source
        self.writer = writer

    @property
    def descriptor(self) -> StreamDescriptor:
        return self.source.descriptor

    async def start(self) -> None:
        self.writer.register_stream(self.descriptor)
        await self.source.start()

    async def stop(self) -> None:
        await self.source.stop()

    async def frames(self) -> AsyncIterator[SignalFrame]:
        async for frame in self.source.frames():
            await self.writer.write(frame)
            yield frame
