"""Record, inspect, replay, and export neurOS session archives."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from typing import Any, Awaitable, Callable, Iterable

import yaml

from neuros.config import load_config, resolve_config
from neuros.errors import ConfigurationError
from neuros.recording import (
    ArchiveReplaySource,
    RecordingSource,
    SessionArchiveReader,
    SessionArchiveWriter,
    StreamIdentitySource,
    export_nwb,
    export_zarr,
)
from neuros.runtime import NodeKind, RuntimeExecutor, RuntimeGraph

from .config_commands import _assert_decoder_ready

OutputCallback = Callable[[Any], Awaitable[None] | None]


def _raw_config(path: str | Path) -> dict[str, Any]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ConfigurationError("configuration root must be a mapping")
    return raw


def _recording_graph(graph: RuntimeGraph, writer: SessionArchiveWriter) -> RuntimeGraph:
    """Decorate graph sources with canonical stream identity and recording.

    A driver may expose a class/device-derived descriptor ID such as
    ``mockdriver`` while the runtime config binds it to a semantic stream such
    as ``eeg``. The configured graph identity is canonical for recording and
    replay; the device-native identity is retained as frame/descriptor metadata.
    """

    result = RuntimeGraph()
    for node in graph.nodes.values():
        operator = node.operator
        if node.kind is NodeKind.SOURCE:
            configured_stream_id = node.metadata.get("stream_id")
            if configured_stream_id:
                operator = StreamIdentitySource(operator, str(configured_stream_id))
            operator = RecordingSource(operator, writer)
        result.add_node(replace(node, operator=operator))
    for edge in graph.edges:
        result.connect(edge)
    result.validate()
    return result


async def _consume_outputs(executor: RuntimeExecutor, callback: OutputCallback | None) -> None:
    async for output in executor.outputs():
        if callback is None:
            continue
        result = callback(output)
        if asyncio.iscoroutine(result):
            await result


async def record_config(
    config_path: str | Path,
    output: str | Path,
    *,
    session_id: str,
    duration_s: float,
    overwrite: bool = False,
    export_formats: Iterable[str] = (),
    on_output: OutputCallback | None = None,
) -> dict[str, Any]:
    if duration_s <= 0:
        raise ConfigurationError("record duration must be positive")
    config = load_config(config_path)
    raw_config = _raw_config(config_path)
    resolved = resolve_config(config)
    _assert_decoder_ready(resolved.decoder)
    writer = SessionArchiveWriter(
        output,
        session_id=session_id,
        config=raw_config,
        metadata={"config_path": str(config_path)},
        overwrite=overwrite,
    )
    executor = RuntimeExecutor(_recording_graph(resolved.graph, writer))
    await executor.start()
    consumer = asyncio.create_task(_consume_outputs(executor, on_output))
    try:
        await asyncio.sleep(duration_s)
        await executor.stop()
        await consumer
        snapshot = executor.snapshot()
        await writer.close(runtime_metrics=snapshot)
    except BaseException:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        if executor.state.value not in {"stopped", "failed"}:
            await executor.stop()
        await writer.close(runtime_metrics=executor.snapshot())
        raise

    exports: dict[str, str] = {}
    archive_path = Path(output)
    for export_format in export_formats:
        normalized = export_format.lower()
        if normalized == "zarr":
            target = archive_path.parent / f"{archive_path.name}.zarr"
            exports[normalized] = str(export_zarr(archive_path, target))
        elif normalized == "nwb":
            target = archive_path.parent / f"{archive_path.name}.nwb"
            exports[normalized] = str(export_nwb(archive_path, target))
        else:
            raise ConfigurationError(f"Unsupported export format: {export_format}")
    return {
        "archive": str(archive_path),
        "exports": exports,
        **SessionArchiveReader(archive_path).summary(),
    }


async def replay_archive(
    archive: str | Path,
    config_path: str | Path,
    *,
    realtime: bool = False,
    speed: float = 1.0,
    duration_s: float | None = None,
    on_output: OutputCallback | None = None,
) -> dict[str, Any]:
    if speed <= 0:
        raise ConfigurationError("replay speed must be positive")
    reader = SessionArchiveReader(archive)
    config = load_config(config_path)
    configured_ids = {stream.stream_id for stream in config.streams}
    missing = configured_ids - set(reader.stream_ids)
    if missing:
        raise ConfigurationError(f"Archive is missing configured streams: {sorted(missing)}")
    overrides = {
        stream.stream_id: ArchiveReplaySource(
            reader, stream.stream_id, realtime=realtime, speed=speed
        )
        for stream in config.streams
    }
    resolved = resolve_config(config, source_overrides=overrides)
    _assert_decoder_ready(resolved.decoder)
    executor = RuntimeExecutor(resolved.graph)
    await executor.start()
    consumer = asyncio.create_task(_consume_outputs(executor, on_output))
    try:
        if duration_s is None:
            await executor.wait()
        else:
            if duration_s <= 0:
                raise ConfigurationError("replay duration must be positive")
            await asyncio.sleep(duration_s)
            await executor.stop()
        await consumer
    except BaseException:
        consumer.cancel()
        await asyncio.gather(consumer, return_exceptions=True)
        if executor.state.value not in {"stopped", "failed"}:
            await executor.stop()
        raise
    return executor.snapshot()


def inspect_archive(path: str | Path, *, verify_hashes: bool = False) -> dict[str, Any]:
    reader = SessionArchiveReader(path, verify_hashes=verify_hashes)
    result = reader.summary()
    if verify_hashes:
        for stream_id in reader.stream_ids:
            # Iteration performs payload verification lazily.
            sum(1 for _ in reader.iter_frames(stream_id))
        result["integrity"] = "verified"
    else:
        result["integrity"] = "not_checked"
    return result
