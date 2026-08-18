"""Config-backed validation and execution commands."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Awaitable, Callable

from neuros.config import load_config, resolve_config
from neuros.errors import ConfigurationError
from neuros.runtime import RuntimeExecutor

OutputCallback = Callable[[Any], Awaitable[None] | None]


def validate_config(path: str | Path) -> dict[str, Any]:
    config = load_config(path)
    resolved = resolve_config(config)
    return {
        "path": str(Path(path)),
        "schema_version": config.schema_version,
        "streams": [stream.stream_id for stream in resolved.streams],
        "decoder": config.decoder.plugin,
        "sinks": [item.plugin for item in config.sinks],
        "monitors": [item.plugin for item in config.monitors],
        "runtime": {
            "queue_capacity": config.runtime.queue_capacity,
            "overflow_policy": config.runtime.overflow_policy.value,
        },
        "graph": {
            "nodes": list(resolved.graph.topological_order()),
            "edges": [f"{edge.source}->{edge.target}" for edge in resolved.graph.edges],
        },
    }


def _assert_decoder_ready(decoder: Any) -> None:
    if hasattr(decoder, "is_trained") and not bool(decoder.is_trained):
        raise ConfigurationError(
            f"Decoder {type(decoder).__name__} is not trained. Load a trained artifact "
            "or use a training-free decoder such as 'threshold' for runtime smoke tests."
        )


async def execute_config(
    path: str | Path,
    *,
    duration_s: float | None = None,
    on_output: OutputCallback | None = None,
) -> dict[str, Any]:
    if duration_s is not None and duration_s <= 0:
        raise ConfigurationError("duration must be positive")
    config = load_config(path)
    resolved = resolve_config(config)
    _assert_decoder_ready(resolved.decoder)
    executor = RuntimeExecutor(resolved.graph)
    await executor.start()

    async def consume() -> None:
        async for output in executor.outputs():
            if on_output is None:
                continue
            result = on_output(output)
            if asyncio.iscoroutine(result):
                await result

    consumer = asyncio.create_task(consume(), name="neuros:cli-output")
    try:
        if duration_s is None:
            await executor.wait()
        else:
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
