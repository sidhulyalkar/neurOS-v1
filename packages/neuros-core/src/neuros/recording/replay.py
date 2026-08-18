"""Deterministic recording and replay primitives for neurOS."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterable

from neuros.contracts import SignalFrame, StreamDescriptor


@dataclass(slots=True)
class FrameRecorder:
    """Minimal in-memory sink useful for tests and deterministic experiments."""

    frames: list[SignalFrame] = field(default_factory=list)

    async def write(self, item: SignalFrame) -> None:
        if not isinstance(item, SignalFrame):
            raise TypeError("FrameRecorder accepts SignalFrame objects")
        self.frames.append(item)

    def snapshot(self) -> tuple[SignalFrame, ...]:
        return tuple(self.frames)


class ReplaySource:
    """Replay a fixed frame sequence while preserving recorded timestamps."""

    def __init__(
        self,
        descriptor: StreamDescriptor,
        frames: Iterable[SignalFrame],
        *,
        realtime: bool = False,
        speed: float = 1.0,
    ) -> None:
        if speed <= 0:
            raise ValueError("speed must be positive")
        self._descriptor = descriptor
        self._frames = tuple(frames)
        self.realtime = realtime
        self.speed = speed
        self._running = False
        for frame in self._frames:
            if frame.stream_id != descriptor.stream_id:
                raise ValueError("all replay frames must match descriptor.stream_id")

    @property
    def descriptor(self) -> StreamDescriptor:
        return self._descriptor

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    async def frames(self) -> AsyncIterator[SignalFrame]:
        if not self._running:
            raise RuntimeError("ReplaySource must be started before frames()")
        previous_time_ns: int | None = None
        for frame in self._frames:
            if not self._running:
                break
            if self.realtime and previous_time_ns is not None:
                delay_s = max(0.0, frame.timestamp_ns - previous_time_ns) / 1_000_000_000.0
                await asyncio.sleep(delay_s / self.speed)
            yield frame
            previous_time_ns = frame.timestamp_ns
