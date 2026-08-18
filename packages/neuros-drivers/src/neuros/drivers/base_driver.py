"""Base classes and interfaces for neurOS drivers.

Legacy tuple streaming remains supported, while ``frames()`` exposes the new
canonical ``SignalFrame`` contract with explicit timing metadata.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import numpy as np

from neuros.contracts import ClockDomain, SignalFrame, StreamDescriptor
from neuros.runtime import OverflowPolicy, QueueStats, put_with_policy


class BaseDriver(ABC):
    """Abstract base class for physical and simulated data sources."""

    def __init__(
        self,
        sampling_rate: float = 250.0,
        channels: int = 8,
        *,
        stream_id: str | None = None,
        modality: str = "unknown",
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        self.sampling_rate = sampling_rate
        self.channels = channels
        self.stream_id = stream_id or self.__class__.__name__.lower()
        self.modality = modality
        self.overflow_policy = overflow_policy
        self._task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._queue_stats = QueueStats()
        self._running = False

    @property
    def descriptor(self) -> StreamDescriptor:
        return StreamDescriptor(
            stream_id=self.stream_id,
            modality=self.modality,
            sample_rate_hz=self.sampling_rate,
            channel_names=tuple(f"ch{i}" for i in range(self.channels)),
            clock_domain=ClockDomain.UNKNOWN,
            device=self.__class__.__name__,
        )

    @property
    def queue_stats(self) -> QueueStats:
        return self._queue_stats

    async def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except asyncio.QueueEmpty:
                break

    async def _run(self) -> None:
        async for item in self._stream():
            try:
                await put_with_policy(
                    self._queue,
                    item,
                    policy=self.overflow_policy,
                    stats=self._queue_stats,
                )
            except asyncio.CancelledError:
                break
            if not self._running:
                break

    async def __aiter__(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        """Legacy iterator yielding ``(timestamp_seconds, data)`` tuples."""
        while self._running:
            try:
                item = await self._queue.get()
                self._queue.task_done()
                yield item
            except asyncio.CancelledError:
                break

    async def frames(self) -> AsyncIterator[SignalFrame]:
        """Yield canonical :class:`SignalFrame` objects."""
        sequence_id = 0
        async for timestamp, data in self:
            yield SignalFrame.from_legacy(
                stream_id=self.stream_id,
                sequence_id=sequence_id,
                timestamp_seconds=float(timestamp),
                data=np.asarray(data),
                sample_rate_hz=self.sampling_rate,
                clock_domain=ClockDomain.UNKNOWN,
                metadata={"driver": self.__class__.__name__},
            )
            sequence_id += 1

    @abstractmethod
    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        raise NotImplementedError
