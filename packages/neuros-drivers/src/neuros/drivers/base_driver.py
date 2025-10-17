"""
Base classes and interfaces for neurOS drivers.

A driver abstracts away details of a physical or simulated device and provides
an asynchronous generator that yields neural samples.  Concrete drivers
inherit from :class:`BaseDriver` and implement the `_stream()` coroutine.
"""

from __future__ import annotations

import asyncio
import time
from abc import ABC, abstractmethod
from typing import AsyncIterator, Optional

import numpy as np


class BaseDriver(ABC):
    """Abstract base class for all drivers.

    A driver must implement the :meth:`_stream` method, which yields
    timestamped samples at the device's sampling rate.  Each yielded item is
    a tuple ``(timestamp, data)`` where ``timestamp`` is a float (seconds
    since epoch) and ``data`` is a 1‑D or 2‑D NumPy array representing
    channels × samples.

    Drivers can be started and stopped using the :meth:`start` and
    :meth:`stop` methods, which manage internal tasks and resources.
    """

    def __init__(self, sampling_rate: float = 250.0, channels: int = 8) -> None:
        self.sampling_rate = sampling_rate
        self.channels = channels
        self._task: Optional[asyncio.Task] = None
        self._queue: asyncio.Queue = asyncio.Queue(maxsize=100)
        self._running: bool = False

    async def start(self) -> None:
        """Start streaming from the device."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop streaming from the device and clean up resources."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        # flush any remaining items
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def _run(self) -> None:
        """Internal loop to call the driver's `_stream` coroutine."""
        async for item in self._stream():
            # backpressure: drop samples if queue is full
            try:
                await self._queue.put(item)
            except asyncio.CancelledError:
                break
            if not self._running:
                break

    async def __aiter__(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        """Return an asynchronous iterator over timestamped samples."""
        while self._running:
            try:
                item = await self._queue.get()
                yield item
            except asyncio.CancelledError:
                break

    @abstractmethod
    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        """Concrete drivers override this coroutine to produce data.

        This coroutine should yield timestamped data indefinitely while the
        driver is running.  Yields must be produced at intervals consistent
        with the sampling rate.  The base class handles queuing and iteration.
        """

        raise NotImplementedError