"""Explicit queue/backpressure semantics for real-time neurOS pipelines."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any


class OverflowPolicy(str, Enum):
    """Behavior when a bounded runtime queue is full."""

    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    FAIL = "fail"


@dataclass(slots=True)
class QueueStats:
    """Mutable telemetry for a runtime queue edge."""

    accepted: int = 0
    dropped: int = 0
    high_water_mark: int = 0

    def observe_depth(self, depth: int) -> None:
        self.high_water_mark = max(self.high_water_mark, depth)


async def put_with_policy(
    queue: asyncio.Queue,
    item: Any,
    *,
    policy: OverflowPolicy,
    stats: QueueStats | None = None,
) -> bool:
    """Insert an item according to an explicit overflow policy."""

    if stats is None:
        stats = QueueStats()

    if policy is OverflowPolicy.BLOCK:
        await queue.put(item)
        stats.accepted += 1
        stats.observe_depth(queue.qsize())
        return True

    try:
        queue.put_nowait(item)
    except asyncio.QueueFull:
        if policy is OverflowPolicy.DROP_NEWEST:
            stats.dropped += 1
            stats.observe_depth(queue.qsize())
            return False
        if policy is OverflowPolicy.FAIL:
            raise
        if policy is OverflowPolicy.DROP_OLDEST:
            try:
                queue.get_nowait()
                queue.task_done()
            except asyncio.QueueEmpty:
                pass
            stats.dropped += 1
            queue.put_nowait(item)
        else:
            raise ValueError(f"Unsupported overflow policy: {policy}")

    stats.accepted += 1
    stats.observe_depth(queue.qsize())
    return True
