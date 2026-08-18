import asyncio

import pytest

from neuros.runtime import OverflowPolicy, QueueStats, put_with_policy


@pytest.mark.asyncio
async def test_drop_oldest_keeps_freshest_data():
    queue = asyncio.Queue(maxsize=2)
    stats = QueueStats()
    await put_with_policy(queue, 1, policy=OverflowPolicy.DROP_OLDEST, stats=stats)
    await put_with_policy(queue, 2, policy=OverflowPolicy.DROP_OLDEST, stats=stats)
    accepted = await put_with_policy(queue, 3, policy=OverflowPolicy.DROP_OLDEST, stats=stats)
    assert accepted is True
    assert stats.dropped == 1
    assert [queue.get_nowait(), queue.get_nowait()] == [2, 3]


@pytest.mark.asyncio
async def test_drop_newest_reports_rejection():
    queue = asyncio.Queue(maxsize=1)
    stats = QueueStats()
    await put_with_policy(queue, "old", policy=OverflowPolicy.DROP_NEWEST, stats=stats)
    accepted = await put_with_policy(queue, "new", policy=OverflowPolicy.DROP_NEWEST, stats=stats)
    assert accepted is False
    assert stats.dropped == 1
    assert queue.get_nowait() == "old"
