"""Feature fusion operator for multi-modal neurOS pipelines."""

from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import numpy as np

from neuros.agents.base_agent import BaseAgent
from neuros.runtime import OverflowPolicy, QueueStats, put_with_policy


class FusionAgent(BaseAgent):
    """Concatenate the most recent feature vector from every modality."""

    def __init__(
        self,
        input_queues: List[asyncio.Queue],
        output_queue: asyncio.Queue,
        name: Optional[str] = None,
        *,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        queue_stats: QueueStats | None = None,
    ) -> None:
        super().__init__(name=name or "FusionAgent")
        if not input_queues:
            raise ValueError("FusionAgent requires at least one input queue")
        self.input_queues = input_queues
        self.output_queue = output_queue
        self.overflow_policy = overflow_policy
        self.queue_stats = queue_stats or QueueStats()
        self._latest: List[Optional[Tuple[float, np.ndarray]]] = [None] * len(input_queues)

    async def run(self) -> None:
        try:
            while True:
                tasks = {
                    asyncio.create_task(q.get()): idx
                    for idx, q in enumerate(self.input_queues)
                }
                done, pending = await asyncio.wait(
                    tasks.keys(), return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    idx = tasks[task]
                    try:
                        timestamp, features = task.result()
                        self._latest[idx] = (float(timestamp), np.asarray(features))
                    finally:
                        self.input_queues[idx].task_done()
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
                if all(item is not None for item in self._latest):
                    complete = [item for item in self._latest if item is not None]
                    latest_ts = max(ts for ts, _ in complete)
                    fused = np.concatenate([feat for _, feat in complete], axis=0)
                    await put_with_policy(
                        self.output_queue,
                        (latest_ts, fused),
                        policy=self.overflow_policy,
                        stats=self.queue_stats,
                    )
        except asyncio.CancelledError:
            return
