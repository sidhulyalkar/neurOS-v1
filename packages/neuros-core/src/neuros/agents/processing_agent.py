"""Signal-processing operator for the legacy agent runtime."""

from __future__ import annotations

import asyncio
from typing import Iterable, Optional

from neuros.agents.base_agent import BaseAgent
from neuros.runtime import OverflowPolicy, QueueStats, put_with_policy


class ProcessingAgent(BaseAgent):
    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        filters: Iterable[object],
        extractor: object,
        monitor: Optional[object] = None,
        *,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        queue_stats: QueueStats | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=kwargs.get("name", "ProcessingAgent"))
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.filters = list(filters)
        self.extractor = extractor
        self.running = False
        self.monitor = monitor
        self.overflow_policy = overflow_policy
        self.queue_stats = queue_stats or QueueStats()

    async def run(self) -> None:
        self.running = True
        while self.running:
            try:
                timestamp, data = await self.input_queue.get()
            except asyncio.CancelledError:
                break
            try:
                if self.monitor is not None:
                    try:
                        self.monitor.update(data)
                    except Exception:
                        self.logger.exception("Quality monitor update failed")
                for filt in self.filters:
                    data = filt.apply(data)
                features = self.extractor.extract(data)
                accepted = await put_with_policy(
                    self.output_queue,
                    (timestamp, features),
                    policy=self.overflow_policy,
                    stats=self.queue_stats,
                )
                if not accepted:
                    self.logger.debug("Feature queue full; newest feature dropped")
            finally:
                self.input_queue.task_done()

    async def stop(self) -> None:
        self.running = False
