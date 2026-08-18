"""Device source adapter for the legacy agent runtime."""

from __future__ import annotations

import asyncio
from typing import Any

from neuros.agents.base_agent import BaseAgent
from neuros.runtime import OverflowPolicy, QueueStats, put_with_policy


class DeviceAgent(BaseAgent):
    def __init__(
        self,
        driver: Any,
        output_queue: asyncio.Queue,
        *,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        queue_stats: QueueStats | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name=kwargs.get("name", "DeviceAgent"))
        self.driver = driver
        self.output_queue = output_queue
        self.overflow_policy = overflow_policy
        self.queue_stats = queue_stats or QueueStats()
        self.running = False

    async def run(self) -> None:
        self.logger.info("Starting driver…")
        await self.driver.start()
        self.running = True
        try:
            async for timestamp, data in self.driver:
                if not self.running:
                    break
                accepted = await put_with_policy(
                    self.output_queue,
                    (timestamp, data),
                    policy=self.overflow_policy,
                    stats=self.queue_stats,
                )
                if not accepted:
                    self.logger.debug("Raw queue full; newest sample dropped")
        finally:
            await self.driver.stop()

    async def stop(self) -> None:
        self.running = False
        await self.driver.stop()
