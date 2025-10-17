"""
Device agent for neurOS.

The :class:`DeviceAgent` wraps a driver and streams data into an output
queue.  It manages starting and stopping the driver and handles backpressure
by dropping samples if the queue is full.
"""

from __future__ import annotations

import asyncio
from typing import Optional

from neuros.drivers.base_driver import BaseDriver
from neuros.agents.base_agent import BaseAgent


class DeviceAgent(BaseAgent):
    def __init__(self, driver: BaseDriver, output_queue: asyncio.Queue, **kwargs) -> None:
        super().__init__(name=kwargs.get("name", "DeviceAgent"))
        self.driver = driver
        self.output_queue = output_queue
        self.running = False

    async def run(self) -> None:
        self.logger.info("Starting driver…")
        await self.driver.start()
        self.running = True
        async for timestamp, data in self.driver:
            if not self.running:
                break
            try:
                # put data into queue; drop if queue is full
                self.output_queue.put_nowait((timestamp, data))
            except asyncio.QueueFull:
                # drop sample to prevent backlog
                self.logger.debug("Queue full – dropping sample")
                pass
        await self.driver.stop()

    async def stop(self) -> None:
        self.running = False
        await self.driver.stop()