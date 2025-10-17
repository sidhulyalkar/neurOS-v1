"""
Processing agent for neurOS.

The :class:`ProcessingAgent` applies a sequence of filters to incoming raw
signals and extracts feature vectors.  It reads from an input queue of
timestamped arrays and writes feature vectors to an output queue.  Filters
must implement an `apply` method and the extractor must provide an
`extract` method.
"""

from __future__ import annotations

import asyncio
from typing import Iterable, List, Optional

import numpy as np

from neuros.agents.base_agent import BaseAgent


class ProcessingAgent(BaseAgent):
    def __init__(
        self,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        filters: Iterable[object],
        extractor: object,
        monitor: Optional[object] = None,
        **kwargs,
    ) -> None:
        super().__init__(name=kwargs.get("name", "ProcessingAgent"))
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.filters = list(filters)
        self.extractor = extractor
        self.running = False
        # optional quality monitor; called with raw data before filtering
        self.monitor = monitor

    async def run(self) -> None:
        self.running = True
        while self.running:
            try:
                timestamp, data = await self.input_queue.get()
            except asyncio.CancelledError:
                break
            # update quality monitor with raw data before filtering
            if self.monitor is not None:
                try:
                    self.monitor.update(data)
                except Exception:
                    # ignore monitoring errors to avoid disrupting pipeline
                    pass
            # apply each filter sequentially
            for filt in self.filters:
                data = filt.apply(data)
            # extract features
            features = self.extractor.extract(data)
            try:
                self.output_queue.put_nowait((timestamp, features))
            except asyncio.QueueFull:
                self.logger.debug("Processing output queue full – dropping features")
                pass

    async def stop(self) -> None:
        self.running = False