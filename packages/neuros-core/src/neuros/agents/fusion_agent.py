"""
Fusion agent for neurOS.

The FusionAgent consumes feature vectors from multiple modalities,
synchronises them in time and produces a single fused feature vector.
It is designed to work within the asynchronous agent architecture of
neurOS, where each modality is processed by its own processing
pipeline.  The fusion strategy is simple concatenation: the latest
feature vector from each modality is concatenated to form the fused
vector.  More sophisticated fusion methods (e.g. attention or
statistical alignment) could be implemented by subclassing this
agent.

When a modality produces features at a slower rate than others, the
FusionAgent will reuse the last available feature for that modality
until a new one arrives.  A fused vector is emitted whenever any
modality produces new features and all modalities have at least one
feature available.  The resulting timestamp corresponds to the time
of the most recently updated modality, but this can be adapted in
future versions.
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Optional, Tuple

import numpy as np

from neuros.agents.base_agent import BaseAgent


class FusionAgent(BaseAgent):
    """Fuse feature vectors from multiple modalities.

    Parameters
    ----------
    input_queues : list of asyncio.Queue
        Queues from which individual modality feature tuples are read.
        Each element of the queue must be a tuple ``(timestamp,
        features)`` where ``features`` is a 1‑D NumPy array.
    output_queue : asyncio.Queue
        Queue to which fused feature tuples are written.  Fused
        elements have the form ``(timestamp, fused_features)`` where
        ``fused_features`` is a 1‑D NumPy array containing the
        concatenated features from all modalities.
    name : str, optional
        Name of the agent for logging and debugging.
    """

    def __init__(self, input_queues: List[asyncio.Queue], output_queue: asyncio.Queue, name: Optional[str] = None) -> None:
        super().__init__(name=name or "FusionAgent")
        self.input_queues = input_queues
        self.output_queue = output_queue
        self._latest: List[Optional[Tuple[float, np.ndarray]]] = [None] * len(input_queues)

    async def run(self) -> None:
        # continually read from input queues and fuse features
        pending: List[asyncio.Task] = []
        try:
            while True:
                # create tasks to read one item from each queue
                tasks = {asyncio.create_task(q.get()): idx for idx, q in enumerate(self.input_queues)}
                done, _ = await asyncio.wait(tasks.keys(), return_when=asyncio.FIRST_COMPLETED)
                # process the completed tasks
                for task in done:
                    idx = tasks[task]
                    try:
                        timestamp, features = task.result()
                    except Exception:
                        continue
                    # store latest features for this modality
                    self._latest[idx] = (timestamp, features)
                # cancel unfinished tasks to avoid leaked awaits
                for task in tasks.keys():
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                # check if all modalities have emitted at least one feature
                if all(item is not None for item in self._latest):
                    # pick the most recent timestamp among modalities
                    latest_ts = max(ts for ts, _ in self._latest if ts is not None)
                    fused = np.concatenate([feat for _, feat in self._latest], axis=0)
                    await self.output_queue.put((latest_ts, fused))
        except asyncio.CancelledError:
            # gracefully exit on cancellation
            return