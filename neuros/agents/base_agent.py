"""
Base agent class for neurOS.

An agent encapsulates an independent task within the pipeline.  Each agent
defines an asynchronous :meth:`run` method that produces data or side
effects.  The orchestrator coordinates agents by running their
coroutines concurrently and passing data between them via asyncio queues.
"""

from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Optional


class BaseAgent(ABC):
    """Abstract base class for all agents."""

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name or self.__class__.__name__
        self.logger = logging.getLogger(self.name)

    @abstractmethod
    async def run(self) -> None:
        """Run the agent asynchronously."""
        raise NotImplementedError