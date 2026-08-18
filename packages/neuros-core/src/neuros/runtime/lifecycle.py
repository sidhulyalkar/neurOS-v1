"""Lifecycle state and events for neurOS runtimes."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


class RuntimeState(str, Enum):
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"
    DEGRADED = "degraded"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    event: str
    state: RuntimeState
    monotonic_time_ns: int = field(default_factory=time.monotonic_ns)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
