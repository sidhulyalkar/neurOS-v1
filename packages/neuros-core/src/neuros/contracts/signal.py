"""Canonical neural signal contracts used across neurOS.

These dataclasses intentionally live in ``neuros-core`` so drivers, runtimes,
models, storage backends, and ORION can exchange neural data without importing
one another's concrete implementations.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class ClockDomain(str, Enum):
    """Clock used by a timestamp."""

    DEVICE = "device"
    HOST_MONOTONIC = "host_monotonic"
    SYNCHRONIZED = "synchronized"
    UNKNOWN = "unknown"


class QualityFlag(IntFlag):
    """Composable signal-quality flags."""

    GOOD = 0
    DROPPED_SAMPLES = auto()
    CLOCK_UNCERTAIN = auto()
    SATURATED = auto()
    CLIPPED = auto()
    DISCONNECTED_CHANNEL = auto()
    ARTIFACT_SUSPECTED = auto()


@dataclass(frozen=True, slots=True)
class StreamDescriptor:
    """Static metadata describing a neural or behavioral stream."""

    stream_id: str
    modality: str
    sample_rate_hz: float
    channel_names: tuple[str, ...] = ()
    channel_types: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    device: str | None = None
    manufacturer: str | None = None
    clock_domain: ClockDomain = ClockDomain.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.channel_types and len(self.channel_types) != len(self.channel_names):
            raise ValueError("channel_types must match channel_names length")
        if self.units and len(self.units) != len(self.channel_names):
            raise ValueError("units must match channel_names length")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


@dataclass(frozen=True, slots=True)
class SignalFrame:
    """A timestamped chunk of neural data with explicit clock semantics."""

    stream_id: str
    sequence_id: int
    data: NDArray[np.generic]
    sample_rate_hz: float
    host_receive_time_ns: int
    device_time_ns: int | None = None
    synchronized_time_ns: int | None = None
    clock_domain: ClockDomain = ClockDomain.UNKNOWN
    quality: QualityFlag = QualityFlag.GOOD
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if self.sequence_id < 0:
            raise ValueError("sequence_id must be >= 0")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        arr = np.asarray(self.data)
        if arr.ndim == 0:
            raise ValueError("SignalFrame.data must have at least one dimension")
        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def timestamp_ns(self) -> int:
        """Best available timestamp in nanoseconds."""
        if self.synchronized_time_ns is not None:
            return self.synchronized_time_ns
        if self.device_time_ns is not None:
            return self.device_time_ns
        return self.host_receive_time_ns

    @property
    def timestamp_seconds(self) -> float:
        return self.timestamp_ns / 1_000_000_000.0

    @classmethod
    def from_legacy(
        cls,
        *,
        stream_id: str,
        sequence_id: int,
        timestamp_seconds: float,
        data: NDArray[np.generic],
        sample_rate_hz: float,
        clock_domain: ClockDomain = ClockDomain.UNKNOWN,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SignalFrame":
        """Convert the legacy ``(timestamp_seconds, ndarray)`` representation."""
        timestamp_ns = int(timestamp_seconds * 1_000_000_000)
        return cls(
            stream_id=stream_id,
            sequence_id=sequence_id,
            data=np.asarray(data),
            sample_rate_hz=sample_rate_hz,
            host_receive_time_ns=time.monotonic_ns(),
            device_time_ns=timestamp_ns,
            clock_domain=clock_domain,
            metadata=metadata or {},
        )
