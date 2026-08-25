"""Decoder-ready neural window contracts.

A :class:`SignalFrame` represents one timestamped streaming chunk. A
:class:`NeuralWindow` represents exactly one decoder-ready temporal window with
canonical ``(channels, time)`` geometry. Keeping these concepts distinct removes
array-shape guessing from model/runtime boundaries and makes window provenance
replayable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .signal import ClockDomain, QualityFlag


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """Deterministic sample-domain windowing specification."""

    window_samples: int
    stride_samples: int

    def __post_init__(self) -> None:
        if self.window_samples <= 0:
            raise ValueError("window_samples must be positive")
        if self.stride_samples <= 0:
            raise ValueError("stride_samples must be positive")
        if self.stride_samples > self.window_samples:
            raise ValueError("stride_samples cannot exceed window_samples")

    @property
    def overlap_samples(self) -> int:
        return self.window_samples - self.stride_samples

    @classmethod
    def from_seconds(
        cls,
        *,
        sample_rate_hz: float,
        window_seconds: float,
        stride_seconds: float,
    ) -> "WindowSpec":
        if sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if window_seconds <= 0 or stride_seconds <= 0:
            raise ValueError("window_seconds and stride_seconds must be positive")
        return cls(
            window_samples=int(round(window_seconds * sample_rate_hz)),
            stride_samples=int(round(stride_seconds * sample_rate_hz)),
        )


@dataclass(frozen=True, slots=True)
class NeuralWindow:
    """One decoder-ready neural window with explicit geometry and provenance.

    ``data`` is always two-dimensional and channel-major: ``(channels, time)``.
    The runtime adds the batch axis at decoder invocation, yielding
    ``(batch=1, channels, time)`` for raw-window neural decoders.
    """

    stream_id: str
    window_id: int
    data: NDArray[np.generic]
    sample_rate_hz: float
    start_time_ns: int
    end_time_ns: int
    channel_names: tuple[str, ...] = ()
    source_sequence_ids: tuple[int, ...] = ()
    clock_domain: ClockDomain = ClockDomain.UNKNOWN
    quality: QualityFlag = QualityFlag.GOOD
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.stream_id:
            raise ValueError("stream_id must be non-empty")
        if self.window_id < 0:
            raise ValueError("window_id must be >= 0")
        if self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be positive")
        if self.start_time_ns < 0 or self.end_time_ns <= self.start_time_ns:
            raise ValueError("NeuralWindow requires a positive half-open time interval")

        arr = np.asarray(self.data)
        if arr.ndim != 2:
            raise ValueError(
                "NeuralWindow.data must have shape (channels, time); "
                f"received {tuple(arr.shape)}"
            )
        if arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError("NeuralWindow.data cannot contain empty axes")
        if not np.isfinite(arr).all():
            raise ValueError("NeuralWindow.data contains NaN or infinite values")
        if self.channel_names and len(self.channel_names) != arr.shape[0]:
            raise ValueError("channel_names must match the channel axis")
        if any(sequence_id < 0 for sequence_id in self.source_sequence_ids):
            raise ValueError("source_sequence_ids must be non-negative")
        if any(
            current <= previous
            for previous, current in zip(
                self.source_sequence_ids, self.source_sequence_ids[1:]
            )
        ):
            raise ValueError("source_sequence_ids must be strictly increasing")

        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def n_channels(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[1])

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / self.sample_rate_hz

    def as_batch(self) -> NDArray[np.generic]:
        """Return one explicit decoder batch with shape ``(1, channels, time)``."""

        return self.data[np.newaxis, ...]
