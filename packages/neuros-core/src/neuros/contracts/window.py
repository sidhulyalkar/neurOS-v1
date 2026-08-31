"""Decoder-ready neural window contracts.

A :class:`SignalFrame` represents one timestamped streaming chunk. A
:class:`NeuralWindow` represents exactly one decoder-ready temporal window with
canonical ``(channels, time)`` geometry. Keeping these concepts distinct removes
array-shape guessing from model/runtime boundaries and makes window provenance
replayable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray

from .signal import ClockDomain, QualityFlag, _freeze_metadata_mapping


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

    Window arrays and metadata are detached from caller-owned mutable state at
    construction. ``data`` is stored read-only, matching :class:`SignalFrame`'s
    immutable software-observation boundary.
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
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ValueError("stream_id must be non-empty")
        if isinstance(self.window_id, (bool, np.bool_)) or not isinstance(
            self.window_id, Integral
        ):
            raise TypeError("window_id must be an integer")
        if int(self.window_id) < 0:
            raise ValueError("window_id must be >= 0")
        object.__setattr__(self, "window_id", int(self.window_id))

        if isinstance(self.sample_rate_hz, (bool, np.bool_)) or not isinstance(
            self.sample_rate_hz, Real
        ):
            raise TypeError("sample_rate_hz must be a real numeric scalar")
        sample_rate_hz = float(self.sample_rate_hz)
        if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
        object.__setattr__(self, "sample_rate_hz", sample_rate_hz)

        for field_name, value in (
            ("start_time_ns", self.start_time_ns),
            ("end_time_ns", self.end_time_ns),
        ):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
                raise TypeError(f"{field_name} must be an integer")
        start_time_ns = int(self.start_time_ns)
        end_time_ns = int(self.end_time_ns)
        if start_time_ns < 0 or end_time_ns <= start_time_ns:
            raise ValueError("NeuralWindow requires a positive half-open time interval")
        object.__setattr__(self, "start_time_ns", start_time_ns)
        object.__setattr__(self, "end_time_ns", end_time_ns)

        arr = np.array(self.data, copy=True, subok=False)
        if arr.ndim != 2:
            raise ValueError(
                "NeuralWindow.data must have shape (channels, time); "
                f"received {tuple(arr.shape)}"
            )
        if arr.shape[0] <= 0 or arr.shape[1] <= 0:
            raise ValueError("NeuralWindow.data cannot contain empty axes")
        if arr.dtype.kind not in "biufc":
            raise TypeError(
                "NeuralWindow.data must use a boolean or numeric dtype; "
                f"received {arr.dtype}"
            )
        if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
            raise ValueError("NeuralWindow.data contains NaN or infinite values")

        channel_names = tuple(self.channel_names)
        if any(not isinstance(name, str) or not name.strip() for name in channel_names):
            raise ValueError("channel_names must contain non-empty strings")
        if channel_names and len(channel_names) != arr.shape[0]:
            raise ValueError("channel_names must match the channel axis")
        object.__setattr__(self, "channel_names", channel_names)

        source_sequence_ids: list[int] = []
        for sequence_id in self.source_sequence_ids:
            if isinstance(sequence_id, (bool, np.bool_)) or not isinstance(
                sequence_id, Integral
            ):
                raise TypeError("source_sequence_ids must contain integers")
            resolved = int(sequence_id)
            if resolved < 0:
                raise ValueError("source_sequence_ids must be non-negative")
            source_sequence_ids.append(resolved)
        source_sequence_tuple = tuple(source_sequence_ids)
        if any(
            current <= previous
            for previous, current in zip(
                source_sequence_tuple, source_sequence_tuple[1:]
            )
        ):
            raise ValueError("source_sequence_ids must be strictly increasing")
        object.__setattr__(self, "source_sequence_ids", source_sequence_tuple)

        object.__setattr__(self, "clock_domain", ClockDomain(self.clock_domain))
        quality_value = int(self.quality)
        if quality_value < 0:
            raise ValueError("quality must be non-negative")
        known_mask = 0
        for flag in QualityFlag:
            known_mask |= int(flag)
        if quality_value & ~known_mask:
            raise ValueError("quality contains undefined QualityFlag bits")
        object.__setattr__(self, "quality", QualityFlag(quality_value))

        arr.setflags(write=False)
        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "metadata", _freeze_metadata_mapping(self.metadata))

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
