"""Canonical spike-event schemas and neurOS SignalFrame translation for ORION."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from neuros.contracts import SignalFrame


@dataclass(frozen=True, order=True, slots=True)
class SpikeEvent:
    timestamp_ns: int
    unit_id: int

    def __post_init__(self) -> None:
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")
        if self.unit_id < 0:
            raise ValueError("unit_id must be non-negative")


@dataclass(frozen=True, slots=True)
class MotifInterval:
    label: str
    start_ns: int
    end_ns: int
    units: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("motif label must be non-empty")
        if self.start_ns < 0 or self.end_ns <= self.start_ns:
            raise ValueError("motif interval must have positive duration")


def normalize_events(events: Iterable[SpikeEvent]) -> tuple[SpikeEvent, ...]:
    """Return events in a deterministic timestamp/unit order."""
    return tuple(sorted(events, key=lambda event: (event.timestamp_ns, event.unit_id)))


def events_from_frames(frames: Sequence[SignalFrame]) -> tuple[SpikeEvent, ...]:
    """Translate supported neurOS spike representations into events.

    Supported conventions:
    - ``metadata['representation'] == 'spike_event'`` with ``metadata['unit_id']``.
    - ``metadata['representation'] == 'spike_counts'`` with a one-dimensional
      count vector and optional ``metadata['unit_ids']``.

    This deliberately refuses to infer spikes from arbitrary continuous signals.
    Spike detection/sorting belongs upstream and must be explicit in provenance.
    """

    events: list[SpikeEvent] = []
    for frame in frames:
        representation = frame.metadata.get("representation")
        timestamp_ns = int(frame.synchronized_time_ns or frame.device_time_ns or frame.host_receive_time_ns)
        if representation == "spike_event":
            unit_id = frame.metadata.get("unit_id")
            if unit_id is None:
                raise ValueError("spike_event frame requires metadata['unit_id']")
            events.append(SpikeEvent(timestamp_ns=timestamp_ns, unit_id=int(unit_id)))
            continue
        if representation == "spike_counts":
            counts = np.asarray(frame.data)
            if counts.ndim != 1:
                raise ValueError("spike_counts frame data must be one-dimensional")
            unit_ids = frame.metadata.get("unit_ids")
            if unit_ids is None:
                unit_ids = tuple(range(len(counts)))
            if len(unit_ids) != len(counts):
                raise ValueError("unit_ids must align with spike count channels")
            for unit_id, count in zip(unit_ids, counts):
                count_int = int(count)
                if count_int < 0 or not np.isclose(count, count_int):
                    raise ValueError("spike counts must be non-negative integers")
                events.extend(
                    SpikeEvent(timestamp_ns=timestamp_ns, unit_id=int(unit_id))
                    for _ in range(count_int)
                )
            continue
        raise ValueError(
            "ORION spike tokenizers require explicit spike_event or spike_counts SignalFrames"
        )
    return normalize_events(events)


def events_to_frames(
    events: Iterable[SpikeEvent],
    *,
    stream_id: str = "spikes",
    sample_rate_hz: float = 1000.0,
) -> list[SignalFrame]:
    """Create event-style SignalFrames for tests, replay, and benchmark adapters."""
    result: list[SignalFrame] = []
    for sequence_id, event in enumerate(normalize_events(events)):
        result.append(
            SignalFrame(
                stream_id=stream_id,
                sequence_id=sequence_id,
                data=np.asarray([1], dtype=np.int16),
                sample_rate_hz=sample_rate_hz,
                host_receive_time_ns=event.timestamp_ns,
                synchronized_time_ns=event.timestamp_ns,
                metadata={"representation": "spike_event", "unit_id": event.unit_id},
            )
        )
    return result
