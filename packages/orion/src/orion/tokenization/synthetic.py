"""Deterministic synthetic spike sessions with labeled ORION motif ground truth."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from orion.tokenization.events import MotifInterval, SpikeEvent, normalize_events


@dataclass(frozen=True, slots=True)
class SyntheticSpikeSession:
    events: tuple[SpikeEvent, ...]
    motifs: tuple[MotifInterval, ...]
    duration_ns: int
    n_units: int
    seed: int


def _ns(seconds: float) -> int:
    return int(round(seconds * 1_000_000_000.0))


def generate_synthetic_session(
    *,
    seed: int = 0,
    n_units: int = 16,
    repeats: int = 4,
    background_rate_hz: float = 2.0,
) -> SyntheticSpikeSession:
    """Generate Poisson background plus six repeated neural motifs.

    The generator is intentionally mechanistic and labeled. It is not claimed to
    reproduce a specific biological preparation; its purpose is to expose timing,
    population, burst, pause, and assembly structure under controlled ground truth.
    """

    if n_units < 12 or repeats < 2 or background_rate_hz < 0:
        raise ValueError("synthetic benchmark requires >=12 units, >=2 repeats, non-negative rate")
    labels = (
        "burst",
        "synchrony",
        "assembly",
        "leader_chain",
        "pause_rebound",
        "movement_volley",
    )
    motif_spacing_s = 0.45
    block_s = len(labels) * motif_spacing_s + 0.75
    duration_s = repeats * block_s + 1.0
    duration_ns = _ns(duration_s)
    rng = np.random.default_rng(seed)

    events: list[SpikeEvent] = []
    for unit_id in range(n_units):
        expected = background_rate_hz * duration_s
        count = int(rng.poisson(expected))
        times = rng.integers(0, duration_ns, size=count, endpoint=False)
        events.extend(SpikeEvent(int(time), unit_id) for time in times)

    motifs: list[MotifInterval] = []
    pause_windows: list[tuple[int, int, int]] = []
    for repeat in range(repeats):
        block_start_s = 0.5 + repeat * block_s
        for label_index, label in enumerate(labels):
            start_ns = _ns(block_start_s + label_index * motif_spacing_s)
            end_ns = start_ns + _ns(0.18)

            if label == "burst":
                unit = repeat % n_units
                for index in range(7):
                    events.append(SpikeEvent(start_ns + index * _ns(0.004), unit))
                units = (unit,)

            elif label == "synchrony":
                units = tuple(range(6))
                for unit in units:
                    events.append(
                        SpikeEvent(start_ns + int(rng.integers(0, _ns(0.003))), unit)
                    )

            elif label == "assembly":
                units = tuple(range(6, 12))
                for volley in range(4):
                    base = start_ns + volley * _ns(0.025)
                    for unit in units:
                        events.append(
                            SpikeEvent(base + int(rng.integers(0, _ns(0.004))), unit)
                        )

            elif label == "leader_chain":
                units = tuple(range(8))
                for index, unit in enumerate(units):
                    events.append(SpikeEvent(start_ns + index * _ns(0.006), unit))

            elif label == "pause_rebound":
                unit = (repeat + 3) % n_units
                units = (unit,)
                pause_windows.append((start_ns, start_ns + _ns(0.12), unit))
                rebound = start_ns + _ns(0.125)
                for index in range(5):
                    events.append(SpikeEvent(rebound + index * _ns(0.005), unit))

            else:  # movement_volley
                units = tuple(range(4, 14))
                for index, unit in enumerate(units):
                    # Broad onset volley with a deterministic propagation gradient.
                    events.append(
                        SpikeEvent(
                            start_ns + index * _ns(0.002) + int(rng.integers(0, _ns(0.002))),
                            unit,
                        )
                    )

            motifs.append(MotifInterval(label, start_ns, end_ns, units))

    if pause_windows:
        filtered: list[SpikeEvent] = []
        for event in events:
            suppress = any(
                unit == event.unit_id and start <= event.timestamp_ns < end
                for start, end, unit in pause_windows
            )
            if not suppress:
                filtered.append(event)
        events = filtered
        # Re-add rebound spikes that were deliberately inside the removal logic's
        # surrounding motif construction only when they fall after the pause.
        for start, _, unit in pause_windows:
            rebound = start + _ns(0.125)
            for index in range(5):
                candidate = SpikeEvent(rebound + index * _ns(0.005), unit)
                if candidate not in events:
                    events.append(candidate)

    return SyntheticSpikeSession(
        events=normalize_events(events),
        motifs=tuple(motifs),
        duration_ns=duration_ns,
        n_units=n_units,
        seed=seed,
    )


def jitter_events(
    events: Iterable[SpikeEvent],
    *,
    std_ms: float,
    seed: int,
    duration_ns: int | None = None,
) -> tuple[SpikeEvent, ...]:
    if std_ms < 0:
        raise ValueError("std_ms must be non-negative")
    rng = np.random.default_rng(seed)
    limit = duration_ns if duration_ns is not None else None
    result = []
    for event in events:
        jitter = int(rng.normal(0.0, std_ms * 1_000_000.0))
        timestamp = max(0, event.timestamp_ns + jitter)
        if limit is not None:
            timestamp = min(limit - 1, timestamp)
        result.append(SpikeEvent(timestamp, event.unit_id))
    return normalize_events(result)


def dropout_units(
    events: Iterable[SpikeEvent],
    *,
    probability: float,
    n_units: int,
    seed: int,
) -> tuple[SpikeEvent, ...]:
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    dropped = set(np.flatnonzero(rng.random(n_units) < probability).tolist())
    return normalize_events(event for event in events if event.unit_id not in dropped)
