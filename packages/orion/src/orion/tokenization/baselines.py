"""Interpretable ORION spike tokenization baselines."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from neuros.contracts import SignalFrame
from orion.contracts import NeuroTokenBatch, TokenizerManifest
from orion.tokenization.events import SpikeEvent, events_from_frames, normalize_events


def _make_batch(
    tokenizer_id: str,
    token_ids: Iterable[int],
    timestamps_ns: Iterable[int],
    *,
    side_features: dict[str, Iterable[Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> NeuroTokenBatch:
    ids = np.asarray(list(token_ids), dtype=np.int64)
    times = np.asarray(list(timestamps_ns), dtype=np.int64)
    side = {
        key: np.asarray(list(values)) for key, values in (side_features or {}).items()
    }
    return NeuroTokenBatch(
        token_ids=ids,
        timestamps_ns=times,
        side_features=side,
        metadata={"tokenizer_id": tokenizer_id, **(metadata or {})},
    )


class EventSpikeTokenizer:
    """One token per sorted spike event, preserving exact event timing."""

    UNIT_OFFSET = 1000

    def __init__(self) -> None:
        self._manifest = TokenizerManifest("event", "1.0.0", parameters={})

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def encode_events(self, events: Sequence[SpikeEvent]) -> NeuroTokenBatch:
        events = normalize_events(events)
        return _make_batch(
            "event",
            (self.UNIT_OFFSET + event.unit_id for event in events),
            (event.timestamp_ns for event in events),
            side_features={"unit_id": (event.unit_id for event in events)},
            metadata={"input_events": len(events)},
        )

    def encode(self, frames: list[SignalFrame]) -> NeuroTokenBatch:
        return self.encode_events(events_from_frames(frames))


class BinnedCountTokenizer:
    """Collapse spikes into unit-specific count tokens within fixed time bins."""

    UNIT_OFFSET = 2000

    def __init__(self, bin_ms: float = 10.0) -> None:
        if bin_ms <= 0:
            raise ValueError("bin_ms must be positive")
        self.bin_ns = int(round(bin_ms * 1_000_000.0))
        self._manifest = TokenizerManifest(
            "binned_count", "1.0.0", parameters={"bin_ms": bin_ms}
        )

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def encode_events(self, events: Sequence[SpikeEvent]) -> NeuroTokenBatch:
        events = normalize_events(events)
        if not events:
            return _make_batch("binned_count", [], [], metadata={"input_events": 0})
        origin = events[0].timestamp_ns
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for event in events:
            bin_index = (event.timestamp_ns - origin) // self.bin_ns
            counts[(int(bin_index), event.unit_id)] += 1
        keys = sorted(counts)
        return _make_batch(
            "binned_count",
            (self.UNIT_OFFSET + unit_id for _, unit_id in keys),
            (origin + bin_index * self.bin_ns for bin_index, _ in keys),
            side_features={
                "unit_id": (unit_id for _, unit_id in keys),
                "count": (counts[key] for key in keys),
                "bin_index": (bin_index for bin_index, _ in keys),
            },
            metadata={"input_events": len(events), "bin_ns": self.bin_ns},
        )

    def encode(self, frames: list[SignalFrame]) -> NeuroTokenBatch:
        return self.encode_events(events_from_frames(frames))


class ISIRelativeTimeTokenizer:
    """Represent global relative timing with quantized WAIT + unit SPIKE tokens."""

    WAIT_OFFSET = 10
    SPIKE_OFFSET = 3000

    def __init__(
        self,
        *,
        min_wait_ms: float = 0.25,
        max_wait_ms: float = 2000.0,
        wait_bins: int = 32,
    ) -> None:
        if min_wait_ms <= 0 or max_wait_ms <= min_wait_ms or wait_bins < 2:
            raise ValueError("invalid wait bin configuration")
        self.edges_ns = np.geomspace(
            min_wait_ms * 1_000_000.0,
            max_wait_ms * 1_000_000.0,
            wait_bins + 1,
        )
        self._manifest = TokenizerManifest(
            "isi_relative",
            "1.0.0",
            parameters={
                "min_wait_ms": min_wait_ms,
                "max_wait_ms": max_wait_ms,
                "wait_bins": wait_bins,
            },
        )

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def _wait_bin(self, delta_ns: int) -> int:
        if delta_ns <= 0:
            return 0
        return int(np.clip(np.searchsorted(self.edges_ns, delta_ns, side="right") - 1, 0, len(self.edges_ns) - 2))

    def encode_events(self, events: Sequence[SpikeEvent]) -> NeuroTokenBatch:
        events = normalize_events(events)
        token_ids: list[int] = []
        timestamps: list[int] = []
        kinds: list[int] = []
        units: list[int] = []
        delta_ms: list[float] = []
        local_isi_ms: list[float] = []
        last_global: int | None = None
        last_by_unit: dict[int, int] = {}
        for event in events:
            delta = 0 if last_global is None else event.timestamp_ns - last_global
            if delta > 0:
                token_ids.append(self.WAIT_OFFSET + self._wait_bin(delta))
                timestamps.append(event.timestamp_ns)
                kinds.append(0)
                units.append(-1)
                delta_ms.append(delta / 1_000_000.0)
                local_isi_ms.append(np.nan)
            token_ids.append(self.SPIKE_OFFSET + event.unit_id)
            timestamps.append(event.timestamp_ns)
            kinds.append(1)
            units.append(event.unit_id)
            delta_ms.append(delta / 1_000_000.0)
            local = event.timestamp_ns - last_by_unit[event.unit_id] if event.unit_id in last_by_unit else -1
            local_isi_ms.append(local / 1_000_000.0 if local >= 0 else np.nan)
            last_global = event.timestamp_ns
            last_by_unit[event.unit_id] = event.timestamp_ns
        return _make_batch(
            "isi_relative",
            token_ids,
            timestamps,
            side_features={
                "kind": kinds,
                "unit_id": units,
                "delta_ms": delta_ms,
                "unit_isi_ms": local_isi_ms,
            },
            metadata={"input_events": len(events)},
        )

    def encode(self, frames: list[SignalFrame]) -> NeuroTokenBatch:
        return self.encode_events(events_from_frames(frames))


@dataclass(frozen=True, slots=True)
class _SemanticToken:
    timestamp_ns: int
    token_id: int
    kind: int
    unit_id: int
    count: int
    duration_ns: int


class BurstTokenizer:
    """Compress within-unit rapid firing into burst/pause/rebound semantics."""

    EVENT_OFFSET = 4000
    BURST_OFFSET = 5000
    PAUSE_OFFSET = 6000
    REBOUND_OFFSET = 7000

    def __init__(
        self,
        *,
        burst_isi_ms: float = 12.0,
        min_burst_spikes: int = 3,
        pause_ms: float = 150.0,
    ) -> None:
        if burst_isi_ms <= 0 or min_burst_spikes < 2 or pause_ms <= burst_isi_ms:
            raise ValueError("invalid burst configuration")
        self.burst_isi_ns = int(burst_isi_ms * 1_000_000.0)
        self.min_burst_spikes = min_burst_spikes
        self.pause_ns = int(pause_ms * 1_000_000.0)
        self._manifest = TokenizerManifest(
            "burst",
            "1.0.0",
            parameters={
                "burst_isi_ms": burst_isi_ms,
                "min_burst_spikes": min_burst_spikes,
                "pause_ms": pause_ms,
            },
        )

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def encode_events(self, events: Sequence[SpikeEvent]) -> NeuroTokenBatch:
        events = normalize_events(events)
        by_unit: dict[int, list[SpikeEvent]] = defaultdict(list)
        for event in events:
            by_unit[event.unit_id].append(event)
        semantic: list[_SemanticToken] = []
        for unit_id, unit_events in by_unit.items():
            clusters: list[list[SpikeEvent]] = []
            current: list[SpikeEvent] = []
            for event in unit_events:
                if current and event.timestamp_ns - current[-1].timestamp_ns > self.burst_isi_ns:
                    clusters.append(current)
                    current = []
                current.append(event)
            if current:
                clusters.append(current)

            previous_end: int | None = None
            for cluster in clusters:
                start = cluster[0].timestamp_ns
                end = cluster[-1].timestamp_ns
                if previous_end is not None and start - previous_end >= self.pause_ns:
                    semantic.append(
                        _SemanticToken(
                            previous_end + self.pause_ns,
                            self.PAUSE_OFFSET + unit_id,
                            2,
                            unit_id,
                            0,
                            start - previous_end,
                        )
                    )
                    semantic.append(
                        _SemanticToken(start, self.REBOUND_OFFSET + unit_id, 3, unit_id, len(cluster), 0)
                    )
                if len(cluster) >= self.min_burst_spikes:
                    semantic.append(
                        _SemanticToken(
                            start,
                            self.BURST_OFFSET + unit_id,
                            1,
                            unit_id,
                            len(cluster),
                            end - start,
                        )
                    )
                else:
                    semantic.extend(
                        _SemanticToken(event.timestamp_ns, self.EVENT_OFFSET + unit_id, 0, unit_id, 1, 0)
                        for event in cluster
                    )
                previous_end = end
        semantic.sort(key=lambda token: (token.timestamp_ns, token.kind, token.unit_id))
        return _make_batch(
            "burst",
            (token.token_id for token in semantic),
            (token.timestamp_ns for token in semantic),
            side_features={
                "kind": (token.kind for token in semantic),
                "unit_id": (token.unit_id for token in semantic),
                "count": (token.count for token in semantic),
                "duration_ns": (token.duration_ns for token in semantic),
            },
            metadata={"input_events": len(events)},
        )

    def encode(self, frames: list[SignalFrame]) -> NeuroTokenBatch:
        return self.encode_events(events_from_frames(frames))


class SynchronyPacketTokenizer:
    """Collapse small-window population coactivation into synchrony packets."""

    EVENT_OFFSET = 8000
    PACKET_TOKEN = 9000

    def __init__(self, *, window_ms: float = 5.0, min_active_units: int = 3) -> None:
        if window_ms <= 0 or min_active_units < 2:
            raise ValueError("invalid synchrony configuration")
        self.window_ns = int(window_ms * 1_000_000.0)
        self.min_active_units = min_active_units
        self._manifest = TokenizerManifest(
            "synchrony",
            "1.0.0",
            parameters={"window_ms": window_ms, "min_active_units": min_active_units},
        )

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def encode_events(self, events: Sequence[SpikeEvent]) -> NeuroTokenBatch:
        events = normalize_events(events)
        ids: list[int] = []
        times: list[int] = []
        kinds: list[int] = []
        active_counts: list[int] = []
        units_hash: list[int] = []
        index = 0
        while index < len(events):
            start = events[index].timestamp_ns
            end_index = index
            active: set[int] = set()
            while end_index < len(events) and events[end_index].timestamp_ns - start <= self.window_ns:
                active.add(events[end_index].unit_id)
                end_index += 1
            if len(active) >= self.min_active_units:
                ids.append(self.PACKET_TOKEN)
                times.append(start)
                kinds.append(1)
                active_counts.append(len(active))
                units_hash.append(hash(tuple(sorted(active))) & 0x7FFFFFFF)
                index = end_index
            else:
                event = events[index]
                ids.append(self.EVENT_OFFSET + event.unit_id)
                times.append(event.timestamp_ns)
                kinds.append(0)
                active_counts.append(1)
                units_hash.append(event.unit_id)
                index += 1
        return _make_batch(
            "synchrony",
            ids,
            times,
            side_features={
                "kind": kinds,
                "active_unit_count": active_counts,
                "unit_signature": units_hash,
            },
            metadata={"input_events": len(events), "window_ns": self.window_ns},
        )

    def encode(self, frames: list[SignalFrame]) -> NeuroTokenBatch:
        return self.encode_events(events_from_frames(frames))
