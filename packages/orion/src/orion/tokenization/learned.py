"""Lightweight learned ORION tokenizers with explicit fit/encode boundaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from neuros.contracts import SignalFrame
from orion.contracts import NeuroTokenBatch, TokenizerManifest
from orion.tokenization.events import SpikeEvent, events_from_frames, normalize_events


def _rasterize(
    events: Sequence[SpikeEvent],
    *,
    bin_ns: int,
    n_units: int | None = None,
    origin_ns: int | None = None,
    end_ns: int | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    events = normalize_events(events)
    inferred_units = max((event.unit_id for event in events), default=-1) + 1
    units = max(inferred_units, n_units or 0)
    if units <= 0:
        return np.zeros((0, 0), dtype=np.float32), np.zeros(0, dtype=np.int64), 0
    if not events and origin_ns is None:
        return np.zeros((0, units), dtype=np.float32), np.zeros(0, dtype=np.int64), units
    origin = int(origin_ns if origin_ns is not None else events[0].timestamp_ns)
    finish = int(end_ns if end_ns is not None else events[-1].timestamp_ns + bin_ns)
    bins = max(1, int(np.ceil((finish - origin) / bin_ns)))
    raster = np.zeros((bins, units), dtype=np.float32)
    for event in events:
        index = min(bins - 1, max(0, (event.timestamp_ns - origin) // bin_ns))
        raster[int(index), event.unit_id] += 1.0
    timestamps = origin + np.arange(bins, dtype=np.int64) * bin_ns
    return raster, timestamps, units


def _windows(raster: np.ndarray, timestamps: np.ndarray, window_bins: int) -> tuple[np.ndarray, np.ndarray]:
    if raster.shape[0] < window_bins:
        return np.zeros((0, window_bins * raster.shape[1]), dtype=np.float32), np.zeros(0, dtype=np.int64)
    vectors = np.stack(
        [raster[index : index + window_bins].reshape(-1) for index in range(raster.shape[0] - window_bins + 1)]
    )
    times = timestamps[: len(vectors)]
    return vectors.astype(np.float32), times.astype(np.int64)


class VQMotifTokenizer:
    """Vector-quantize local spike raster motifs with deterministic Lloyd k-means.

    This is intentionally a small representation-learning baseline, not a VQ-VAE.
    It tests whether a learned local codebook earns its complexity before ORION
    introduces a neural encoder/decoder around the quantizer.
    """

    TOKEN_OFFSET = 10_000

    def __init__(
        self,
        *,
        bin_ms: float = 5.0,
        window_ms: float = 50.0,
        codebook_size: int = 16,
        seed: int = 0,
        max_iter: int = 50,
    ) -> None:
        if bin_ms <= 0 or window_ms < bin_ms or codebook_size < 2 or max_iter < 1:
            raise ValueError("invalid VQ motif configuration")
        self.bin_ns = int(bin_ms * 1_000_000.0)
        self.window_bins = max(1, int(round(window_ms / bin_ms)))
        self.codebook_size = codebook_size
        self.seed = seed
        self.max_iter = max_iter
        self.codebook_: np.ndarray | None = None
        self.n_units_: int | None = None
        self._manifest = TokenizerManifest(
            "vq_motif",
            "1.0.0",
            parameters={
                "bin_ms": bin_ms,
                "window_ms": window_ms,
                "codebook_size": codebook_size,
                "seed": seed,
                "max_iter": max_iter,
            },
        )

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def fit_events(self, events: Sequence[SpikeEvent]) -> "VQMotifTokenizer":
        raster, timestamps, units = _rasterize(events, bin_ns=self.bin_ns)
        vectors, _ = _windows(raster, timestamps, self.window_bins)
        if len(vectors) < self.codebook_size:
            raise ValueError(
                f"Need at least {self.codebook_size} raster windows to fit VQMotifTokenizer"
            )
        rng = np.random.default_rng(self.seed)
        initial_indices = rng.choice(len(vectors), size=self.codebook_size, replace=False)
        centers = vectors[initial_indices].astype(np.float64, copy=True)
        for _ in range(self.max_iter):
            distances = ((vectors[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            labels = distances.argmin(axis=1)
            updated = centers.copy()
            for cluster in range(self.codebook_size):
                members = vectors[labels == cluster]
                if len(members):
                    updated[cluster] = members.mean(axis=0)
                else:
                    # Deterministically reseed empty clusters with the point farthest
                    # from its assigned center.
                    nearest = distances[np.arange(len(vectors)), labels]
                    updated[cluster] = vectors[int(np.argmax(nearest))]
            if np.allclose(updated, centers, rtol=0.0, atol=1e-7):
                centers = updated
                break
            centers = updated
        self.codebook_ = centers.astype(np.float32)
        self.n_units_ = units
        return self

    def fit(self, frames: list[SignalFrame]) -> "VQMotifTokenizer":
        return self.fit_events(events_from_frames(frames))

    def encode_events(self, events: Sequence[SpikeEvent]) -> NeuroTokenBatch:
        if self.codebook_ is None or self.n_units_ is None:
            raise RuntimeError("VQMotifTokenizer must be fit before encode")
        events = normalize_events(events)
        if not events:
            return NeuroTokenBatch(
                token_ids=np.zeros(0, dtype=np.int64),
                timestamps_ns=np.zeros(0, dtype=np.int64),
                metadata={"tokenizer_id": "vq_motif", "input_events": 0},
            )
        raster, timestamps, _ = _rasterize(
            events, bin_ns=self.bin_ns, n_units=self.n_units_
        )
        vectors, times = _windows(raster, timestamps, self.window_bins)
        if not len(vectors):
            return NeuroTokenBatch(
                token_ids=np.zeros(0, dtype=np.int64),
                timestamps_ns=np.zeros(0, dtype=np.int64),
                metadata={"tokenizer_id": "vq_motif", "input_events": len(events)},
            )
        distances = ((vectors[:, None, :] - self.codebook_[None, :, :]) ** 2).sum(axis=2)
        labels = distances.argmin(axis=1).astype(np.int64)
        nearest = np.sqrt(distances[np.arange(len(vectors)), labels])
        usage = np.bincount(labels, minlength=self.codebook_size)
        return NeuroTokenBatch(
            token_ids=self.TOKEN_OFFSET + labels,
            timestamps_ns=times,
            side_features={"cluster": labels, "quantization_error": nearest},
            metadata={
                "tokenizer_id": "vq_motif",
                "input_events": len(events),
                "codebook_usage": usage.tolist(),
                "active_codes": int(np.count_nonzero(usage)),
            },
        )

    def encode(self, frames: list[SignalFrame]) -> NeuroTokenBatch:
        return self.encode_events(events_from_frames(frames))


@dataclass(frozen=True, slots=True)
class AssemblySummary:
    component: int
    top_units: tuple[int, ...]
    threshold: float


class AssemblyTokenizer:
    """Discover reproducible population assemblies using deterministic SVD."""

    TOKEN_OFFSET = 12_000
    ON = 0
    PEAK = 1
    OFF = 2

    def __init__(
        self,
        *,
        bin_ms: float = 20.0,
        n_assemblies: int = 4,
        threshold_std: float = 1.5,
        top_units: int = 5,
    ) -> None:
        if bin_ms <= 0 or n_assemblies < 1 or threshold_std <= 0 or top_units < 1:
            raise ValueError("invalid assembly configuration")
        self.bin_ns = int(bin_ms * 1_000_000.0)
        self.n_assemblies = n_assemblies
        self.threshold_std = threshold_std
        self.top_units = top_units
        self.components_: np.ndarray | None = None
        self.thresholds_: np.ndarray | None = None
        self.n_units_: int | None = None
        self.summaries_: tuple[AssemblySummary, ...] = ()
        self._manifest = TokenizerManifest(
            "assembly",
            "1.0.0",
            parameters={
                "bin_ms": bin_ms,
                "n_assemblies": n_assemblies,
                "threshold_std": threshold_std,
                "top_units": top_units,
            },
        )

    @property
    def manifest(self) -> TokenizerManifest:
        return self._manifest

    def fit_events(self, events: Sequence[SpikeEvent]) -> "AssemblyTokenizer":
        raster, _, units = _rasterize(events, bin_ns=self.bin_ns)
        if raster.shape[0] < 2 or units < 1:
            raise ValueError("Need at least two active bins to fit AssemblyTokenizer")
        centered = raster - raster.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        count = min(self.n_assemblies, vt.shape[0])
        components = vt[:count].copy()
        # Fix SVD sign ambiguity so fits are stable across implementations.
        for index in range(count):
            pivot = int(np.argmax(np.abs(components[index])))
            if components[index, pivot] < 0:
                components[index] *= -1
        activation = np.abs(raster @ components.T)
        thresholds = activation.mean(axis=0) + self.threshold_std * activation.std(axis=0)
        summaries = []
        for index, component in enumerate(components):
            top = np.argsort(np.abs(component))[::-1][: self.top_units]
            summaries.append(
                AssemblySummary(index, tuple(int(unit) for unit in top), float(thresholds[index]))
            )
        self.components_ = components.astype(np.float32)
        self.thresholds_ = thresholds.astype(np.float32)
        self.n_units_ = units
        self.summaries_ = tuple(summaries)
        return self

    def fit(self, frames: list[SignalFrame]) -> "AssemblyTokenizer":
        return self.fit_events(events_from_frames(frames))

    def encode_events(self, events: Sequence[SpikeEvent]) -> NeuroTokenBatch:
        if self.components_ is None or self.thresholds_ is None or self.n_units_ is None:
            raise RuntimeError("AssemblyTokenizer must be fit before encode")
        events = normalize_events(events)
        if not events:
            return NeuroTokenBatch(
                token_ids=np.zeros(0, dtype=np.int64),
                timestamps_ns=np.zeros(0, dtype=np.int64),
                metadata={"tokenizer_id": "assembly", "input_events": 0},
            )
        raster, timestamps, _ = _rasterize(
            events, bin_ns=self.bin_ns, n_units=self.n_units_
        )
        activation = np.abs(raster @ self.components_.T)
        records: list[tuple[int, int, int, float]] = []
        for component in range(self.components_.shape[0]):
            active = activation[:, component] >= self.thresholds_[component]
            index = 0
            while index < len(active):
                if not active[index]:
                    index += 1
                    continue
                start = index
                while index + 1 < len(active) and active[index + 1]:
                    index += 1
                end = index
                local = activation[start : end + 1, component]
                peak = start + int(np.argmax(local))
                records.extend(
                    [
                        (int(timestamps[start]), component, self.ON, float(activation[start, component])),
                        (int(timestamps[peak]), component, self.PEAK, float(activation[peak, component])),
                        (int(timestamps[end] + self.bin_ns), component, self.OFF, float(activation[end, component])),
                    ]
                )
                index += 1
        records.sort(key=lambda item: (item[0], item[1], item[2]))
        ids = np.asarray(
            [self.TOKEN_OFFSET + component * 3 + kind for _, component, kind, _ in records],
            dtype=np.int64,
        )
        return NeuroTokenBatch(
            token_ids=ids,
            timestamps_ns=np.asarray([item[0] for item in records], dtype=np.int64),
            side_features={
                "component": np.asarray([item[1] for item in records], dtype=np.int64),
                "kind": np.asarray([item[2] for item in records], dtype=np.int8),
                "activation": np.asarray([item[3] for item in records], dtype=np.float32),
            },
            metadata={
                "tokenizer_id": "assembly",
                "input_events": len(events),
                "assemblies": [
                    {
                        "component": summary.component,
                        "top_units": summary.top_units,
                        "threshold": summary.threshold,
                    }
                    for summary in self.summaries_
                ],
            },
        )

    def encode(self, frames: list[SignalFrame]) -> NeuroTokenBatch:
        return self.encode_events(events_from_frames(frames))
