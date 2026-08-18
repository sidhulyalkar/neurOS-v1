"""Controlled ORION tokenizer comparison and reporting harness."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from neuros.quality import BenchmarkManifest
from orion.contracts import NeuroTokenBatch
from orion.tokenization.baselines import (
    BinnedCountTokenizer,
    BurstTokenizer,
    EventSpikeTokenizer,
    ISIRelativeTimeTokenizer,
    SynchronyPacketTokenizer,
)
from orion.tokenization.events import MotifInterval, SpikeEvent
from orion.tokenization.learned import AssemblyTokenizer, VQMotifTokenizer
from orion.tokenization.synthetic import (
    SyntheticSpikeSession,
    dropout_units,
    generate_synthetic_session,
    jitter_events,
)


@dataclass(frozen=True, slots=True)
class TokenizerScore:
    tokenizer: str
    input_events: int
    token_count: int
    compression_ratio: float
    token_entropy_bits: float
    encode_ms: float
    motif_decoding_accuracy: float
    jitter_similarity: float
    unit_dropout_similarity: float
    active_token_types: int


def token_entropy(batch: NeuroTokenBatch) -> float:
    if len(batch.token_ids) == 0:
        return 0.0
    _, counts = np.unique(batch.token_ids, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def token_histogram(batch: NeuroTokenBatch, bins: int = 128) -> np.ndarray:
    histogram = np.zeros(bins, dtype=np.float64)
    for token_id in np.asarray(batch.token_ids, dtype=np.int64):
        histogram[int(token_id) % bins] += 1.0
    norm = np.linalg.norm(histogram)
    return histogram / norm if norm > 0 else histogram


def histogram_similarity(a: NeuroTokenBatch, b: NeuroTokenBatch) -> float:
    left = token_histogram(a)
    right = token_histogram(b)
    return float(np.clip(np.dot(left, right), 0.0, 1.0))


def _motif_vector(
    batch: NeuroTokenBatch,
    motif: MotifInterval,
    *,
    hash_bins: int = 64,
    temporal_bins: int = 4,
) -> np.ndarray:
    vector = np.zeros(hash_bins * temporal_bins, dtype=np.float64)
    duration = max(1, motif.end_ns - motif.start_ns)
    for token_id, timestamp in zip(batch.token_ids, batch.timestamps_ns):
        timestamp_int = int(timestamp)
        if not motif.start_ns <= timestamp_int < motif.end_ns:
            continue
        phase = min(
            temporal_bins - 1,
            int((timestamp_int - motif.start_ns) * temporal_bins / duration),
        )
        vector[phase * hash_bins + int(token_id) % hash_bins] += 1.0
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 0 else vector


def motif_decoding_accuracy(
    train_batch: NeuroTokenBatch,
    train_motifs: Sequence[MotifInterval],
    test_batch: NeuroTokenBatch,
    test_motifs: Sequence[MotifInterval],
) -> float:
    """Nearest-centroid motif decoding using only training-session centroids."""

    by_label: dict[str, list[np.ndarray]] = {}
    for motif in train_motifs:
        by_label.setdefault(motif.label, []).append(_motif_vector(train_batch, motif))
    centroids = {
        label: np.mean(np.stack(vectors), axis=0) for label, vectors in by_label.items()
    }
    if not centroids or not test_motifs:
        return 0.0
    correct = 0
    total = 0
    labels = sorted(centroids)
    for motif in test_motifs:
        if motif.label not in centroids:
            continue
        vector = _motif_vector(test_batch, motif)
        distances = {
            label: float(np.sum((vector - centroids[label]) ** 2)) for label in labels
        }
        prediction = min(distances, key=distances.get)
        correct += int(prediction == motif.label)
        total += 1
    return correct / total if total else 0.0


def default_tokenizers(config: Mapping[str, Any] | None = None) -> list[Any]:
    raw = dict(config or {})
    return [
        EventSpikeTokenizer(),
        BinnedCountTokenizer(bin_ms=float(raw.get("bin_ms", 10.0))),
        ISIRelativeTimeTokenizer(),
        BurstTokenizer(
            burst_isi_ms=float(raw.get("burst_isi_ms", 12.0)),
            min_burst_spikes=int(raw.get("min_burst_spikes", 3)),
        ),
        SynchronyPacketTokenizer(
            window_ms=float(raw.get("synchrony_window_ms", 5.0)),
            min_active_units=int(raw.get("min_active_units", 3)),
        ),
        VQMotifTokenizer(
            bin_ms=float(raw.get("vq_bin_ms", 5.0)),
            window_ms=float(raw.get("vq_window_ms", 50.0)),
            codebook_size=int(raw.get("vq_codebook_size", 12)),
            seed=int(raw.get("seed", 0)),
        ),
        AssemblyTokenizer(
            bin_ms=float(raw.get("assembly_bin_ms", 20.0)),
            n_assemblies=int(raw.get("n_assemblies", 4)),
        ),
    ]


def _fit_if_needed(tokenizer: Any, events: Sequence[SpikeEvent]) -> None:
    fit = getattr(tokenizer, "fit_events", None)
    if fit is not None:
        fit(events)


def benchmark_tokenizers(
    *,
    train: SyntheticSpikeSession,
    test: SyntheticSpikeSession,
    tokenizers: Iterable[Any] | None = None,
    jitter_ms: float = 5.0,
    unit_dropout_probability: float = 0.25,
    seed: int = 0,
) -> tuple[TokenizerScore, ...]:
    tokenizers = list(tokenizers or default_tokenizers({"seed": seed}))
    jittered = jitter_events(
        test.events, std_ms=jitter_ms, seed=seed + 101, duration_ns=test.duration_ns
    )
    dropped = dropout_units(
        test.events,
        probability=unit_dropout_probability,
        n_units=test.n_units,
        seed=seed + 202,
    )
    scores: list[TokenizerScore] = []
    for tokenizer in tokenizers:
        _fit_if_needed(tokenizer, train.events)
        train_batch = tokenizer.encode_events(train.events)
        started = time.perf_counter_ns()
        test_batch = tokenizer.encode_events(test.events)
        encode_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        jitter_batch = tokenizer.encode_events(jittered)
        dropout_batch = tokenizer.encode_events(dropped)
        token_count = len(test_batch.token_ids)
        compression = len(test.events) / token_count if token_count else math.inf
        scores.append(
            TokenizerScore(
                tokenizer=tokenizer.manifest.tokenizer_id,
                input_events=len(test.events),
                token_count=token_count,
                compression_ratio=float(compression),
                token_entropy_bits=token_entropy(test_batch),
                encode_ms=encode_ms,
                motif_decoding_accuracy=motif_decoding_accuracy(
                    train_batch, train.motifs, test_batch, test.motifs
                ),
                jitter_similarity=histogram_similarity(test_batch, jitter_batch),
                unit_dropout_similarity=histogram_similarity(test_batch, dropout_batch),
                active_token_types=int(len(np.unique(test_batch.token_ids))),
            )
        )
    return tuple(scores)


def run_synthetic_benchmark(config: Mapping[str, Any]) -> dict[str, Any]:
    synthetic = dict(config.get("synthetic", {}))
    benchmark = dict(config.get("benchmark", {}))
    tokenizer_config = dict(config.get("tokenizers", {}))
    seed = int(benchmark.get("seed", 42))
    train = generate_synthetic_session(
        seed=seed,
        n_units=int(synthetic.get("n_units", 16)),
        repeats=int(synthetic.get("repeats", 4)),
        background_rate_hz=float(synthetic.get("background_rate_hz", 2.0)),
    )
    test = generate_synthetic_session(
        seed=seed + 1,
        n_units=train.n_units,
        repeats=int(synthetic.get("repeats", 4)),
        background_rate_hz=float(synthetic.get("background_rate_hz", 2.0)),
    )
    tokenizers = default_tokenizers(tokenizer_config | {"seed": seed})
    scores = benchmark_tokenizers(
        train=train,
        test=test,
        tokenizers=tokenizers,
        jitter_ms=float(benchmark.get("jitter_ms", 5.0)),
        unit_dropout_probability=float(benchmark.get("unit_dropout_probability", 0.25)),
        seed=seed,
    )
    manifest = BenchmarkManifest.capture(
        "orion-synthetic-tokenization-v1",
        config=config,
        data_fingerprint={
            "train_seed": train.seed,
            "test_seed": test.seed,
            "n_units": train.n_units,
            "train_events": len(train.events),
            "test_events": len(test.events),
            "motif_labels": sorted({motif.label for motif in train.motifs}),
        },
        seed=seed,
    )
    return {
        "manifest": manifest.to_dict(),
        "train": {
            "events": len(train.events),
            "motifs": len(train.motifs),
            "duration_ns": train.duration_ns,
        },
        "test": {
            "events": len(test.events),
            "motifs": len(test.motifs),
            "duration_ns": test.duration_ns,
        },
        "scores": [asdict(score) for score in scores],
    }


def write_benchmark_reports(report: Mapping[str, Any], output_dir: str | Path) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    scores = list(report["scores"])
    if scores:
        with (output / "comparison_table.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scores[0]))
            writer.writeheader()
            writer.writerows(scores)
    lines = [
        "# ORION Tokenizer Cards",
        "",
        "All scores use a separately seeded test session. Fit-requiring tokenizers are fit only on the training session.",
        "",
    ]
    for score in scores:
        lines.extend(
            [
                f"## {score['tokenizer']}",
                "",
                f"- tokens: {score['token_count']} from {score['input_events']} spikes",
                f"- compression: {score['compression_ratio']:.3f}x",
                f"- entropy: {score['token_entropy_bits']:.3f} bits",
                f"- motif decoding accuracy: {score['motif_decoding_accuracy']:.3f}",
                f"- 5 ms jitter histogram similarity: {score['jitter_similarity']:.3f}",
                f"- unit-dropout histogram similarity: {score['unit_dropout_similarity']:.3f}",
                f"- encode time: {score['encode_ms']:.3f} ms",
                "",
            ]
        )
    (output / "tokenizer_cards.md").write_text("\n".join(lines), encoding="utf-8")
