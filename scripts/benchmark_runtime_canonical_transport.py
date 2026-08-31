#!/usr/bin/env python3
"""Non-gating transport benchmark across realistic neurOS payload classes.

This experiment complements the raw-array crossover benchmark. It distinguishes
transport *support* from transport *latency*: canonical neurOS contracts may be
intentionally unpickleable because immutable provenance uses MappingProxyType.
Unsupported pickle workloads are therefore recorded as such instead of being
coerced into arrays merely to manufacture a two-column timing comparison.

For supported workloads, process startup is excluded with warmups, every result
is semantically verified, transport order alternates across repeats, and the
artifact records raw samples plus numeric-array and shared-manifest sizes.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import pickle
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from neuros.contracts import DecoderOutput, NeuralWindow, SignalFrame
from neuros.runtime.process_worker import PersistentProcessWorker
from neuros.runtime.shared_process_worker import SharedMemoryProcessWorker
from neuros.runtime.transport import SharedMemoryMailbox


class IdentityOperator:
    def transform(self, item: Any) -> Any:
        return item


@dataclass(frozen=True)
class Workload:
    name: str
    payload: Any
    verify: Callable[[Any, Any], None]


def _source_revision() -> str:
    explicit = os.environ.get("BENCH_SOURCE_SHA")
    if explicit:
        return explicit
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = (len(ordered) - 1) * q
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _iterations_for(array_bytes: int) -> int:
    if array_bytes <= 64 * 1024:
        return 30
    if array_bytes <= 1024 * 1024:
        return 12
    return 5


def _verify_array(expected: Any, actual: Any) -> None:
    if not isinstance(actual, np.ndarray) or not np.array_equal(actual, expected):
        raise RuntimeError("ndarray round trip changed payload")


def _verify_signal_frame(expected: SignalFrame, actual: Any) -> None:
    if not isinstance(actual, SignalFrame):
        raise RuntimeError(f"expected SignalFrame, got {type(actual).__name__}")
    fields = (
        "stream_id",
        "sequence_id",
        "sample_rate_hz",
        "host_receive_time_ns",
        "device_time_ns",
        "synchronized_time_ns",
        "clock_domain",
        "quality",
    )
    for field in fields:
        if getattr(actual, field) != getattr(expected, field):
            raise RuntimeError(f"SignalFrame field {field} changed")
    if not np.array_equal(actual.data, expected.data):
        raise RuntimeError("SignalFrame data changed")
    if dict(actual.metadata) != dict(expected.metadata):
        raise RuntimeError("SignalFrame metadata changed")


def _verify_neural_window(expected: NeuralWindow, actual: Any) -> None:
    if not isinstance(actual, NeuralWindow):
        raise RuntimeError(f"expected NeuralWindow, got {type(actual).__name__}")
    fields = (
        "stream_id",
        "window_id",
        "sample_rate_hz",
        "start_time_ns",
        "end_time_ns",
        "channel_names",
        "source_sequence_ids",
        "clock_domain",
        "quality",
    )
    for field in fields:
        if getattr(actual, field) != getattr(expected, field):
            raise RuntimeError(f"NeuralWindow field {field} changed")
    if not np.array_equal(actual.data, expected.data):
        raise RuntimeError("NeuralWindow data changed")
    if dict(actual.metadata) != dict(expected.metadata):
        raise RuntimeError("NeuralWindow metadata changed")


def _verify_decoder_output(expected: DecoderOutput, actual: Any) -> None:
    if not isinstance(actual, DecoderOutput):
        raise RuntimeError(f"expected DecoderOutput, got {type(actual).__name__}")
    scalar_fields = (
        "confidence",
        "uncertainty",
        "model_id",
        "model_version",
        "inference_time_ns",
    )
    for field in scalar_fields:
        if getattr(actual, field) != getattr(expected, field):
            raise RuntimeError(f"DecoderOutput field {field} changed")
    for field in ("prediction", "probabilities", "logits", "embedding"):
        expected_value = getattr(expected, field)
        actual_value = getattr(actual, field)
        if expected_value is None:
            if actual_value is not None:
                raise RuntimeError(f"DecoderOutput field {field} changed")
        elif not np.array_equal(actual_value, expected_value):
            raise RuntimeError(f"DecoderOutput array {field} changed")
    if dict(actual.metadata) != dict(expected.metadata):
        raise RuntimeError("DecoderOutput metadata changed")


def _workloads(target_array_bytes: int) -> tuple[Workload, ...]:
    if target_array_bytes <= 0 or target_array_bytes % 32:
        raise ValueError("target array bytes must be a positive multiple of 32")

    elements = target_array_bytes // np.dtype(np.float32).itemsize
    raw = np.arange(elements, dtype=np.float32)

    channels = 8
    samples = elements // channels
    frame_data = np.arange(samples * channels, dtype=np.float32).reshape(samples, channels)
    frame = SignalFrame(
        stream_id="eeg",
        sequence_id=7,
        data=frame_data,
        sample_rate_hz=250.0,
        host_receive_time_ns=1_000_000,
        device_time_ns=995_000,
        metadata={
            "axis_order": ("sample", "channel"),
            "channel_names": tuple(f"EEG{index:02d}" for index in range(channels)),
            "session": {"id": "canonical-bench", "trial": 3},
        },
    )

    window_data = np.arange(channels * samples, dtype=np.float32).reshape(channels, samples)
    window = NeuralWindow(
        stream_id="eeg",
        window_id=11,
        data=window_data,
        sample_rate_hz=250.0,
        start_time_ns=1_000_000,
        end_time_ns=1_000_000 + int(samples / 250.0 * 1_000_000_000),
        channel_names=tuple(f"EEG{index:02d}" for index in range(channels)),
        source_sequence_ids=(5, 6, 7),
        metadata={"pipeline": "canonical-bench", "nested": {"fold": 2}},
    )

    fixed_array_bytes = 8 + 4 * 4 + 4 * 4
    embedding_bytes = max(4, target_array_bytes - fixed_array_bytes)
    embedding_elements = max(1, embedding_bytes // 4)
    output = DecoderOutput(
        prediction=np.array([2], dtype=np.int64),
        confidence=0.82,
        uncertainty=0.18,
        probabilities=np.array([0.05, 0.08, 0.82, 0.05], dtype=np.float32),
        logits=np.array([-1.5, -1.0, 2.2, -1.4], dtype=np.float32),
        embedding=np.arange(embedding_elements, dtype=np.float32),
        model_id="canonical-bench-model",
        model_version="v1",
        inference_time_ns=420_000,
        metadata={"window_id": 11, "calibration": {"temperature": 1.0}},
    )

    return (
        Workload("ndarray", raw, _verify_array),
        Workload("signal_frame", frame, _verify_signal_frame),
        Workload("neural_window", window, _verify_neural_window),
        Workload("decoder_output", output, _verify_decoder_output),
    )


def _numeric_array_bytes(value: Any) -> int:
    if isinstance(value, np.ndarray):
        return int(value.nbytes)
    if isinstance(value, SignalFrame):
        return int(value.data.nbytes)
    if isinstance(value, NeuralWindow):
        return int(value.data.nbytes)
    if isinstance(value, DecoderOutput):
        total = 0
        for field in ("prediction", "probabilities", "logits", "embedding"):
            item = getattr(value, field)
            if isinstance(item, np.ndarray):
                total += int(item.nbytes)
        return total
    raise TypeError(f"unknown benchmark payload {type(value)!r}")


def _pickle_support(payload: Any) -> tuple[bool, str | None]:
    try:
        pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


def _shared_representation_metrics(payload: Any) -> tuple[int, int]:
    array_bytes = _numeric_array_bytes(payload)
    capacity = max(64 * 1024, array_bytes + 64 * 1024)
    box = SharedMemoryMailbox(capacity)
    try:
        envelope = box.encode(payload, lease_id=1)
        manifest_bytes = len(
            json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        return int(envelope["bytes_used"]), manifest_bytes
    finally:
        box.close_and_unlink()


def _make_worker(transport: str, payload: Any):
    if transport == "pickle":
        return PersistentProcessWorker(
            f"bench-{transport}",
            IdentityOperator(),
            execution_timeout_s=15.0,
        )
    array_bytes = _numeric_array_bytes(payload)
    capacity = max(64 * 1024, array_bytes + 64 * 1024)
    return SharedMemoryProcessWorker(
        f"bench-{transport}",
        IdentityOperator(),
        execution_timeout_s=15.0,
        request_capacity_bytes=capacity,
        response_capacity_bytes=capacity,
    )


async def _measure_once(
    transport: str,
    workload: Workload,
    *,
    warmups: int,
    iterations: int,
) -> list[float]:
    worker = _make_worker(transport, workload.payload)
    try:
        for _ in range(warmups):
            call = await worker.invoke("transform", workload.payload)
            workload.verify(workload.payload, call.result)

        samples_ms: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            call = await worker.invoke("transform", workload.payload)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            workload.verify(workload.payload, call.result)
            samples_ms.append(elapsed_ms)
        return samples_ms
    finally:
        worker.close()


async def run_benchmark(
    payload_sizes: list[int],
    *,
    repeats: int,
    warmups: int,
    fixed_iterations: int | None,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    aggregates: dict[tuple[str, int, str], list[float]] = {}
    support: list[dict[str, Any]] = []

    for target_array_bytes in payload_sizes:
        for workload in _workloads(target_array_bytes):
            array_bytes = _numeric_array_bytes(workload.payload)
            shared_bytes_used, manifest_bytes = _shared_representation_metrics(workload.payload)
            pickle_supported, pickle_error = _pickle_support(workload.payload)
            support.extend(
                [
                    {
                        "workload": workload.name,
                        "target_array_bytes": target_array_bytes,
                        "numeric_array_bytes": array_bytes,
                        "transport": "pickle",
                        "supported": pickle_supported,
                        "unsupported_reason": pickle_error,
                    },
                    {
                        "workload": workload.name,
                        "target_array_bytes": target_array_bytes,
                        "numeric_array_bytes": array_bytes,
                        "transport": "shared_memory",
                        "supported": True,
                        "unsupported_reason": None,
                    },
                ]
            )
            iterations = fixed_iterations or _iterations_for(array_bytes)
            transports = ["shared_memory"]
            if pickle_supported:
                transports.append("pickle")

            for repeat in range(repeats):
                order = tuple(transports) if repeat % 2 == 0 else tuple(reversed(transports))
                for transport in order:
                    samples_ms = await _measure_once(
                        transport,
                        workload,
                        warmups=warmups,
                        iterations=iterations,
                    )
                    aggregates.setdefault(
                        (workload.name, target_array_bytes, transport), []
                    ).extend(samples_ms)
                    records.append(
                        {
                            "workload": workload.name,
                            "target_array_bytes": target_array_bytes,
                            "numeric_array_bytes": array_bytes,
                            "shared_bytes_used": shared_bytes_used,
                            "shared_manifest_json_bytes": manifest_bytes,
                            "transport": transport,
                            "repeat": repeat,
                            "warmups": warmups,
                            "iterations": iterations,
                            "latency_ms": samples_ms,
                        }
                    )

    summary: list[dict[str, Any]] = []
    for (workload, target_array_bytes, transport), values in sorted(aggregates.items()):
        sample_record = next(
            record
            for record in records
            if record["workload"] == workload
            and record["target_array_bytes"] == target_array_bytes
            and record["transport"] == transport
        )
        mean_ms = statistics.fmean(values)
        summary.append(
            {
                "workload": workload,
                "target_array_bytes": target_array_bytes,
                "numeric_array_bytes": sample_record["numeric_array_bytes"],
                "shared_bytes_used": sample_record["shared_bytes_used"],
                "shared_manifest_json_bytes": sample_record["shared_manifest_json_bytes"],
                "transport": transport,
                "samples": len(values),
                "mean_ms": mean_ms,
                "p50_ms": _percentile(values, 0.50),
                "p95_ms": _percentile(values, 0.95),
                "p99_ms": _percentile(values, 0.99),
            }
        )

    return {
        "schema": "neuros.runtime_canonical_transport_benchmark.v2",
        "source_revision": _source_revision(),
        "semantic_source_revision": os.environ.get("BENCH_SEMANTIC_SHA", "unknown"),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "repeats": repeats,
        "warmups": warmups,
        "target_array_sizes": payload_sizes,
        "support": support,
        "summary": summary,
        "records": records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payload-bytes",
        type=int,
        nargs="+",
        default=[16 * 1024, 256 * 1024, 1024 * 1024, 8 * 1024 * 1024],
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument(
        "--output", type=Path, default=Path("canonical-transport-benchmark.json")
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("repeats must be positive and warmups non-negative")
    if args.iterations is not None and args.iterations <= 0:
        raise ValueError("iterations must be positive")
    result = asyncio.run(
        run_benchmark(
            list(args.payload_bytes),
            repeats=args.repeats,
            warmups=args.warmups,
            fixed_iterations=args.iterations,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"support": result["support"], "summary": result["summary"]}, indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
