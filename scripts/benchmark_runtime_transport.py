#!/usr/bin/env python3
"""Non-gating benchmark for neurOS persistent process transports.

The benchmark compares the already-qualified pickle worker with the experimental
shared-memory worker using identical NumPy round trips. It deliberately excludes
process startup from measured samples via warmups, alternates transport order
between repeats, verifies every returned payload, and records distributions.

It is evidence about one machine/runtime combination, not a release threshold.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from neuros.runtime.process_worker import PersistentProcessWorker
from neuros.runtime.shared_process_worker import SharedMemoryProcessWorker


class IdentityOperator:
    def transform(self, item: Any) -> Any:
        return item


def _source_revision() -> str:
    explicit = os.environ.get("GITHUB_SHA")
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


def _iterations_for(payload_bytes: int) -> int:
    if payload_bytes <= 64 * 1024:
        return 50
    if payload_bytes <= 1024 * 1024:
        return 20
    return 8


def _make_worker(transport: str, payload_bytes: int):
    if transport == "pickle":
        return PersistentProcessWorker(
            f"bench-{transport}",
            IdentityOperator(),
            execution_timeout_s=15.0,
        )
    capacity = payload_bytes + 4096
    return SharedMemoryProcessWorker(
        f"bench-{transport}",
        IdentityOperator(),
        execution_timeout_s=15.0,
        request_capacity_bytes=capacity,
        response_capacity_bytes=capacity,
    )


async def _measure_once(
    transport: str,
    payload: np.ndarray,
    *,
    warmups: int,
    iterations: int,
) -> list[float]:
    worker = _make_worker(transport, int(payload.nbytes))
    try:
        for _ in range(warmups):
            call = await worker.invoke("transform", payload)
            if not np.array_equal(call.result, payload):
                raise RuntimeError(f"{transport} warmup changed the payload")

        samples_ms: list[float] = []
        for _ in range(iterations):
            started = time.perf_counter_ns()
            call = await worker.invoke("transform", payload)
            elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            if not np.array_equal(call.result, payload):
                raise RuntimeError(f"{transport} changed the payload")
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
    aggregates: dict[tuple[int, str], list[float]] = {}

    for payload_bytes in payload_sizes:
        if payload_bytes <= 0 or payload_bytes % np.dtype(np.float32).itemsize:
            raise ValueError("payload sizes must be positive multiples of four bytes")
        elements = payload_bytes // np.dtype(np.float32).itemsize
        payload = np.arange(elements, dtype=np.float32)
        iterations = fixed_iterations or _iterations_for(payload_bytes)

        for repeat in range(repeats):
            order = ("pickle", "shared_memory") if repeat % 2 == 0 else (
                "shared_memory",
                "pickle",
            )
            for transport in order:
                samples_ms = await _measure_once(
                    transport,
                    payload,
                    warmups=warmups,
                    iterations=iterations,
                )
                aggregates.setdefault((payload_bytes, transport), []).extend(samples_ms)
                records.append(
                    {
                        "payload_bytes": payload_bytes,
                        "transport": transport,
                        "repeat": repeat,
                        "warmups": warmups,
                        "iterations": iterations,
                        "latency_ms": samples_ms,
                    }
                )

    summary: list[dict[str, Any]] = []
    for (payload_bytes, transport), values in sorted(aggregates.items()):
        mean_ms = statistics.fmean(values)
        p50_ms = _percentile(values, 0.50)
        p95_ms = _percentile(values, 0.95)
        p99_ms = _percentile(values, 0.99)
        round_trip_mib = (2.0 * payload_bytes) / (1024.0 * 1024.0)
        effective_mib_s = round_trip_mib / (mean_ms / 1000.0)
        summary.append(
            {
                "payload_bytes": payload_bytes,
                "transport": transport,
                "samples": len(values),
                "mean_ms": mean_ms,
                "p50_ms": p50_ms,
                "p95_ms": p95_ms,
                "p99_ms": p99_ms,
                "effective_round_trip_mib_s": effective_mib_s,
            }
        )

    return {
        "schema": "neuros.runtime_transport_benchmark.v1",
        "source_revision": _source_revision(),
        "python": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "numpy": np.__version__,
        "repeats": repeats,
        "warmups": warmups,
        "payload_sizes": payload_sizes,
        "summary": summary,
        "records": records,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--payload-bytes",
        type=int,
        nargs="+",
        default=[4 * 1024, 64 * 1024, 1024 * 1024, 8 * 1024 * 1024],
    )
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--output", type=Path, default=Path("transport-benchmark.json"))
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
    print(json.dumps(result["summary"], indent=2))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
