#!/usr/bin/env python3
"""Run generic neurOS runtime and scientific validity gates."""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from neuros.cli.config_commands import execute_config
from neuros.quality import (
    BenchmarkManifest,
    QualityThresholds,
    evaluate_runtime_snapshot,
    frequency_selectivity_probe,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Executable neurOS pipeline YAML")
    parser.add_argument(
        "--thresholds", default="configs/quality/ci.yaml", help="Quality threshold YAML"
    )
    parser.add_argument("--report", default=None, help="Optional JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    threshold_raw = yaml.safe_load(Path(args.thresholds).read_text(encoding="utf-8"))
    runtime_raw = threshold_raw["runtime"]
    scientific_raw = threshold_raw["scientific"]
    benchmark_raw = threshold_raw["benchmark"]

    thresholds = QualityThresholds(**runtime_raw)
    snapshot = asyncio.run(
        execute_config(args.config, duration_s=float(benchmark_raw["duration_s"]))
    )
    runtime_gate = evaluate_runtime_snapshot(snapshot, thresholds)

    probes = [
        frequency_selectivity_probe(
            float(frequency),
            sample_rate_hz=float(scientific_raw["sample_rate_hz"]),
            seed=int(benchmark_raw["seed"]),
        )
        for frequency in scientific_raw["frequencies_hz"]
    ]
    scientific_passed = all(
        probe.passed
        and probe.selectivity_ratio >= float(scientific_raw["min_selectivity_ratio"])
        for probe in probes
    )

    manifest = BenchmarkManifest.capture(
        "generic-ci-quality",
        config={
            "pipeline": Path(args.config).read_text(encoding="utf-8"),
            "thresholds": threshold_raw,
        },
        data_fingerprint={
            "synthetic_frequencies_hz": scientific_raw["frequencies_hz"],
            "sample_rate_hz": scientific_raw["sample_rate_hz"],
        },
        seed=int(benchmark_raw["seed"]),
    )
    report = {
        "passed": runtime_gate.passed and scientific_passed,
        "runtime_gate": {
            "passed": runtime_gate.passed,
            "checks": dict(runtime_gate.checks),
            "metrics": dict(runtime_gate.metrics),
            "failures": list(runtime_gate.failures),
        },
        "scientific_gate": {
            "passed": scientific_passed,
            "probes": [asdict(probe) | {"passed": probe.passed} for probe in probes],
        },
        "manifest": manifest.to_dict(),
    }
    text = json.dumps(report, indent=2, sort_keys=True, default=str)
    print(text)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
