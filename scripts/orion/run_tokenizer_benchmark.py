#!/usr/bin/env python3
"""Run the controlled ORION synthetic tokenizer benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from orion.tokenization import run_synthetic_benchmark, write_benchmark_reports


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config", nargs="?", default="configs/orion/tokenization_smoke.yaml"
    )
    parser.add_argument("--output", default="reports/orion/tokenization-smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    raw = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("benchmark config must be a mapping")
    report = run_synthetic_benchmark(raw)
    write_benchmark_reports(report, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
