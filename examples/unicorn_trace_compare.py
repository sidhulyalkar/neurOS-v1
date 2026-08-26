#!/usr/bin/env python3
"""Compare two Unicorn raw-UDP diagnostic receipts without loading raw EEG.

Inputs may be either a ``neuros.unicorn_raw_udp_capture_receipt`` JSON document
or the nested ``neuros.unicorn_raw_udp_trace_summary`` object itself. The output
is descriptive only. It intentionally has no built-in pass threshold because
physical-vs-synthetic equivalence tolerances require measured evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import json
from pathlib import Path

from neuros.drivers import UnicornRawUdpTraceSummary, compare_unicorn_trace_summaries


def _load_summary(path: Path) -> UnicornRawUdpTraceSummary:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    summary = payload.get("summary", payload)
    if not isinstance(summary, dict):
        raise ValueError(f"{path}: summary must be a JSON object")
    allowed = {field.name for field in fields(UnicornRawUdpTraceSummary)}
    kwargs = {name: summary[name] for name in allowed if name in summary}
    try:
        return UnicornRawUdpTraceSummary(**kwargs)
    except TypeError as exc:
        raise ValueError(f"{path}: incompatible trace-summary schema: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reference = _load_summary(args.reference)
    candidate = _load_summary(args.candidate)
    report = compare_unicorn_trace_summaries(reference, candidate).to_dict()
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
