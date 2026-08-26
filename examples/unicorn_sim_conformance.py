#!/usr/bin/env python3
"""Generate a machine-readable Unicorn Hybrid Black simulation receipt."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from neuros.drivers import run_unicorn_compatibility_suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="unicorn-simulation-compatibility.json")
    args = parser.parse_args()
    report = run_unicorn_compatibility_suite(seed=args.seed)
    payload = report.to_dict()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "report": str(output),
        "schema": payload["schema"],
        "passed": payload["passed"],
        "surfaces": {surface["name"]: surface["passed"] for surface in payload["surfaces"]},
    }, indent=2, sort_keys=True))
    if not report.passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
