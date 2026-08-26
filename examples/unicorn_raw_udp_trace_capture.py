#!/usr/bin/env python3
"""Capture shareable Unicorn raw-UDP interface diagnostics without raw EEG.

The process keeps datagrams only in memory long enough to reduce them to timing,
packet-shape, counter, validation, and battery diagnostics. The output JSON does
not contain EEG channel values or original packet bytes.

`--source-kind user_declared_physical` records the operator's statement. It is
not cryptographic evidence that a physical headset produced the stream.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import time

from neuros.drivers import (
    analyze_unicorn_raw_udp_trace,
    compare_unicorn_trace_to_nominal_contract,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19745)
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument(
        "--source-kind",
        choices=("unknown", "synthetic", "user_declared_physical"),
        default="unknown",
    )
    parser.add_argument("--source-label", default="")
    parser.add_argument("--byte-order", choices=("<", ">", "=", "!"), default="<")
    parser.add_argument("--rate-tolerance-hz", type=float, default=15.0)
    parser.add_argument("--output", default="unicorn-raw-udp-trace-summary.json")
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    if args.seconds <= 0:
        raise SystemExit("--seconds must be positive")
    if args.rate_tolerance_hz <= 0:
        raise SystemExit("--rate-tolerance-hz must be positive")

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.port))
    sock.settimeout(0.100)
    records: list[tuple[float, bytes]] = []
    start = time.perf_counter()
    deadline = start + args.seconds
    try:
        while time.perf_counter() < deadline:
            try:
                payload, _ = sock.recvfrom(4096)
            except socket.timeout:
                continue
            records.append((time.perf_counter() - start, payload))
    finally:
        sock.close()

    summary = analyze_unicorn_raw_udp_trace(
        records,
        source_kind=args.source_kind,
        source_label=args.source_label,
        byte_order=args.byte_order,
    )
    comparison = compare_unicorn_trace_to_nominal_contract(
        summary,
        rate_tolerance_hz=args.rate_tolerance_hz,
    )
    payload = {
        "schema": "neuros.unicorn_raw_udp_capture_receipt.v2",
        "summary": summary.to_dict(),
        "nominal_contract_comparison": comparison.to_dict(),
        "capture_requested_seconds": args.seconds,
        "raw_packets_persisted": False,
        "evidence_boundary": (
            "Capture receipt contains reduced interface diagnostics only. Raw datagrams "
            "exist transiently in process memory and are not persisted by this tool."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "packets": summary.packet_count,
        "decoded": summary.decoded_packet_count,
        "raw_packets_persisted": False,
        "passed_diagnostic_checks": comparison.passed_diagnostic_checks,
    }, sort_keys=True))
    if summary.packet_count == 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
