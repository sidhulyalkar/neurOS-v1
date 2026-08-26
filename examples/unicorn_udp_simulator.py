#!/usr/bin/env python3
"""Publish synthetic Unicorn-compatible raw or Bandpower datagrams over UDP.

This is a game-development substitution endpoint. It reproduces the declared
neurOS interface contracts and deterministic fault schedules, not physical
Bluetooth radio timing. Python/OS scheduling is therefore not presented as a
clock-certified 250 Hz hardware measurement.

Examples:

    # Raw 17-float / 68-byte Unicorn-compatible packets at nominal 250 Hz.
    python examples/unicorn_udp_simulator.py --mode raw --port 19745

    # Bandpower 70-value ASCII reference payloads at nominal 25 Hz after its
    # analysis-window warm-up.
    python examples/unicorn_udp_simulator.py --mode bandpower --port 19746

    # Exercise deterministic failures without opening a socket.
    python examples/unicorn_udp_simulator.py --mode raw \
        --fault-profile mixed-torture --dry-run-packets 200
"""
from __future__ import annotations

import argparse
import heapq
import json
import socket
import time

from neuros.drivers import (
    FAULT_PROFILES,
    UnicornBandpowerUdpStreamSimulator,
    UnicornRawUdpStreamSimulator,
    decode_unicorn_bandpower_ascii,
    decode_unicorn_udp_scan,
    get_unicorn_udp_fault_profile,
)


def _build_stream(mode: str, seed: int, fault_profile: str, byte_order: str):
    profile = get_unicorn_udp_fault_profile(fault_profile)
    if mode == "raw":
        return UnicornRawUdpStreamSimulator(seed=seed, fault_profile=profile, byte_order=byte_order)
    return UnicornBandpowerUdpStreamSimulator(seed=seed, fault_profile=profile)


def _dry_run(stream, mode: str, packets: int, byte_order: str) -> dict:
    emitted = 0
    source_updates = 0
    duplicates = 0
    reordered = 0
    delayed = 0
    counters: list[int] = []
    first_nominal_time_s: float | None = None
    while source_updates < packets:
        datagrams = stream.next_datagrams()
        source_updates += 1
        for datagram in datagrams:
            if first_nominal_time_s is None:
                first_nominal_time_s = datagram.nominal_time_s
            emitted += 1
            duplicates += int(datagram.duplicate_ordinal > 0)
            reordered += int(any(fault.startswith("reordered") for fault in datagram.faults))
            delayed += int("delay" in datagram.faults)
            if mode == "raw":
                values = decode_unicorn_udp_scan(datagram.payload, byte_order=byte_order)
                if values.shape != (17,):
                    raise AssertionError("raw UDP payload did not decode to 17 values")
                counters.append(int(round(float(values[15]))))
            else:
                values = decode_unicorn_bandpower_ascii(datagram.payload)
                if values.shape != (70,):
                    raise AssertionError("Bandpower UDP payload did not decode to 70 values")
    for datagram in stream.flush():
        if first_nominal_time_s is None:
            first_nominal_time_s = datagram.nominal_time_s
        emitted += 1
        if mode == "raw":
            values = decode_unicorn_udp_scan(datagram.payload, byte_order=byte_order)
            counters.append(int(round(float(values[15]))))
    return {
        "schema": "neuros.unicorn_udp_simulator.dry_run.v2",
        "mode": mode,
        "source_updates": source_updates,
        "emitted_datagrams": emitted,
        "initial_delay_s": stream.initial_delay_s,
        "first_nominal_time_s": first_nominal_time_s,
        "dropped_or_held_updates": source_updates - len(set(counters)) if mode == "raw" else None,
        "duplicates": duplicates,
        "reordered_datagrams": reordered,
        "delayed_datagrams": delayed,
        "first_counter": counters[0] if counters else None,
        "last_counter": counters[-1] if counters else None,
        "synthetic": True,
        "evidence_boundary": "Interface/fault simulation only; not physical Bluetooth timing evidence.",
    }


def _run_live(stream, *, host: str, port: int, duration_s: float) -> None:
    destination = (host, port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    start = time.perf_counter()
    next_source_deadline = start + stream.initial_delay_s
    pending: list[tuple[float, int, bytes]] = []
    insertion_order = 0
    sent = 0
    source_updates = 0
    try:
        while True:
            now = time.perf_counter()
            if duration_s > 0 and now - start >= duration_s:
                break

            if now >= next_source_deadline:
                for datagram in stream.next_datagrams():
                    release_abs = start + datagram.release_time_s
                    heapq.heappush(pending, (release_abs, insertion_order, datagram.payload))
                    insertion_order += 1
                source_updates += 1
                next_source_deadline = (
                    start + stream.initial_delay_s + source_updates * stream.interval_s
                )

            now = time.perf_counter()
            while pending and pending[0][0] <= now:
                _, _, payload = heapq.heappop(pending)
                sock.sendto(payload, destination)
                sent += 1

            next_wakeup = next_source_deadline
            if pending:
                next_wakeup = min(next_wakeup, pending[0][0])
            sleep_s = next_wakeup - time.perf_counter()
            if sleep_s > 0:
                time.sleep(min(sleep_s, 0.01))
    except KeyboardInterrupt:
        pass
    finally:
        # Flush is a shutdown cleanup operation, not timing evidence. A packet
        # held solely for a synthetic reorder pair is emitted immediately here.
        for datagram in stream.flush():
            sock.sendto(datagram.payload, destination)
            sent += 1
        sock.close()
        print(json.dumps({
            "source_updates": source_updates,
            "sent_datagrams": sent,
            "destination": f"{host}:{port}",
            "initial_delay_s": stream.initial_delay_s,
            "synthetic": True,
        }, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("raw", "bandpower"), default="raw")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=19745)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fault-profile", choices=tuple(sorted(FAULT_PROFILES)), default="pristine")
    parser.add_argument("--byte-order", choices=("<", ">", "=", "!"), default="<")
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; 0 runs until interrupted")
    parser.add_argument("--dry-run-packets", type=int, default=0)
    args = parser.parse_args()

    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be in [1, 65535]")
    if args.duration < 0:
        raise SystemExit("--duration must be non-negative")
    if args.dry_run_packets < 0:
        raise SystemExit("--dry-run-packets must be non-negative")

    stream = _build_stream(args.mode, args.seed, args.fault_profile, args.byte_order)
    cadence_hz = 1.0 / stream.interval_s
    print(json.dumps({
        "mode": args.mode,
        "nominal_cadence_hz": cadence_hz,
        "initial_delay_s": stream.initial_delay_s,
        "fault_profile": args.fault_profile,
        "synthetic": True,
        "transport_evidence_class": "synthetic_assumption",
        "physical_radio_timing_claim": False,
    }, sort_keys=True))

    if args.dry_run_packets:
        print(json.dumps(_dry_run(stream, args.mode, args.dry_run_packets, args.byte_order), sort_keys=True))
        return
    _run_live(stream, host=args.host, port=args.port, duration_s=args.duration)


if __name__ == "__main__":
    main()
