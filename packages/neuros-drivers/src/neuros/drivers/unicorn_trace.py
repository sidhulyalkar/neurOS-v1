"""Privacy-safe raw-UDP trace diagnostics for Unicorn simulator calibration.

The analyzer intentionally stores *no EEG sample arrays*. It reduces a sequence
of received 68-byte datagrams to interface/timing diagnostics that hardware owners
can share when validating the simulator: packet shape, arrival cadence, counter
gaps, duplicates, reordering, validation state, and battery range.

A source label is user-declared provenance. neurOS cannot cryptographically prove
that a trace came from physical hardware merely because the caller says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .unicorn_network_sim import RAW_UDP_PAYLOAD_BYTES, decode_unicorn_udp_scan

TraceSourceKind = Literal["unknown", "synthetic", "user_declared_physical"]


@dataclass(frozen=True)
class UnicornRawUdpTraceSummary:
    packet_count: int
    decoded_packet_count: int
    malformed_packet_count: int
    duration_s: float
    estimated_arrival_rate_hz: float | None
    mean_interarrival_ms: float | None
    p95_interarrival_ms: float | None
    first_counter: int | None
    last_counter: int | None
    counter_gap_events: int
    inferred_missing_packets: int
    duplicate_packets: int
    out_of_order_packets: int
    validation_zero_packets: int
    battery_min: float | None
    battery_max: float | None
    source_kind: TraceSourceKind = "unknown"
    source_label: str = ""

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.unicorn_raw_udp_trace_summary.v1",
            "packet_count": self.packet_count,
            "decoded_packet_count": self.decoded_packet_count,
            "malformed_packet_count": self.malformed_packet_count,
            "duration_s": self.duration_s,
            "estimated_arrival_rate_hz": self.estimated_arrival_rate_hz,
            "mean_interarrival_ms": self.mean_interarrival_ms,
            "p95_interarrival_ms": self.p95_interarrival_ms,
            "first_counter": self.first_counter,
            "last_counter": self.last_counter,
            "counter_gap_events": self.counter_gap_events,
            "inferred_missing_packets": self.inferred_missing_packets,
            "duplicate_packets": self.duplicate_packets,
            "out_of_order_packets": self.out_of_order_packets,
            "validation_zero_packets": self.validation_zero_packets,
            "battery_min": self.battery_min,
            "battery_max": self.battery_max,
            "source_kind": self.source_kind,
            "source_label": self.source_label,
            "contains_raw_eeg": False,
            "evidence_boundary": (
                "Wire/timing diagnostics only. Source provenance is user-declared; "
                "this summary is not a hardware certification or a human-EEG record."
            ),
        }


@dataclass(frozen=True)
class UnicornTraceContractComparison:
    """Diagnostic comparison against the nominal public raw-UDP contract."""

    packet_shape_clean: bool
    arrival_rate_near_250hz: bool | None
    counter_monotonic_without_duplicates: bool
    observations: dict[str, object]

    @property
    def passed_diagnostic_checks(self) -> bool:
        rate_ok = True if self.arrival_rate_near_250hz is None else self.arrival_rate_near_250hz
        return self.packet_shape_clean and rate_ok and self.counter_monotonic_without_duplicates

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.unicorn_raw_udp_trace_contract_comparison.v1",
            "passed_diagnostic_checks": self.passed_diagnostic_checks,
            "checks": {
                "packet_shape_clean": self.packet_shape_clean,
                "arrival_rate_near_250hz": self.arrival_rate_near_250hz,
                "counter_monotonic_without_duplicates": self.counter_monotonic_without_duplicates,
            },
            "observations": dict(self.observations),
            "evidence_boundary": (
                "Diagnostic comparison to nominal public interface behavior only; "
                "not certification of physical hardware, Bluetooth, or EEG quality."
            ),
        }


def analyze_unicorn_raw_udp_trace(
    records: Iterable[tuple[float, bytes]],
    *,
    source_kind: TraceSourceKind = "unknown",
    source_label: str = "",
    byte_order: str = "<",
) -> UnicornRawUdpTraceSummary:
    """Reduce timestamped datagrams to shareable interface diagnostics."""

    if source_kind not in {"unknown", "synthetic", "user_declared_physical"}:
        raise ValueError("unsupported source_kind")
    rows = [(float(timestamp), bytes(payload)) for timestamp, payload in records]
    if any(not np.isfinite(timestamp) for timestamp, _ in rows):
        raise ValueError("receive timestamps must be finite")
    timestamps = np.asarray([timestamp for timestamp, _ in rows], dtype=float)
    if timestamps.size > 1 and np.any(np.diff(timestamps) < 0):
        raise ValueError("receive timestamps must be monotonic non-decreasing")

    malformed = 0
    decoded = 0
    first_counter: int | None = None
    last_counter: int | None = None
    gap_events = 0
    missing = 0
    duplicates = 0
    out_of_order = 0
    validation_zero = 0
    batteries: list[float] = []

    for _, payload in rows:
        if len(payload) != RAW_UDP_PAYLOAD_BYTES:
            malformed += 1
            continue
        try:
            values = decode_unicorn_udp_scan(payload, byte_order=byte_order)
        except (ValueError, TypeError):
            malformed += 1
            continue
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            malformed += 1
            continue
        counter_float = float(values[15])
        validation_float = float(values[16])
        counter = int(round(counter_float))
        validation = int(round(validation_float))
        if abs(counter_float - counter) > 0.25 or validation not in {0, 1}:
            malformed += 1
            continue

        decoded += 1
        batteries.append(float(values[14]))
        validation_zero += int(validation == 0)
        if first_counter is None:
            first_counter = counter
            last_counter = counter
            continue
        assert last_counter is not None
        delta = counter - last_counter
        if delta == 0:
            duplicates += 1
        elif delta < 0:
            out_of_order += 1
        else:
            if delta > 1:
                gap_events += 1
                missing += delta - 1
            last_counter = counter

    if timestamps.size >= 2:
        interarrival = np.diff(timestamps)
        duration = float(timestamps[-1] - timestamps[0])
        rate = float((timestamps.size - 1) / duration) if duration > 0 else None
        mean_ms = float(np.mean(interarrival) * 1000.0)
        p95_ms = float(np.percentile(interarrival, 95.0) * 1000.0)
    else:
        duration = 0.0
        rate = None
        mean_ms = None
        p95_ms = None

    return UnicornRawUdpTraceSummary(
        packet_count=len(rows),
        decoded_packet_count=decoded,
        malformed_packet_count=malformed,
        duration_s=duration,
        estimated_arrival_rate_hz=rate,
        mean_interarrival_ms=mean_ms,
        p95_interarrival_ms=p95_ms,
        first_counter=first_counter,
        last_counter=last_counter,
        counter_gap_events=gap_events,
        inferred_missing_packets=missing,
        duplicate_packets=duplicates,
        out_of_order_packets=out_of_order,
        validation_zero_packets=validation_zero,
        battery_min=min(batteries) if batteries else None,
        battery_max=max(batteries) if batteries else None,
        source_kind=source_kind,
        source_label=str(source_label),
    )


def compare_unicorn_trace_to_nominal_contract(
    summary: UnicornRawUdpTraceSummary,
    *,
    rate_tolerance_hz: float = 15.0,
) -> UnicornTraceContractComparison:
    """Run deliberately broad diagnostics against the nominal 250 Hz interface.

    The arrival-rate tolerance is a consumer diagnostic policy, not a published
    Bluetooth jitter specification. Short captures or heavily loaded systems may
    fail this check without proving a headset defect.
    """

    if rate_tolerance_hz <= 0:
        raise ValueError("rate_tolerance_hz must be positive")
    rate = summary.estimated_arrival_rate_hz
    rate_ok = None if rate is None else abs(rate - 250.0) <= rate_tolerance_hz
    counter_clean = summary.duplicate_packets == 0 and summary.out_of_order_packets == 0
    return UnicornTraceContractComparison(
        packet_shape_clean=summary.malformed_packet_count == 0,
        arrival_rate_near_250hz=rate_ok,
        counter_monotonic_without_duplicates=counter_clean,
        observations={
            "estimated_arrival_rate_hz": rate,
            "rate_tolerance_hz": float(rate_tolerance_hz),
            "counter_gap_events": summary.counter_gap_events,
            "inferred_missing_packets": summary.inferred_missing_packets,
            "validation_zero_packets": summary.validation_zero_packets,
            "source_kind": summary.source_kind,
            "source_label": summary.source_label,
        },
    )
