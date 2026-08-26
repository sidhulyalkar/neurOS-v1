"""Privacy-safe raw-UDP trace diagnostics for Unicorn simulator calibration.

The analyzer intentionally stores *no EEG sample arrays*. It reduces a sequence
of received 68-byte datagrams to interface/timing diagnostics that hardware owners
can share when validating the simulator: packet shape, arrival cadence, counter
gaps, duplicates, reordering, validation state, and battery range.

Counter-loss inference is conservative. A forward gap creates an unresolved
missing interval; a late packet inside that interval reconciles the loss instead
of being counted as permanently missing. If a backward counter cannot be
explained by a previously observed gap, the counter epoch is marked ambiguous
because the public interface does not document reset/wrap semantics.

A source label is user-declared provenance. neurOS cannot cryptographically prove
that a trace came from physical hardware merely because the caller says so.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Literal

import numpy as np

from .unicorn_network_sim import RAW_UDP_PAYLOAD_BYTES, decode_unicorn_udp_scan
from .unicorn_receiver_guard import FLOAT32_EXACT_INTEGER_MAX

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
    inferred_missing_packets: int | None
    duplicate_packets: int
    out_of_order_packets: int
    validation_zero_packets: int
    battery_min: float | None
    battery_max: float | None
    recovered_reordered_packets: int = 0
    counter_precision_ambiguous: bool = False
    counter_epoch_ambiguous: bool = False
    source_kind: TraceSourceKind = "unknown"
    source_label: str = ""

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.unicorn_raw_udp_trace_summary.v2",
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
            "recovered_reordered_packets": self.recovered_reordered_packets,
            "counter_precision_ambiguous": self.counter_precision_ambiguous,
            "counter_epoch_ambiguous": self.counter_epoch_ambiguous,
            "validation_zero_packets": self.validation_zero_packets,
            "battery_min": self.battery_min,
            "battery_max": self.battery_max,
            "source_kind": self.source_kind,
            "source_label": self.source_label,
            "contains_raw_eeg": False,
            "raw_packets_persisted": False,
            "evidence_boundary": (
                "Wire/timing diagnostics only. Counter-loss inference assumes one observable "
                "counter epoch unless marked ambiguous. Source provenance is user-declared; "
                "this summary is not a hardware certification or a human-EEG record."
            ),
        }


@dataclass(frozen=True)
class UnicornTraceContractComparison:
    """Diagnostic comparison against the nominal public raw-UDP contract."""

    packet_shape_clean: bool
    arrival_rate_near_250hz: bool | None
    counter_monotonic_without_duplicates: bool
    counter_gap_free: bool | None
    counter_precision_unambiguous: bool
    counter_epoch_unambiguous: bool
    observations: dict[str, object]

    @property
    def passed_diagnostic_checks(self) -> bool:
        # Unknown cadence or gap state is not evidence of passing. This is a
        # diagnostic qualification result, so every required check must be known
        # and true rather than treating missing evidence as success.
        return (
            self.packet_shape_clean
            and self.arrival_rate_near_250hz is True
            and self.counter_monotonic_without_duplicates
            and self.counter_gap_free is True
            and self.counter_precision_unambiguous
            and self.counter_epoch_unambiguous
        )

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.unicorn_raw_udp_trace_contract_comparison.v2",
            "passed_diagnostic_checks": self.passed_diagnostic_checks,
            "checks": {
                "packet_shape_clean": self.packet_shape_clean,
                "arrival_rate_near_250hz": self.arrival_rate_near_250hz,
                "counter_monotonic_without_duplicates": self.counter_monotonic_without_duplicates,
                "counter_gap_free": self.counter_gap_free,
                "counter_precision_unambiguous": self.counter_precision_unambiguous,
                "counter_epoch_unambiguous": self.counter_epoch_unambiguous,
            },
            "observations": dict(self.observations),
            "evidence_boundary": (
                "Diagnostic comparison to nominal public interface behavior only; "
                "not certification of physical hardware, Bluetooth, or EEG quality."
            ),
        }


@dataclass(frozen=True)
class UnicornTraceDeltaReport:
    """Descriptive differences between two trace summaries, with no pass/fail claim."""

    reference_source_kind: TraceSourceKind
    reference_source_label: str
    candidate_source_kind: TraceSourceKind
    candidate_source_label: str
    metrics: dict[str, float | None]

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.unicorn_raw_udp_trace_delta.v1",
            "reference": {
                "source_kind": self.reference_source_kind,
                "source_label": self.reference_source_label,
            },
            "candidate": {
                "source_kind": self.candidate_source_kind,
                "source_label": self.candidate_source_label,
            },
            "metrics": dict(self.metrics),
            "passed": None,
            "evidence_boundary": (
                "Descriptive trace deltas only. No default tolerance is promoted to a "
                "manufacturer specification or physical-hardware equivalence claim."
            ),
        }


def _consume_missing_counter(intervals: list[tuple[int, int]], counter: int) -> bool:
    """Remove one late counter from sorted disjoint inclusive missing intervals."""

    for index, (start, end) in enumerate(intervals):
        if counter < start:
            return False
        if counter > end:
            continue
        if start == end:
            intervals.pop(index)
        elif counter == start:
            intervals[index] = (start + 1, end)
        elif counter == end:
            intervals[index] = (start, end - 1)
        else:
            intervals[index] = (start, counter - 1)
            intervals.insert(index + 1, (counter + 1, end))
        return True
    return False


def _missing_count(intervals: list[tuple[int, int]]) -> int:
    return sum(end - start + 1 for start, end in intervals)


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
    high_water: int | None = None
    gap_events = 0
    duplicates = 0
    out_of_order = 0
    recovered_reordered = 0
    validation_zero = 0
    precision_ambiguous = False
    epoch_ambiguous = False
    batteries: list[float] = []
    seen_counters: set[int] = set()
    unresolved_missing: list[tuple[int, int]] = []

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

        if abs(counter_float) > FLOAT32_EXACT_INTEGER_MAX:
            precision_ambiguous = True
            continue

        if counter in seen_counters:
            duplicates += 1
            continue
        seen_counters.add(counter)

        if high_water is None:
            high_water = counter
            continue
        if counter > high_water:
            if counter > high_water + 1:
                gap_events += 1
                unresolved_missing.append((high_water + 1, counter - 1))
            high_water = counter
            continue

        # A lower unseen counter is either a late packet that repairs a known
        # gap, or an unexplained counter-epoch discontinuity. We can distinguish
        # the first case without inventing device reset/wrap semantics.
        out_of_order += 1
        if _consume_missing_counter(unresolved_missing, counter):
            recovered_reordered += 1
        else:
            epoch_ambiguous = True

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

    missing: int | None
    if precision_ambiguous or epoch_ambiguous:
        missing = None
    else:
        missing = _missing_count(unresolved_missing)

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
        recovered_reordered_packets=recovered_reordered,
        counter_precision_ambiguous=precision_ambiguous,
        counter_epoch_ambiguous=epoch_ambiguous,
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
    gap_free = (
        None
        if summary.inferred_missing_packets is None
        else summary.inferred_missing_packets == 0
    )
    return UnicornTraceContractComparison(
        packet_shape_clean=summary.malformed_packet_count == 0,
        arrival_rate_near_250hz=rate_ok,
        counter_monotonic_without_duplicates=counter_clean,
        counter_gap_free=gap_free,
        counter_precision_unambiguous=not summary.counter_precision_ambiguous,
        counter_epoch_unambiguous=not summary.counter_epoch_ambiguous,
        observations={
            "estimated_arrival_rate_hz": rate,
            "rate_tolerance_hz": float(rate_tolerance_hz),
            "counter_gap_events": summary.counter_gap_events,
            "inferred_missing_packets": summary.inferred_missing_packets,
            "recovered_reordered_packets": summary.recovered_reordered_packets,
            "validation_zero_packets": summary.validation_zero_packets,
            "source_kind": summary.source_kind,
            "source_label": summary.source_label,
        },
    )


def _safe_fraction(numerator: int, denominator: int) -> float | None:
    return None if denominator <= 0 else float(numerator) / float(denominator)


def _missing_fraction(summary: UnicornRawUdpTraceSummary) -> float | None:
    """Fraction of unique packet opportunities that remain unresolved missing."""

    if summary.inferred_missing_packets is None:
        return None
    unique_decoded = max(0, summary.decoded_packet_count - summary.duplicate_packets)
    return _safe_fraction(
        summary.inferred_missing_packets,
        unique_decoded + summary.inferred_missing_packets,
    )


def _delta(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None:
        return None
    return float(candidate - reference)


def compare_unicorn_trace_summaries(
    reference: UnicornRawUdpTraceSummary,
    candidate: UnicornRawUdpTraceSummary,
) -> UnicornTraceDeltaReport:
    """Describe how two traces differ without declaring either one ground truth.

    This is the calibration bridge between the synthetic endpoint and a short
    user-declared physical trace. It deliberately has no default pass threshold:
    a measured discrepancy is evidence to inspect, not proof of equivalence or a
    hardware defect.
    """

    reference_missing_fraction = _missing_fraction(reference)
    candidate_missing_fraction = _missing_fraction(candidate)
    reference_malformed = _safe_fraction(
        reference.malformed_packet_count,
        reference.packet_count,
    )
    candidate_malformed = _safe_fraction(
        candidate.malformed_packet_count,
        candidate.packet_count,
    )
    reference_invalid = _safe_fraction(
        reference.validation_zero_packets,
        reference.decoded_packet_count,
    )
    candidate_invalid = _safe_fraction(
        candidate.validation_zero_packets,
        candidate.decoded_packet_count,
    )
    reference_reorder = _safe_fraction(
        reference.out_of_order_packets,
        reference.decoded_packet_count,
    )
    candidate_reorder = _safe_fraction(
        candidate.out_of_order_packets,
        candidate.decoded_packet_count,
    )

    return UnicornTraceDeltaReport(
        reference_source_kind=reference.source_kind,
        reference_source_label=reference.source_label,
        candidate_source_kind=candidate.source_kind,
        candidate_source_label=candidate.source_label,
        metrics={
            "arrival_rate_delta_hz": _delta(
                candidate.estimated_arrival_rate_hz,
                reference.estimated_arrival_rate_hz,
            ),
            "mean_interarrival_delta_ms": _delta(
                candidate.mean_interarrival_ms,
                reference.mean_interarrival_ms,
            ),
            "p95_interarrival_delta_ms": _delta(
                candidate.p95_interarrival_ms,
                reference.p95_interarrival_ms,
            ),
            "malformed_fraction_delta": _delta(
                candidate_malformed,
                reference_malformed,
            ),
            "validation_zero_fraction_delta": _delta(
                candidate_invalid,
                reference_invalid,
            ),
            "reorder_fraction_delta": _delta(
                candidate_reorder,
                reference_reorder,
            ),
            "unresolved_missing_fraction_delta": _delta(
                candidate_missing_fraction,
                reference_missing_fraction,
            ),
        },
    )