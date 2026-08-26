from __future__ import annotations

import struct

import pytest

from neuros.drivers.unicorn_network_sim import decode_unicorn_udp_scan
from neuros.drivers.unicorn_receiver_guard import FLOAT32_EXACT_INTEGER_MAX
from neuros.drivers.unicorn_trace import (
    analyze_unicorn_raw_udp_trace,
    compare_unicorn_trace_summaries,
    compare_unicorn_trace_to_nominal_contract,
)
from neuros.drivers.unicorn_transport_sim import UnicornRawUdpStreamSimulator, UnicornUdpFaultProfile


def _collect(stream: UnicornRawUdpStreamSimulator, updates: int):
    records = []
    for _ in range(updates):
        for datagram in stream.next_datagrams():
            records.append((datagram.release_time_s, datagram.payload))
    for datagram in stream.flush():
        records.append((datagram.release_time_s, datagram.payload))
    records.sort(key=lambda item: item[0])
    return records


def _set_counter(payload: bytes, counter: float) -> bytes:
    values = decode_unicorn_udp_scan(payload).astype(float)
    values[15] = counter
    return struct.pack("<17f", *values.tolist())


def test_pristine_trace_reduces_to_shareable_metadata_without_raw_eeg():
    stream = UnicornRawUdpStreamSimulator(seed=151)
    summary = analyze_unicorn_raw_udp_trace(
        _collect(stream, 251),
        source_kind="synthetic",
        source_label="ci-pristine",
    )
    payload = summary.to_dict()
    assert payload["schema"] == "neuros.unicorn_raw_udp_trace_summary.v2"
    assert payload["packet_count"] == 251
    assert payload["decoded_packet_count"] == 251
    assert payload["malformed_packet_count"] == 0
    assert payload["estimated_arrival_rate_hz"] == pytest.approx(250.0)
    assert payload["mean_interarrival_ms"] == pytest.approx(4.0)
    assert payload["first_counter"] == 0
    assert payload["last_counter"] == 250
    assert payload["counter_gap_events"] == 0
    assert payload["inferred_missing_packets"] == 0
    assert payload["duplicate_packets"] == 0
    assert payload["out_of_order_packets"] == 0
    assert payload["counter_precision_ambiguous"] is False
    assert payload["counter_epoch_ambiguous"] is False
    assert payload["contains_raw_eeg"] is False
    assert payload["raw_packets_persisted"] is False

    comparison = compare_unicorn_trace_to_nominal_contract(summary)
    assert comparison.counter_gap_free is True
    assert comparison.passed_diagnostic_checks is True


def test_periodic_loss_is_visible_as_unresolved_counter_gaps():
    stream = UnicornRawUdpStreamSimulator(
        seed=157,
        fault_profile=UnicornUdpFaultProfile(name="drop-10", drop_every=10),
    )
    summary = analyze_unicorn_raw_udp_trace(_collect(stream, 50), source_kind="synthetic")
    assert summary.packet_count == 45
    assert summary.malformed_packet_count == 0
    assert summary.counter_gap_events == 4
    assert summary.inferred_missing_packets == 4
    assert summary.duplicate_packets == 0
    assert summary.out_of_order_packets == 0
    assert summary.estimated_arrival_rate_hz is not None
    assert summary.estimated_arrival_rate_hz < 250.0
    comparison = compare_unicorn_trace_to_nominal_contract(summary)
    assert comparison.counter_gap_free is False
    assert comparison.passed_diagnostic_checks is False


def test_reordered_packet_repairs_transient_gap_instead_of_becoming_false_loss():
    reorder_stream = UnicornRawUdpStreamSimulator(
        seed=167,
        fault_profile=UnicornUdpFaultProfile(name="reorder-five", reorder_every=5),
    )
    summary = analyze_unicorn_raw_udp_trace(_collect(reorder_stream, 20))
    assert summary.out_of_order_packets > 0
    assert summary.counter_gap_events > 0
    assert summary.recovered_reordered_packets == summary.out_of_order_packets
    assert summary.inferred_missing_packets == 0
    assert summary.counter_epoch_ambiguous is False
    # The strict nominal comparison still reports reordering rather than hiding it.
    assert compare_unicorn_trace_to_nominal_contract(summary).passed_diagnostic_checks is False


def test_duplicate_probe_is_distinguished_from_reordering():
    duplicate_stream = UnicornRawUdpStreamSimulator(
        seed=163,
        fault_profile=UnicornUdpFaultProfile(name="dup-five", duplicate_every=5),
    )
    summary = analyze_unicorn_raw_udp_trace(_collect(duplicate_stream, 20))
    assert summary.duplicate_packets == 4
    assert summary.out_of_order_packets == 0
    assert summary.inferred_missing_packets == 0


def test_unexplained_backward_counter_marks_epoch_ambiguous_instead_of_inventing_loss():
    stream = UnicornRawUdpStreamSimulator(seed=169)
    packets = [_collect(stream, 1)[0][1] for _ in range(3)]
    records = [
        (0.000, _set_counter(packets[0], 100.0)),
        (0.004, _set_counter(packets[1], 101.0)),
        (0.008, _set_counter(packets[2], 0.0)),
    ]
    summary = analyze_unicorn_raw_udp_trace(records)
    assert summary.out_of_order_packets == 1
    assert summary.counter_epoch_ambiguous is True
    assert summary.inferred_missing_packets is None
    comparison = compare_unicorn_trace_to_nominal_contract(summary)
    assert comparison.counter_epoch_unambiguous is False
    assert comparison.counter_gap_free is None
    assert comparison.passed_diagnostic_checks is False


def test_counter_precision_boundary_suppresses_exact_loss_claims():
    stream = UnicornRawUdpStreamSimulator(seed=171)
    packet = _collect(stream, 1)[0][1]
    summary = analyze_unicorn_raw_udp_trace([
        (0.0, _set_counter(packet, float(FLOAT32_EXACT_INTEGER_MAX + 2)))
    ])
    assert summary.counter_precision_ambiguous is True
    assert summary.inferred_missing_packets is None
    comparison = compare_unicorn_trace_to_nominal_contract(summary)
    assert comparison.counter_precision_unambiguous is False
    assert comparison.passed_diagnostic_checks is False


def test_trace_delta_report_is_descriptive_and_has_no_default_equivalence_threshold():
    reference_stream = UnicornRawUdpStreamSimulator(seed=181)
    candidate_stream = UnicornRawUdpStreamSimulator(seed=181)
    reference = analyze_unicorn_raw_udp_trace(
        _collect(reference_stream, 251), source_kind="synthetic", source_label="reference"
    )
    candidate = analyze_unicorn_raw_udp_trace(
        _collect(candidate_stream, 251),
        source_kind="user_declared_physical",
        source_label="candidate",
    )
    report = compare_unicorn_trace_summaries(reference, candidate).to_dict()
    assert report["passed"] is None
    assert report["metrics"]["arrival_rate_delta_hz"] == pytest.approx(0.0)
    assert report["metrics"]["mean_interarrival_delta_ms"] == pytest.approx(0.0)
    assert report["reference"]["source_kind"] == "synthetic"
    assert report["candidate"]["source_kind"] == "user_declared_physical"
    assert "No default tolerance" in report["evidence_boundary"]


def test_malformed_payload_is_counted_but_never_persisted():
    stream = UnicornRawUdpStreamSimulator(seed=173)
    records = _collect(stream, 3)
    records.insert(1, (0.002, b"bad"))
    records.sort(key=lambda item: item[0])
    summary = analyze_unicorn_raw_udp_trace(records, source_kind="unknown")
    assert summary.packet_count == 4
    assert summary.decoded_packet_count == 3
    assert summary.malformed_packet_count == 1
    assert summary.to_dict()["contains_raw_eeg"] is False
    assert summary.to_dict()["raw_packets_persisted"] is False


def test_user_declared_physical_is_explicitly_not_cryptographic_provenance():
    summary = analyze_unicorn_raw_udp_trace(
        [],
        source_kind="user_declared_physical",
        source_label="lab-headset-a",
    )
    payload = summary.to_dict()
    assert payload["source_kind"] == "user_declared_physical"
    assert payload["source_label"] == "lab-headset-a"
    assert "user-declared" in payload["evidence_boundary"]
    assert compare_unicorn_trace_to_nominal_contract(summary).passed_diagnostic_checks is False


def test_trace_requires_monotonic_receive_timestamps():
    with pytest.raises(ValueError):
        analyze_unicorn_raw_udp_trace([(1.0, b"x"), (0.5, b"y")])
