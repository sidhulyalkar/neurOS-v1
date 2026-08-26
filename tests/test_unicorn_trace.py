from __future__ import annotations

import pytest

from neuros.drivers.unicorn_trace import (
    analyze_unicorn_raw_udp_trace,
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


def test_pristine_trace_reduces_to_shareable_metadata_without_raw_eeg():
    stream = UnicornRawUdpStreamSimulator(seed=151)
    summary = analyze_unicorn_raw_udp_trace(
        _collect(stream, 251),
        source_kind="synthetic",
        source_label="ci-pristine",
    )
    payload = summary.to_dict()
    assert payload["packet_count"] == 251
    assert payload["decoded_packet_count"] == 251
    assert payload["malformed_packet_count"] == 0
    assert payload["estimated_arrival_rate_hz"] == pytest.approx(250.0)
    assert payload["mean_interarrival_ms"] == pytest.approx(4.0)
    assert payload["first_counter"] == 0
    assert payload["last_counter"] == 250
    assert payload["counter_gap_events"] == 0
    assert payload["duplicate_packets"] == 0
    assert payload["out_of_order_packets"] == 0
    assert payload["contains_raw_eeg"] is False
    assert "eeg" not in payload or payload["contains_raw_eeg"] is False

    comparison = compare_unicorn_trace_to_nominal_contract(summary)
    assert comparison.passed_diagnostic_checks is True


def test_periodic_loss_is_visible_as_counter_gaps_without_malformed_packets():
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
    # The arrival rate is lower because the test records transport delivery
    # times rather than pretending dropped packets arrived.
    assert summary.estimated_arrival_rate_hz is not None
    assert summary.estimated_arrival_rate_hz < 250.0


def test_duplicate_and_reorder_probe_are_distinguished():
    duplicate_stream = UnicornRawUdpStreamSimulator(
        seed=163,
        fault_profile=UnicornUdpFaultProfile(name="dup-five", duplicate_every=5),
    )
    duplicate_summary = analyze_unicorn_raw_udp_trace(_collect(duplicate_stream, 20))
    assert duplicate_summary.duplicate_packets == 4
    assert duplicate_summary.out_of_order_packets == 0

    reorder_stream = UnicornRawUdpStreamSimulator(
        seed=167,
        fault_profile=UnicornUdpFaultProfile(name="reorder-five", reorder_every=5),
    )
    reorder_summary = analyze_unicorn_raw_udp_trace(_collect(reorder_stream, 20))
    assert reorder_summary.out_of_order_packets > 0
    assert reorder_summary.counter_gap_events > 0


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


def test_trace_requires_monotonic_receive_timestamps():
    with pytest.raises(ValueError):
        analyze_unicorn_raw_udp_trace([(1.0, b"x"), (0.5, b"y")])
