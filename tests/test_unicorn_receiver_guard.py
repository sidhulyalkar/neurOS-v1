from __future__ import annotations

import struct

import pytest

from neuros.drivers.unicorn_network_sim import decode_unicorn_udp_scan
from neuros.drivers.unicorn_receiver_guard import (
    FLOAT32_EXACT_INTEGER_MAX,
    UnicornRawUdpGuard,
    UnicornRawUdpGuardConfig,
)
from neuros.drivers.unicorn_transport_sim import UnicornRawUdpStreamSimulator, UnicornUdpFaultProfile


def _packet(stream: UnicornRawUdpStreamSimulator) -> bytes:
    datagrams = stream.next_datagrams()
    assert len(datagrams) == 1
    return datagrams[0].payload


def _replace_field(payload: bytes, index: int, value: float) -> bytes:
    values = decode_unicorn_udp_scan(payload).astype(float)
    values[index] = value
    return struct.pack("<17f", *values.tolist())


def _set_validation(payload: bytes, value: float) -> bytes:
    return _replace_field(payload, 16, value)


def _set_counter(payload: bytes, value: float) -> bytes:
    return _replace_field(payload, 15, value)


def test_authority_requires_consecutive_valid_sequential_packets():
    stream = UnicornRawUdpStreamSimulator(seed=101)
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=3))
    first = guard.ingest(_packet(stream), received_monotonic_s=0.000)
    second = guard.ingest(_packet(stream), received_monotonic_s=0.004)
    third = guard.ingest(_packet(stream), received_monotonic_s=0.008)
    assert [first.sequence_status, second.sequence_status, third.sequence_status] == [
        "first", "sequential", "sequential"
    ]
    assert [first.authority_allowed, second.authority_allowed, third.authority_allowed] == [False, False, True]
    state = guard.state(now_monotonic_s=0.009)
    assert state.authority_allowed is True
    assert state.stream_live is True


def test_counter_gap_revokes_authority_and_requires_recovery_streak():
    profile = UnicornUdpFaultProfile(name="drop-fourth", drop_every=4)
    stream = UnicornRawUdpStreamSimulator(seed=103, fault_profile=profile)
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=2))
    assert guard.ingest(stream.next_datagrams()[0].payload, received_monotonic_s=0.000).authority_allowed is False
    assert guard.ingest(stream.next_datagrams()[0].payload, received_monotonic_s=0.004).authority_allowed is True
    guard.ingest(stream.next_datagrams()[0].payload, received_monotonic_s=0.008)
    assert stream.next_datagrams() == ()  # counter 3 lost in transport
    gap = guard.ingest(stream.next_datagrams()[0].payload, received_monotonic_s=0.016)
    assert gap.health == "gap"
    assert gap.sequence_status == "gap"
    assert gap.missed_packets == 1
    assert gap.authority_allowed is False
    assert guard.ingest(stream.next_datagrams()[0].payload, received_monotonic_s=0.020).authority_allowed is False
    assert guard.ingest(stream.next_datagrams()[0].payload, received_monotonic_s=0.024).authority_allowed is True


def test_validation_zero_preserves_liveness_and_counter_but_revokes_authority():
    stream = UnicornRawUdpStreamSimulator(seed=107)
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=2))
    guard.ingest(_packet(stream), received_monotonic_s=0.000)
    assert guard.ingest(_packet(stream), received_monotonic_s=0.004).authority_allowed is True
    invalid_payload = _set_validation(_packet(stream), 0.0)
    invalid = guard.ingest(invalid_payload, received_monotonic_s=0.008)
    assert invalid.health == "invalid"
    assert invalid.packet_status == "decodable"
    assert invalid.sequence_status == "sequential"
    assert invalid.validation_asserted is False
    assert invalid.counter == 2
    assert invalid.authority_allowed is False
    assert guard.state(now_monotonic_s=0.009).stream_live is True
    recovered_one = guard.ingest(_packet(stream), received_monotonic_s=0.012)
    recovered_two = guard.ingest(_packet(stream), received_monotonic_s=0.016)
    assert recovered_one.health == recovered_two.health == "healthy"
    assert recovered_one.authority_allowed is False
    assert recovered_two.authority_allowed is True


def test_invalid_packet_cannot_hide_a_simultaneous_counter_gap():
    stream = UnicornRawUdpStreamSimulator(seed=108)
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=2))
    guard.ingest(_packet(stream), received_monotonic_s=0.000)  # counter 0
    guard.ingest(_packet(stream), received_monotonic_s=0.004)  # counter 1
    _packet(stream)  # counter 2 never reaches the receiver
    invalid_gap = guard.ingest(
        _set_validation(_packet(stream), 0.0),  # counter 3
        received_monotonic_s=0.012,
    )
    assert invalid_gap.health == "invalid"  # compact backward-compatible summary
    assert invalid_gap.validation_asserted is False
    assert invalid_gap.sequence_status == "gap"
    assert invalid_gap.missed_packets == 1
    assert guard.last_counter == 3
    first_recovery = guard.ingest(_packet(stream), received_monotonic_s=0.016)
    second_recovery = guard.ingest(_packet(stream), received_monotonic_s=0.020)
    assert first_recovery.sequence_status == second_recovery.sequence_status == "sequential"
    assert first_recovery.authority_allowed is False
    assert second_recovery.authority_allowed is True


def test_late_invalid_packet_does_not_rewind_counter_high_water():
    stream = UnicornRawUdpStreamSimulator(seed=109)
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=1))
    first = _packet(stream)   # 0
    old = _packet(stream)     # 1
    latest = _packet(stream)  # 2
    guard.ingest(first, received_monotonic_s=0.000)
    guard.ingest(old, received_monotonic_s=0.004)
    guard.ingest(latest, received_monotonic_s=0.008)
    late = guard.ingest(_set_validation(old, 0.0), received_monotonic_s=0.009)
    assert late.health == "invalid"
    assert late.sequence_status == "out_of_order"
    assert guard.last_counter == 2
    next_packet = guard.ingest(_packet(stream), received_monotonic_s=0.012)
    assert next_packet.sequence_status == "sequential"
    assert next_packet.counter == 3


def test_duplicate_and_reordered_packets_fail_closed():
    duplicate_stream = UnicornRawUdpStreamSimulator(
        seed=110,
        fault_profile=UnicornUdpFaultProfile(name="dup-third", duplicate_every=3),
    )
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=1))
    guard.ingest(duplicate_stream.next_datagrams()[0].payload, received_monotonic_s=0.000)
    guard.ingest(duplicate_stream.next_datagrams()[0].payload, received_monotonic_s=0.004)
    pair = duplicate_stream.next_datagrams()
    assert len(pair) == 2
    assert guard.ingest(pair[0].payload, received_monotonic_s=0.008).authority_allowed is True
    duplicate = guard.ingest(pair[1].payload, received_monotonic_s=0.0081)
    assert duplicate.health == "duplicate"
    assert duplicate.sequence_status == "duplicate"
    assert duplicate.authority_allowed is False

    reorder_stream = UnicornRawUdpStreamSimulator(
        seed=113,
        fault_profile=UnicornUdpFaultProfile(name="reorder-third", reorder_every=3),
    )
    reorder_guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=1))
    reorder_guard.ingest(reorder_stream.next_datagrams()[0].payload, received_monotonic_s=0.000)
    reorder_guard.ingest(reorder_stream.next_datagrams()[0].payload, received_monotonic_s=0.004)
    assert reorder_stream.next_datagrams() == ()
    pair = reorder_stream.next_datagrams()
    newer = reorder_guard.ingest(pair[0].payload, received_monotonic_s=0.012)
    older = reorder_guard.ingest(pair[1].payload, received_monotonic_s=0.013)
    assert newer.health == "gap"
    assert newer.sequence_status == "gap"
    assert older.health == "out_of_order"
    assert older.sequence_status == "out_of_order"
    assert newer.authority_allowed is older.authority_allowed is False
    assert reorder_guard.last_counter == 3


def test_counter_above_float32_unit_step_exactness_fails_closed():
    stream = UnicornRawUdpStreamSimulator(seed=119)
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=1))
    payload = _set_counter(_packet(stream), float(FLOAT32_EXACT_INTEGER_MAX + 2))
    observation = guard.ingest(payload, received_monotonic_s=0.0)
    assert observation.packet_status == "decodable"
    assert observation.sequence_status == "precision_ambiguous"
    assert observation.counter_step_exact is False
    assert observation.health == "counter_ambiguous"
    assert observation.authority_allowed is False


def test_counter_reset_requires_explicit_new_epoch_before_authority_can_recover():
    stream = UnicornRawUdpStreamSimulator(seed=121)
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=2))
    packet = _packet(stream)

    first = guard.ingest(_set_counter(packet, 100.0), received_monotonic_s=0.000)
    second = guard.ingest(_set_counter(packet, 101.0), received_monotonic_s=0.004)
    assert first.sequence_status == "first"
    assert second.authority_allowed is True

    reset_without_lifecycle = guard.ingest(
        _set_counter(packet, 0.0),
        received_monotonic_s=0.008,
    )
    assert reset_without_lifecycle.sequence_status == "out_of_order"
    assert reset_without_lifecycle.authority_allowed is False
    assert guard.last_counter == 101

    guard.begin_new_epoch()
    reset_state = guard.state(now_monotonic_s=0.009)
    assert reset_state.health == "stale"
    assert reset_state.stream_live is False
    assert reset_state.authority_allowed is False
    assert reset_state.last_counter is None

    new_first = guard.ingest(_set_counter(packet, 0.0), received_monotonic_s=0.012)
    new_second = guard.ingest(_set_counter(packet, 1.0), received_monotonic_s=0.016)
    assert new_first.sequence_status == "first"
    assert new_first.authority_allowed is False
    assert new_second.sequence_status == "sequential"
    assert new_second.authority_allowed is True


def test_stale_interval_revokes_authority_and_fresh_packet_cannot_inherit_old_streak():
    stream = UnicornRawUdpStreamSimulator(seed=127)
    guard = UnicornRawUdpGuard(
        UnicornRawUdpGuardConfig(stale_after_s=0.050, recovery_packets=2)
    )
    guard.ingest(_packet(stream), received_monotonic_s=0.000)
    assert guard.ingest(_packet(stream), received_monotonic_s=0.004).authority_allowed is True
    stale = guard.state(now_monotonic_s=0.100)
    assert stale.health == "stale"
    assert stale.stream_live is False
    assert stale.authority_allowed is False
    fresh_one = guard.ingest(_packet(stream), received_monotonic_s=0.104)
    fresh_two = guard.ingest(_packet(stream), received_monotonic_s=0.108)
    assert fresh_one.authority_allowed is False
    assert fresh_two.authority_allowed is True


def test_malformed_packet_never_refreshes_stream_liveness():
    guard = UnicornRawUdpGuard()
    malformed = guard.ingest(b"too short", received_monotonic_s=1.0)
    assert malformed.health == "malformed"
    assert malformed.packet_status == "malformed"
    assert malformed.authority_allowed is False
    state = guard.state(now_monotonic_s=1.01)
    assert state.health == "stale"
    assert state.stream_live is False


def test_guard_config_rejects_invalid_policies():
    with pytest.raises(ValueError):
        UnicornRawUdpGuardConfig(stale_after_s=0.0).validate()
    with pytest.raises(ValueError):
        UnicornRawUdpGuardConfig(recovery_packets=0).validate()
