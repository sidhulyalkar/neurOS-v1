from __future__ import annotations

import struct

import pytest

from neuros.drivers.unicorn_network_sim import decode_unicorn_udp_scan
from neuros.drivers.unicorn_receiver_guard import UnicornRawUdpGuard, UnicornRawUdpGuardConfig
from neuros.drivers.unicorn_transport_sim import UnicornRawUdpStreamSimulator, UnicornUdpFaultProfile


def _packet(stream: UnicornRawUdpStreamSimulator) -> bytes:
    datagrams = stream.next_datagrams()
    assert len(datagrams) == 1
    return datagrams[0].payload


def _set_validation(payload: bytes, value: float) -> bytes:
    values = decode_unicorn_udp_scan(payload).astype(float)
    values[16] = value
    return struct.pack("<17f", *values.tolist())


def test_authority_requires_consecutive_valid_sequential_packets():
    stream = UnicornRawUdpStreamSimulator(seed=101)
    guard = UnicornRawUdpGuard(UnicornRawUdpGuardConfig(recovery_packets=3))
    first = guard.ingest(_packet(stream), received_monotonic_s=0.000)
    second = guard.ingest(_packet(stream), received_monotonic_s=0.004)
    third = guard.ingest(_packet(stream), received_monotonic_s=0.008)
    assert [first.authority_allowed, second.authority_allowed, third.authority_allowed] == [False, False, True]
    assert guard.state(now_monotonic_s=0.009).authority_allowed is True


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
    assert invalid.counter == 2
    assert invalid.authority_allowed is False
    # Counter 3 follows the observed invalid packet sequentially; liveness is
    # continuous but authority still has to rebuild from a fresh healthy streak.
    recovered_one = guard.ingest(_packet(stream), received_monotonic_s=0.012)
    recovered_two = guard.ingest(_packet(stream), received_monotonic_s=0.016)
    assert recovered_one.health == recovered_two.health == "healthy"
    assert recovered_one.authority_allowed is False
    assert recovered_two.authority_allowed is True


def test_duplicate_and_reordered_packets_fail_closed():
    duplicate_stream = UnicornRawUdpStreamSimulator(
        seed=109,
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
    assert older.health == "out_of_order"
    assert newer.authority_allowed is older.authority_allowed is False


def test_stale_interval_revokes_authority_and_fresh_packet_cannot_inherit_old_streak():
    stream = UnicornRawUdpStreamSimulator(seed=127)
    guard = UnicornRawUdpGuard(
        UnicornRawUdpGuardConfig(stale_after_s=0.050, recovery_packets=2)
    )
    guard.ingest(_packet(stream), received_monotonic_s=0.000)
    assert guard.ingest(_packet(stream), received_monotonic_s=0.004).authority_allowed is True
    stale = guard.state(now_monotonic_s=0.100)
    assert stale.health == "stale"
    assert stale.authority_allowed is False
    fresh_one = guard.ingest(_packet(stream), received_monotonic_s=0.104)
    fresh_two = guard.ingest(_packet(stream), received_monotonic_s=0.108)
    assert fresh_one.authority_allowed is False
    assert fresh_two.authority_allowed is True


def test_malformed_packet_never_updates_authority():
    guard = UnicornRawUdpGuard()
    malformed = guard.ingest(b"too short", received_monotonic_s=1.0)
    assert malformed.health == "malformed"
    assert malformed.authority_allowed is False
    assert guard.state(now_monotonic_s=1.01).health == "stale"


def test_guard_config_rejects_invalid_policies():
    with pytest.raises(ValueError):
        UnicornRawUdpGuardConfig(stale_after_s=0.0).validate()
    with pytest.raises(ValueError):
        UnicornRawUdpGuardConfig(recovery_packets=0).validate()
