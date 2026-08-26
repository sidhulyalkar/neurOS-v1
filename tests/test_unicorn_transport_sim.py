from __future__ import annotations

import numpy as np
import pytest

from neuros.drivers.unicorn_network_sim import (
    RAW_UDP_PAYLOAD_BYTES,
    decode_unicorn_bandpower_ascii,
    decode_unicorn_udp_scan,
)
from neuros.drivers.unicorn_transport_sim import (
    DeterministicPacketFaultEngine,
    UnicornBandpowerUdpStreamSimulator,
    UnicornRawUdpStreamSimulator,
    UnicornUdpFaultProfile,
    get_unicorn_udp_fault_profile,
)


def _counter(datagram) -> int:
    # Raw UDP wire order is BAT, CNT, VALID at indices 14..16.
    return int(round(float(decode_unicorn_udp_scan(datagram.payload)[15])))


def test_pristine_raw_stream_preserves_68_byte_packets_and_counter_continuity():
    stream = UnicornRawUdpStreamSimulator(seed=3)
    packets = [stream.next_datagrams() for _ in range(4)]
    assert all(len(item) == 1 for item in packets)
    flattened = [item[0] for item in packets]
    assert all(len(packet.payload) == RAW_UDP_PAYLOAD_BYTES for packet in flattened)
    assert [_counter(packet) for packet in flattened] == [0, 1, 2, 3]
    assert [packet.source_sequence for packet in flattened] == [0, 1, 2, 3]


def test_transport_drop_does_not_rewind_device_counter():
    profile = UnicornUdpFaultProfile(name="drop-third", drop_every=3)
    stream = UnicornRawUdpStreamSimulator(seed=5, fault_profile=profile)
    first = stream.next_datagrams()[0]
    second = stream.next_datagrams()[0]
    dropped = stream.next_datagrams()
    fourth = stream.next_datagrams()[0]
    assert dropped == ()
    assert [_counter(first), _counter(second), _counter(fourth)] == [0, 1, 3]
    assert fourth.source_sequence == 3


def test_duplicate_probe_emits_same_payload_with_explicit_duplicate_ordinal():
    profile = UnicornUdpFaultProfile(name="dup-second", duplicate_every=2)
    stream = UnicornRawUdpStreamSimulator(seed=7, fault_profile=profile)
    assert len(stream.next_datagrams()) == 1
    pair = stream.next_datagrams()
    assert len(pair) == 2
    assert pair[0].payload == pair[1].payload
    assert pair[0].source_sequence == pair[1].source_sequence == 1
    assert pair[0].duplicate_ordinal == 0
    assert pair[1].duplicate_ordinal == 1
    assert "duplicate" in pair[1].faults


def test_reorder_probe_releases_newer_packet_before_held_packet():
    profile = UnicornUdpFaultProfile(name="reorder-third", reorder_every=3)
    stream = UnicornRawUdpStreamSimulator(seed=11, fault_profile=profile)
    assert stream.next_datagrams()[0].source_sequence == 0
    assert stream.next_datagrams()[0].source_sequence == 1
    assert stream.next_datagrams() == ()  # sequence 2 held
    reordered = stream.next_datagrams()   # sequence 3 released before sequence 2
    assert [packet.source_sequence for packet in reordered] == [3, 2]
    assert [_counter(packet) for packet in reordered] == [3, 2]
    assert "reordered_before_previous" in reordered[0].faults
    assert "reordered_after_next" in reordered[1].faults


def test_delay_probe_changes_release_time_without_changing_source_time():
    profile = UnicornUdpFaultProfile(name="delay-second", delay_every=2, delay_ms=12.0)
    engine = DeterministicPacketFaultEngine(profile, nominal_interval_s=0.004)
    first = engine.process(b"a", source_sequence=0, nominal_time_s=0.0)[0]
    second = engine.process(b"b", source_sequence=1, nominal_time_s=0.004)[0]
    assert first.release_time_s == first.nominal_time_s
    assert second.nominal_time_s == pytest.approx(0.004)
    assert second.release_time_s == pytest.approx(0.016)
    assert "delay" in second.faults


def test_named_fault_profiles_are_explicit_synthetic_assumptions():
    torture = get_unicorn_udp_fault_profile("mixed-torture")
    assert torture.evidence_class == "synthetic_assumption"
    assert torture.drop_every > 0
    assert torture.duplicate_every > 0
    assert torture.delay_every > 0
    assert torture.reorder_every > 0
    with pytest.raises(KeyError):
        get_unicorn_udp_fault_profile("real-bluetooth-statistics")


def test_bandpower_udp_stream_emits_one_70_value_frame_per_25hz_update():
    stream = UnicornBandpowerUdpStreamSimulator(seed=13)
    first = stream.next_datagrams()
    second = stream.next_datagrams()
    assert len(first) == len(second) == 1
    assert first[0].nominal_time_s == pytest.approx(0.0)
    assert second[0].nominal_time_s == pytest.approx(1.0 / 25.0)
    values = decode_unicorn_bandpower_ascii(first[0].payload)
    assert values.shape == (70,)
    assert np.all(np.isfinite(values))


def test_fault_profile_validation_rejects_ambiguous_delay_and_impossible_reorder():
    with pytest.raises(ValueError):
        UnicornUdpFaultProfile(name="bad-delay", delay_ms=5.0).validate()
    with pytest.raises(ValueError):
        UnicornUdpFaultProfile(name="bad-delay", delay_every=2, delay_ms=0.0).validate()
    with pytest.raises(ValueError):
        UnicornUdpFaultProfile(name="bad-reorder", reorder_every=1).validate()
