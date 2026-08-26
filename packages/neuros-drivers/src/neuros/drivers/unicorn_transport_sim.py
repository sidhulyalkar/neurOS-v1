"""Deterministic real-time transport simulation for Unicorn-compatible UDP streams.

This module intentionally sits *after* the device twin. It does not alter EEG
physiology or manufacturer telemetry semantics. Instead it turns already-encoded
raw-UDP or Bandpower payloads into a reproducible delivery schedule with explicit
loss, duplication, delay, and reordering probes.

Two source-clock quantities are kept separate:

``initial_delay_s``
    Time from stream start until the first source update can exist. Raw EEG has
    no synthetic warm-up delay. Bandpower requires its complete analysis window.

``interval_s``
    Cadence between subsequent source updates.

The fault profiles are synthetic test policies. They are not claimed to describe
measured Unicorn Bluetooth or operating-system network statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from .unicorn_hybrid_black_sim import (
    UnicornHybridBlackSimulationConfig,
    UnicornHybridBlackSimulator,
)
from .unicorn_network_sim import (
    UnicornBandpowerReferenceStream,
    encode_unicorn_bandpower_ascii,
    encode_unicorn_udp_scan,
)

TransportEvidenceClass = Literal["synthetic_assumption"]


@dataclass(frozen=True)
class UnicornUdpFaultProfile:
    """Deterministic packet-fault policy for hardware-free robustness tests.

    Periodic fields use one-based human-readable cadence. For example,
    ``drop_every=10`` drops source packets 10, 20, 30, ... while source sequence
    numbers remain zero-based internally. A zero value disables that fault.

    ``reorder_every=N`` holds packet N, then releases packet N+1 before it. This
    gives a receiver a deterministic adjacent-packet inversion without invoking
    wall-clock randomness.
    """

    name: str = "pristine"
    drop_every: int = 0
    duplicate_every: int = 0
    delay_every: int = 0
    delay_ms: float = 0.0
    reorder_every: int = 0
    evidence_class: TransportEvidenceClass = "synthetic_assumption"
    description: str = "No synthetic transport faults."

    def validate(self) -> None:
        if not self.name.strip():
            raise ValueError("fault profile name must be non-empty")
        for field_name in ("drop_every", "duplicate_every", "delay_every", "reorder_every"):
            value = int(getattr(self, field_name))
            if value < 0:
                raise ValueError(f"{field_name} must be non-negative")
        if self.delay_ms < 0:
            raise ValueError("delay_ms must be non-negative")
        if self.delay_every == 0 and self.delay_ms != 0:
            raise ValueError("delay_ms requires delay_every > 0")
        if self.delay_every > 0 and self.delay_ms <= 0:
            raise ValueError("delay_every requires delay_ms > 0")
        if self.reorder_every == 1:
            raise ValueError("reorder_every=1 cannot form stable adjacent pairs")


FAULT_PROFILES: dict[str, UnicornUdpFaultProfile] = {
    "pristine": UnicornUdpFaultProfile(),
    "periodic-loss": UnicornUdpFaultProfile(
        name="periodic-loss",
        drop_every=50,
        description="Synthetic 2% periodic packet-loss probe; not a measured hardware loss rate.",
    ),
    "duplicate-probe": UnicornUdpFaultProfile(
        name="duplicate-probe",
        duplicate_every=40,
        description="Synthetic periodic duplicate-packet probe.",
    ),
    "reorder-probe": UnicornUdpFaultProfile(
        name="reorder-probe",
        reorder_every=40,
        description="Synthetic adjacent-packet reordering probe.",
    ),
    "delay-probe": UnicornUdpFaultProfile(
        name="delay-probe",
        delay_every=25,
        delay_ms=20.0,
        description="Synthetic periodic delivery-delay probe.",
    ),
    "mixed-torture": UnicornUdpFaultProfile(
        name="mixed-torture",
        drop_every=29,
        duplicate_every=41,
        delay_every=17,
        delay_ms=16.0,
        reorder_every=53,
        description=(
            "Synthetic deterministic mixed loss/duplicate/delay/reorder torture profile. "
            "Cadences are test choices, not manufacturer statistics."
        ),
    ),
}
for _profile in FAULT_PROFILES.values():
    _profile.validate()


def get_unicorn_udp_fault_profile(name: str) -> UnicornUdpFaultProfile:
    """Return a named immutable synthetic fault profile."""

    key = str(name).strip().lower()
    try:
        return FAULT_PROFILES[key]
    except KeyError as exc:
        raise KeyError(f"unknown Unicorn UDP fault profile {name!r}; choose from {sorted(FAULT_PROFILES)}") from exc


@dataclass(frozen=True)
class ScheduledDatagram:
    """One synthetic transport delivery decision.

    ``nominal_time_s`` is the source cadence time. ``release_time_s`` is the
    earliest time a real-time sender should emit the datagram. Reordered packets
    can have an older nominal time while being released after a newer packet.
    """

    payload: bytes
    source_sequence: int
    nominal_time_s: float
    release_time_s: float
    duplicate_ordinal: int = 0
    faults: tuple[str, ...] = ()


class DeterministicPacketFaultEngine:
    """Stateful packet transformer with reproducible loss/reorder semantics."""

    def __init__(self, profile: UnicornUdpFaultProfile, *, nominal_interval_s: float) -> None:
        profile.validate()
        if nominal_interval_s <= 0:
            raise ValueError("nominal_interval_s must be positive")
        self.profile = profile
        self.nominal_interval_s = float(nominal_interval_s)
        self._held_for_reorder: ScheduledDatagram | None = None

    @staticmethod
    def _periodic(sequence: int, every: int) -> bool:
        return every > 0 and (sequence + 1) % every == 0

    def _decorate(self, packet: ScheduledDatagram) -> tuple[ScheduledDatagram, ...]:
        profile = self.profile
        sequence = packet.source_sequence
        faults = list(packet.faults)
        release = packet.release_time_s
        if self._periodic(sequence, profile.delay_every):
            release += profile.delay_ms / 1000.0
            faults.append("delay")
        primary = replace(packet, release_time_s=release, faults=tuple(faults))
        if self._periodic(sequence, profile.duplicate_every):
            duplicate = replace(
                primary,
                duplicate_ordinal=1,
                release_time_s=primary.release_time_s + 1e-9,
                faults=primary.faults + ("duplicate",),
            )
            return (primary, duplicate)
        return (primary,)

    def process(
        self,
        payload: bytes,
        *,
        source_sequence: int,
        nominal_time_s: float,
    ) -> tuple[ScheduledDatagram, ...]:
        """Apply deterministic faults to one source packet.

        A dropped packet is still consumed by the source, so later device counter
        values reveal the gap naturally. This mirrors the correct causal layer:
        the transport loses a packet rather than rewinding the acquisition clock.
        """

        if source_sequence < 0:
            raise ValueError("source_sequence must be non-negative")
        packet = ScheduledDatagram(
            payload=bytes(payload),
            source_sequence=int(source_sequence),
            nominal_time_s=float(nominal_time_s),
            release_time_s=float(nominal_time_s),
        )

        if self._held_for_reorder is not None:
            held = self._held_for_reorder
            self._held_for_reorder = None
            current_dropped = self._periodic(source_sequence, self.profile.drop_every)
            outputs: list[ScheduledDatagram] = []
            if not current_dropped:
                current = replace(packet, faults=("reordered_before_previous",))
                outputs.extend(self._decorate(current))
            held_release = max(
                held.release_time_s,
                nominal_time_s + min(self.nominal_interval_s * 0.5, 0.001),
            )
            held = replace(
                held,
                release_time_s=held_release,
                faults=held.faults + ("reordered_after_next",),
            )
            outputs.extend(self._decorate(held))
            return tuple(outputs)

        if self._periodic(source_sequence, self.profile.drop_every):
            return ()

        if self._periodic(source_sequence, self.profile.reorder_every):
            self._held_for_reorder = replace(packet, faults=("held_for_reorder",))
            return ()

        return self._decorate(packet)

    def flush(self) -> tuple[ScheduledDatagram, ...]:
        """Release a final held packet when a stream stops between reorder pairs."""

        if self._held_for_reorder is None:
            return ()
        held = self._held_for_reorder
        self._held_for_reorder = None
        return self._decorate(replace(held, faults=held.faults + ("flush",)))


class UnicornRawUdpStreamSimulator:
    """Generate raw Unicorn-compatible UDP datagrams at a 250 Hz source cadence."""

    def __init__(
        self,
        *,
        seed: int = 7,
        fault_profile: UnicornUdpFaultProfile | None = None,
        byte_order: str = "<",
    ) -> None:
        self.device = UnicornHybridBlackSimulator(
            config=UnicornHybridBlackSimulationConfig(schema="device17_api", seed=seed)
        )
        self.byte_order = byte_order
        self.sequence = 0
        self.initial_delay_s = 0.0
        self.interval_s = 1.0 / self.device.spec.sampling_rate_hz
        self.faults = DeterministicPacketFaultEngine(
            fault_profile or FAULT_PROFILES["pristine"],
            nominal_interval_s=self.interval_s,
        )

    def next_datagrams(self) -> tuple[ScheduledDatagram, ...]:
        block = self.device.render(1)
        payload = encode_unicorn_udp_scan(block, 0, byte_order=self.byte_order)
        sequence = self.sequence
        self.sequence += 1
        return self.faults.process(
            payload,
            source_sequence=sequence,
            nominal_time_s=self.initial_delay_s + sequence * self.interval_s,
        )

    def flush(self) -> tuple[ScheduledDatagram, ...]:
        return self.faults.flush()


class UnicornBandpowerUdpStreamSimulator:
    """Generate 70-value Bandpower reference datagrams at the documented cadence.

    The public Bandpower interface uses a 250-sample analysis buffer and a
    10-sample hop. At 250 Hz, a stream starting with no history therefore needs
    one second of source acquisition before its first complete feature window can
    exist. ``initial_delay_s`` models that warm-up independently of the later
    25 Hz update cadence.
    """

    def __init__(
        self,
        *,
        seed: int = 7,
        fault_profile: UnicornUdpFaultProfile | None = None,
    ) -> None:
        self.device = UnicornHybridBlackSimulator(
            config=UnicornHybridBlackSimulationConfig(schema="eeg8_anatomical", seed=seed)
        )
        self.bandpower = UnicornBandpowerReferenceStream(
            sampling_rate_hz=self.device.spec.sampling_rate_hz
        )
        self.sequence = 0
        self.initial_delay_s = self.bandpower.buffer_size / self.bandpower.sampling_rate_hz
        self.interval_s = 1.0 / self.bandpower.update_rate_hz
        self.faults = DeterministicPacketFaultEngine(
            fault_profile or FAULT_PROFILES["pristine"],
            nominal_interval_s=self.interval_s,
        )
        self._primed = False

    def next_datagrams(self) -> tuple[ScheduledDatagram, ...]:
        samples = self.bandpower.buffer_size if not self._primed else self.bandpower.hop_samples
        self._primed = True
        block = self.device.render(samples)
        frames = self.bandpower.push(block.eeg_data_uv)
        if len(frames) != 1:
            raise AssertionError(f"expected one Bandpower frame per update, got {len(frames)}")
        payload = encode_unicorn_bandpower_ascii(frames[0].values)
        sequence = self.sequence
        self.sequence += 1
        return self.faults.process(
            payload,
            source_sequence=sequence,
            nominal_time_s=self.initial_delay_s + sequence * self.interval_s,
        )

    def flush(self) -> tuple[ScheduledDatagram, ...]:
        return self.faults.flush()
