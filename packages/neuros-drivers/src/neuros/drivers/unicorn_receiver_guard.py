"""Receiver-side safety contracts for Unicorn raw UDP game integrations.

The device simulator answers what a source can emit. This module answers the
other half of the contract: what a consumer should do when packets are malformed,
invalid, missing, duplicated, reordered, stale, or no longer permit exact counter
step inference.

Four concepts are deliberately kept separate:

* packet decodability: can the 68-byte payload be interpreted safely?
* sequence continuity: what does CNT say about ordering and missing samples?
* validation: is the device VALID field asserted?
* control authority: may neural data currently alter gameplay?

``health`` remains as a compact backward-compatible summary for simple game
integrations. Consumers that need diagnostics should inspect the orthogonal
fields instead.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .unicorn_network_sim import RAW_UDP_PAYLOAD_BYTES, decode_unicorn_udp_scan

# IEEE-754 float32 represents every integer exactly only through 2**24. The
# public Unicorn raw-UDP contract documents CNT as a float but does not document
# wrap/reset behavior. Above this bound, exact +1 packet-continuity inference is
# unsafe and the game guard deliberately fails closed.
FLOAT32_EXACT_INTEGER_MAX = 2**24

RawUdpHealth = Literal[
    "healthy",
    "malformed",
    "invalid",
    "gap",
    "duplicate",
    "out_of_order",
    "counter_ambiguous",
    "stale",
]
RawUdpPacketStatus = Literal["decodable", "malformed"]
RawUdpSequenceStatus = Literal[
    "unknown",
    "first",
    "sequential",
    "gap",
    "duplicate",
    "out_of_order",
    "precision_ambiguous",
]


@dataclass(frozen=True)
class UnicornRawUdpGuardConfig:
    """Consumer policy, not a manufacturer specification."""

    stale_after_s: float = 0.100
    recovery_packets: int = 3
    require_validation: bool = True
    byte_order: str = "<"

    def validate(self) -> None:
        if self.stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive")
        if self.recovery_packets <= 0:
            raise ValueError("recovery_packets must be positive")
        if self.byte_order not in {"<", ">", "=", "!"}:
            raise ValueError("unsupported byte order")


@dataclass(frozen=True)
class UnicornRawUdpObservation:
    health: RawUdpHealth
    received_monotonic_s: float
    counter: int | None
    battery_level: float | None
    validation: int | None
    missed_packets: int = 0
    healthy_streak: int = 0
    authority_allowed: bool = False
    reason: str = ""
    packet_status: RawUdpPacketStatus = "decodable"
    sequence_status: RawUdpSequenceStatus = "unknown"
    validation_asserted: bool | None = None
    counter_step_exact: bool | None = None


@dataclass(frozen=True)
class UnicornRawUdpGuardState:
    health: RawUdpHealth
    authority_allowed: bool
    healthy_streak: int
    last_counter: int | None
    age_s: float | None
    stream_live: bool = False
    sequence_status: RawUdpSequenceStatus = "unknown"
    validation_asserted: bool | None = None


class UnicornRawUdpGuard:
    """Fail-closed consumer state machine for 68-byte Unicorn raw UDP packets."""

    def __init__(self, config: UnicornRawUdpGuardConfig | None = None) -> None:
        self.config = config or UnicornRawUdpGuardConfig()
        self.config.validate()
        self._counter_high_water: int | None = None
        self._last_decodable_receive_s: float | None = None
        self._healthy_streak = 0
        self._last_health: RawUdpHealth = "stale"
        self._last_sequence_status: RawUdpSequenceStatus = "unknown"
        self._last_validation_asserted: bool | None = None

    @property
    def last_counter(self) -> int | None:
        """Highest observed counter in the current undocumented counter epoch."""

        return self._counter_high_water

    @property
    def healthy_streak(self) -> int:
        return self._healthy_streak

    def begin_new_epoch(self) -> None:
        """Explicitly start a new acquisition/counter epoch.

        The public raw-UDP interface does not document counter wrap/reset
        semantics, so the guard never infers an epoch transition from packet
        values. Applications should call this only when their own transport or
        device lifecycle knows that a new acquisition session has begun, for
        example after an intentional reconnect/restart. The operation revokes
        liveness and authority and requires the normal recovery streak again.
        """

        self._counter_high_water = None
        self._last_decodable_receive_s = None
        self._healthy_streak = 0
        self._last_health = "stale"
        self._last_sequence_status = "unknown"
        self._last_validation_asserted = None

    def _malformed(self, received_s: float, *, reason: str) -> UnicornRawUdpObservation:
        self._healthy_streak = 0
        self._last_health = "malformed"
        self._last_sequence_status = "unknown"
        self._last_validation_asserted = None
        return UnicornRawUdpObservation(
            health="malformed",
            received_monotonic_s=received_s,
            counter=None,
            battery_level=None,
            validation=None,
            authority_allowed=False,
            reason=reason,
            packet_status="malformed",
            sequence_status="unknown",
            validation_asserted=None,
            counter_step_exact=None,
        )

    def _sequence(
        self,
        *,
        counter: int,
        counter_float: float,
    ) -> tuple[RawUdpSequenceStatus, int, bool]:
        exact = abs(counter_float) <= FLOAT32_EXACT_INTEGER_MAX
        if self._counter_high_water is not None:
            exact = exact and abs(self._counter_high_water) <= FLOAT32_EXACT_INTEGER_MAX
        if not exact:
            if self._counter_high_water is None or counter > self._counter_high_water:
                self._counter_high_water = counter
            return "precision_ambiguous", 0, False

        if self._counter_high_water is None:
            self._counter_high_water = counter
            return "first", 0, True

        delta = counter - self._counter_high_water
        if delta == 0:
            return "duplicate", 0, True
        if delta < 0:
            return "out_of_order", 0, True

        self._counter_high_water = counter
        if delta == 1:
            return "sequential", 0, True
        return "gap", delta - 1, True

    @staticmethod
    def _health_summary(
        *,
        validation_asserted: bool,
        sequence_status: RawUdpSequenceStatus,
        require_validation: bool,
    ) -> RawUdpHealth:
        # Preserve the old compact behavior for VALID=0 while exposing sequence
        # anomalies independently through sequence_status.
        if require_validation and not validation_asserted:
            return "invalid"
        return {
            "first": "healthy",
            "sequential": "healthy",
            "gap": "gap",
            "duplicate": "duplicate",
            "out_of_order": "out_of_order",
            "precision_ambiguous": "counter_ambiguous",
            "unknown": "malformed",
        }[sequence_status]

    def ingest(self, payload: bytes, *, received_monotonic_s: float) -> UnicornRawUdpObservation:
        """Consume one datagram and update control-authority state."""

        received_s = float(received_monotonic_s)
        if not np.isfinite(received_s):
            raise ValueError("received_monotonic_s must be finite")

        if self._last_decodable_receive_s is not None:
            age_before_packet = max(0.0, received_s - self._last_decodable_receive_s)
            if age_before_packet > self.config.stale_after_s:
                self._healthy_streak = 0
                self._last_health = "stale"

        if len(payload) != RAW_UDP_PAYLOAD_BYTES:
            return self._malformed(
                received_s,
                reason=f"expected {RAW_UDP_PAYLOAD_BYTES} bytes, received {len(payload)}",
            )
        try:
            values = decode_unicorn_udp_scan(payload, byte_order=self.config.byte_order)
        except (ValueError, TypeError) as exc:
            return self._malformed(received_s, reason=str(exc))
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            return self._malformed(received_s, reason="packet contains non-finite or malformed values")

        battery = float(values[14])
        counter_float = float(values[15])
        validation_float = float(values[16])
        counter = int(round(counter_float))
        validation = int(round(validation_float))
        if abs(counter_float - counter) > 0.25:
            return self._malformed(received_s, reason="counter is not sufficiently integer-like")
        if validation not in {0, 1}:
            return self._malformed(received_s, reason="validation indicator is not binary")

        # Any structurally decodable packet refreshes stream liveness. Sequence
        # continuity is classified before VALID so an invalid packet cannot hide
        # a simultaneous transport gap, duplicate, or reorder condition.
        self._last_decodable_receive_s = received_s
        sequence_status, missed, counter_step_exact = self._sequence(
            counter=counter,
            counter_float=counter_float,
        )
        validation_asserted = validation == 1
        self._last_sequence_status = sequence_status
        self._last_validation_asserted = validation_asserted
        health = self._health_summary(
            validation_asserted=validation_asserted,
            sequence_status=sequence_status,
            require_validation=self.config.require_validation,
        )

        sequence_ok = sequence_status in {"first", "sequential"}
        validity_ok = validation_asserted or not self.config.require_validation
        if sequence_ok and validity_ok and counter_step_exact:
            self._healthy_streak += 1
        else:
            self._healthy_streak = 0

        self._last_health = health
        allowed = (
            health == "healthy"
            and self._healthy_streak >= self.config.recovery_packets
            and sequence_ok
            and validity_ok
            and counter_step_exact
        )

        reasons: list[str] = []
        if not validation_asserted:
            reasons.append("validation indicator is not asserted")
        if sequence_status == "gap":
            reasons.append(f"counter gap implies {missed} missing packet(s)")
        elif sequence_status == "duplicate":
            reasons.append("counter repeated")
        elif sequence_status == "out_of_order":
            reasons.append("counter arrived below the observed high-water mark")
        elif sequence_status == "precision_ambiguous":
            reasons.append(
                "counter exceeds float32 unit-step exactness; wrap/reset semantics are undocumented"
            )
        elif sequence_status in {"first", "sequential"} and validation_asserted:
            reasons.append("healthy sequential validated packet")
        elif sequence_status in {"first", "sequential"}:
            reasons.append("sequence is continuous")

        return UnicornRawUdpObservation(
            health=health,
            received_monotonic_s=received_s,
            counter=counter,
            battery_level=battery,
            validation=validation,
            missed_packets=missed,
            healthy_streak=self._healthy_streak,
            authority_allowed=allowed,
            reason="; ".join(reasons),
            packet_status="decodable",
            sequence_status=sequence_status,
            validation_asserted=validation_asserted,
            counter_step_exact=counter_step_exact,
        )

    def state(self, *, now_monotonic_s: float) -> UnicornRawUdpGuardState:
        now = float(now_monotonic_s)
        if not np.isfinite(now):
            raise ValueError("now_monotonic_s must be finite")
        if self._last_decodable_receive_s is None:
            return UnicornRawUdpGuardState(
                health="stale",
                authority_allowed=False,
                healthy_streak=self._healthy_streak,
                last_counter=self._counter_high_water,
                age_s=None,
                stream_live=False,
                sequence_status=self._last_sequence_status,
                validation_asserted=self._last_validation_asserted,
            )
        age = max(0.0, now - self._last_decodable_receive_s)
        if age > self.config.stale_after_s:
            return UnicornRawUdpGuardState(
                health="stale",
                authority_allowed=False,
                healthy_streak=0,
                last_counter=self._counter_high_water,
                age_s=age,
                stream_live=False,
                sequence_status=self._last_sequence_status,
                validation_asserted=self._last_validation_asserted,
            )
        return UnicornRawUdpGuardState(
            health=self._last_health,
            authority_allowed=(
                self._last_health == "healthy"
                and self._healthy_streak >= self.config.recovery_packets
            ),
            healthy_streak=self._healthy_streak,
            last_counter=self._counter_high_water,
            age_s=age,
            stream_live=True,
            sequence_status=self._last_sequence_status,
            validation_asserted=self._last_validation_asserted,
        )
