"""Receiver-side safety contracts for Unicorn raw UDP game integrations.

The device simulator answers what a source can emit. This module answers the
other half of the contract: what a consumer should do when packets are malformed,
invalid, missing, duplicated, reordered, or stale.

The guard deliberately treats transport health as *control authority*. A game may
continue rendering during faults, but neural data should not continue changing
state until the stream has recovered for a configurable number of consecutive
packets.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .unicorn_network_sim import RAW_UDP_PAYLOAD_BYTES, decode_unicorn_udp_scan

RawUdpHealth = Literal[
    "healthy",
    "malformed",
    "invalid",
    "gap",
    "duplicate",
    "out_of_order",
    "stale",
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


@dataclass(frozen=True)
class UnicornRawUdpGuardState:
    health: RawUdpHealth
    authority_allowed: bool
    healthy_streak: int
    last_counter: int | None
    age_s: float | None


class UnicornRawUdpGuard:
    """Fail-closed consumer state machine for 68-byte Unicorn raw UDP packets."""

    def __init__(self, config: UnicornRawUdpGuardConfig | None = None) -> None:
        self.config = config or UnicornRawUdpGuardConfig()
        self.config.validate()
        self._last_counter: int | None = None
        # Most recent structurally decodable packet, even if VALID=0. Stream
        # liveness and gameplay authority are intentionally separate concepts.
        self._last_valid_receive_s: float | None = None
        self._healthy_streak = 0
        self._last_health: RawUdpHealth = "stale"

    @property
    def last_counter(self) -> int | None:
        return self._last_counter

    @property
    def healthy_streak(self) -> int:
        return self._healthy_streak

    def _fault(
        self,
        health: RawUdpHealth,
        received_s: float,
        *,
        counter: int | None = None,
        battery: float | None = None,
        validation: int | None = None,
        missed: int = 0,
        reason: str,
    ) -> UnicornRawUdpObservation:
        self._healthy_streak = 0
        self._last_health = health
        return UnicornRawUdpObservation(
            health=health,
            received_monotonic_s=float(received_s),
            counter=counter,
            battery_level=battery,
            validation=validation,
            missed_packets=missed,
            healthy_streak=0,
            authority_allowed=False,
            reason=reason,
        )

    def ingest(self, payload: bytes, *, received_monotonic_s: float) -> UnicornRawUdpObservation:
        """Consume one datagram and update control-authority state."""

        received_s = float(received_monotonic_s)
        if not np.isfinite(received_s):
            raise ValueError("received_monotonic_s must be finite")
        if self._last_valid_receive_s is not None:
            age_before_packet = max(0.0, received_s - self._last_valid_receive_s)
            if age_before_packet > self.config.stale_after_s:
                # Fresh traffic re-establishes liveness but cannot inherit neural
                # authority accumulated before a stale interval.
                self._healthy_streak = 0
                self._last_health = "stale"
        if len(payload) != RAW_UDP_PAYLOAD_BYTES:
            return self._fault(
                "malformed",
                received_s,
                reason=f"expected {RAW_UDP_PAYLOAD_BYTES} bytes, received {len(payload)}",
            )
        try:
            values = decode_unicorn_udp_scan(payload, byte_order=self.config.byte_order)
        except (ValueError, TypeError) as exc:
            return self._fault("malformed", received_s, reason=str(exc))
        if values.shape != (17,) or not np.all(np.isfinite(values)):
            return self._fault("malformed", received_s, reason="packet contains non-finite or malformed values")

        # Standalone raw UDP order is ... GYR, BAT, CNT, VALID.
        battery = float(values[14])
        counter_float = float(values[15])
        validation_float = float(values[16])
        counter = int(round(counter_float))
        validation = int(round(validation_float))
        if abs(counter_float - counter) > 0.25:
            return self._fault(
                "malformed",
                received_s,
                battery=battery,
                validation=validation,
                reason="counter is not sufficiently integer-like",
            )
        if validation not in {0, 1}:
            return self._fault(
                "malformed",
                received_s,
                counter=counter,
                battery=battery,
                validation=validation,
                reason="validation indicator is not binary",
            )
        if self.config.require_validation and validation != 1:
            # The packet is structurally observable and advances the device
            # counter, but its neural payload cannot grant gameplay authority.
            self._last_counter = counter
            self._last_valid_receive_s = received_s
            return self._fault(
                "invalid",
                received_s,
                counter=counter,
                battery=battery,
                validation=validation,
                reason="validation indicator is not asserted",
            )

        if self._last_counter is not None:
            delta = counter - self._last_counter
            if delta == 0:
                self._last_valid_receive_s = received_s
                return self._fault(
                    "duplicate",
                    received_s,
                    counter=counter,
                    battery=battery,
                    validation=validation,
                    reason="counter repeated",
                )
            if delta < 0:
                self._last_valid_receive_s = received_s
                return self._fault(
                    "out_of_order",
                    received_s,
                    counter=counter,
                    battery=battery,
                    validation=validation,
                    reason="counter moved backwards",
                )
            if delta > 1:
                missed = delta - 1
                self._last_counter = counter
                self._last_valid_receive_s = received_s
                return self._fault(
                    "gap",
                    received_s,
                    counter=counter,
                    battery=battery,
                    validation=validation,
                    missed=missed,
                    reason=f"counter advanced by {delta}; {missed} packet(s) missing",
                )

        self._last_counter = counter
        self._last_valid_receive_s = received_s
        self._healthy_streak += 1
        self._last_health = "healthy"
        allowed = self._healthy_streak >= self.config.recovery_packets
        return UnicornRawUdpObservation(
            health="healthy",
            received_monotonic_s=received_s,
            counter=counter,
            battery_level=battery,
            validation=validation,
            healthy_streak=self._healthy_streak,
            authority_allowed=allowed,
            reason="healthy sequential validated packet",
        )

    def state(self, *, now_monotonic_s: float) -> UnicornRawUdpGuardState:
        now = float(now_monotonic_s)
        if not np.isfinite(now):
            raise ValueError("now_monotonic_s must be finite")
        if self._last_valid_receive_s is None:
            return UnicornRawUdpGuardState(
                health="stale",
                authority_allowed=False,
                healthy_streak=self._healthy_streak,
                last_counter=self._last_counter,
                age_s=None,
            )
        age = max(0.0, now - self._last_valid_receive_s)
        if age > self.config.stale_after_s:
            return UnicornRawUdpGuardState(
                health="stale",
                authority_allowed=False,
                healthy_streak=0,
                last_counter=self._last_counter,
                age_s=age,
            )
        return UnicornRawUdpGuardState(
            health=self._last_health,
            authority_allowed=(
                self._last_health == "healthy"
                and self._healthy_streak >= self.config.recovery_packets
            ),
            healthy_streak=self._healthy_streak,
            last_counter=self._last_counter,
            age_s=age,
        )
