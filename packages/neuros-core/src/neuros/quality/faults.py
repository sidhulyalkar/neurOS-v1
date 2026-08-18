"""Deterministic fault injection for neural-stream resilience testing."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any, AsyncIterator

import numpy as np

from neuros.contracts import QualityFlag, SignalFrame, StreamDescriptor


@dataclass(frozen=True, slots=True)
class FaultProfile:
    seed: int = 0
    packet_drop_probability: float = 0.0
    timestamp_jitter_std_ms: float = 0.0
    channel_dropout_probability: float = 0.0
    clock_drift_ppm: float = 0.0
    additive_noise_std: float = 0.0

    def __post_init__(self) -> None:
        for name in ("packet_drop_probability", "channel_dropout_probability"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.timestamp_jitter_std_ms < 0 or self.additive_noise_std < 0:
            raise ValueError("jitter/noise standard deviations must be non-negative")


def perturb_frame(
    frame: SignalFrame,
    profile: FaultProfile,
    rng: np.random.Generator,
    *,
    origin_device_time_ns: int | None = None,
) -> SignalFrame:
    """Apply frame-local deterministic perturbations without changing sequence ID."""

    data = np.asarray(frame.data).copy()
    quality = frame.quality
    metadata = dict(frame.metadata)

    if profile.channel_dropout_probability > 0 and data.ndim >= 1:
        channel_count = data.shape[0]
        mask = rng.random(channel_count) < profile.channel_dropout_probability
        if mask.any():
            data[mask] = 0
            quality |= QualityFlag.DISCONNECTED_CHANNEL
            metadata["fault_channel_dropout"] = tuple(int(index) for index in np.flatnonzero(mask))

    if profile.additive_noise_std > 0:
        data = data.astype(float, copy=False) + rng.normal(
            0.0, profile.additive_noise_std, size=data.shape
        )
        metadata["fault_additive_noise_std"] = profile.additive_noise_std

    jitter_ns = 0
    if profile.timestamp_jitter_std_ms > 0:
        jitter_ns = int(rng.normal(0.0, profile.timestamp_jitter_std_ms * 1_000_000.0))
        metadata["fault_timestamp_jitter_ns"] = jitter_ns

    device_time = frame.device_time_ns
    synchronized_time = frame.synchronized_time_ns
    if device_time is not None and profile.clock_drift_ppm:
        origin = origin_device_time_ns if origin_device_time_ns is not None else device_time
        elapsed = device_time - origin
        drift_ns = int(elapsed * profile.clock_drift_ppm / 1_000_000.0)
        device_time += drift_ns
        quality |= QualityFlag.CLOCK_UNCERTAIN
        metadata["fault_clock_drift_ppm"] = profile.clock_drift_ppm

    if synchronized_time is not None:
        synchronized_time += jitter_ns
    elif device_time is not None:
        device_time += jitter_ns

    return replace(
        frame,
        data=data,
        device_time_ns=device_time,
        synchronized_time_ns=synchronized_time,
        quality=quality,
        metadata=metadata,
    )


class PerturbedSource:
    """Wrap any Source with reproducible packet/data/timing perturbations."""

    def __init__(self, source: Any, profile: FaultProfile) -> None:
        self.source = source
        self.profile = profile
        self._rng = np.random.default_rng(profile.seed)
        self._origin_device_time_ns: int | None = None
        self.dropped_packets = 0

    @property
    def descriptor(self) -> StreamDescriptor:
        return self.source.descriptor

    async def start(self) -> None:
        self._rng = np.random.default_rng(self.profile.seed)
        self._origin_device_time_ns = None
        self.dropped_packets = 0
        await self.source.start()

    async def stop(self) -> None:
        await self.source.stop()

    async def frames(self) -> AsyncIterator[SignalFrame]:
        async for frame in self.source.frames():
            if self._origin_device_time_ns is None and frame.device_time_ns is not None:
                self._origin_device_time_ns = frame.device_time_ns
            if self.profile.packet_drop_probability > 0 and self._rng.random() < self.profile.packet_drop_probability:
                self.dropped_packets += 1
                await asyncio.sleep(0)
                continue
            yield perturb_frame(
                frame,
                self.profile,
                self._rng,
                origin_device_time_ns=self._origin_device_time_ns,
            )
