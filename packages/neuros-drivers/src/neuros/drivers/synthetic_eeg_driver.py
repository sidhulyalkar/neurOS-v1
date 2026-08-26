"""neurOS driver wrapper for the protocol-grade synthetic EEG generator."""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator

import numpy as np

from neuros.contracts import ClockDomain, StreamDescriptor
from neuros.drivers.base_driver import BaseDriver
from neuros.drivers.synthetic_eeg import ArtifactKind, SyntheticEEGConfig, SyntheticEEGGenerator


class SyntheticEEGDriver(BaseDriver):
    """Stream controllable synthetic EEG through the canonical driver interface."""

    def __init__(self, config: SyntheticEEGConfig | None = None, *, realtime: bool = True, stream_id: str = "synthetic-eeg") -> None:
        self.generator = SyntheticEEGGenerator(config)
        self.realtime = bool(realtime)
        super().__init__(sampling_rate=self.generator.config.sampling_rate_hz, channels=len(self.generator.config.channel_names), stream_id=stream_id, modality="eeg")

    @property
    def descriptor(self) -> StreamDescriptor:
        return StreamDescriptor(
            stream_id=self.stream_id,
            modality="eeg",
            sample_rate_hz=self.sampling_rate,
            channel_names=self.generator.config.channel_names,
            channel_types=tuple("eeg" for _ in range(self.channels)),
            clock_domain=ClockDomain.HOST_MONOTONIC,
            device="SyntheticEEGGenerator",
            metadata={"synthetic": True, "units": "microvolts", "generator": "neuros.synthetic_eeg.v1"},
        )

    def set_attention(self, frequency_hz: float | None, gain: float = 1.0) -> None:
        self.generator.set_attention(frequency_hz, gain)

    def inject_artifact(self, kind: ArtifactKind, duration_seconds: float = 0.35, severity: float = 1.0) -> None:
        self.generator.inject_artifact(kind, duration_seconds, severity)

    def set_channel_gain(self, channel: str | int, gain: float) -> None:
        self.generator.set_channel_gain(channel, gain)

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        period = 1.0 / self.sampling_rate
        next_deadline = time.monotonic()
        while self._running:
            block = self.generator.render(1)
            yield time.time(), block.data_uv[:, 0]
            if self.realtime:
                next_deadline += period
                await asyncio.sleep(max(0.0, next_deadline - time.monotonic()))
            else:
                await asyncio.sleep(0)
