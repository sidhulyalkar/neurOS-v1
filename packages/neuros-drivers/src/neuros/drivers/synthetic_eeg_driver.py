"""neurOS driver wrapper for the protocol-grade synthetic EEG generator."""
from __future__ import annotations

import asyncio
import time
from typing import AsyncIterator, Sequence

import numpy as np

from neuros.contracts import ClockDomain, StreamDescriptor
from neuros.drivers.base_driver import BaseDriver
from neuros.drivers.synthetic_eeg import (
    ArtifactEvent,
    ArtifactKind,
    SyntheticEEGConfig,
    SyntheticEEGGenerator,
)

SYNTHETIC_EEG_GENERATOR_CONTRACT = "neuros.synthetic_eeg.v3"
SYNTHETIC_EEG_ARTIFACT_SCHEDULER_CONTRACT = "neuros.synthetic_eeg.artifact_schedule.v1"


class SyntheticEEGDriver(BaseDriver):
    """Stream controllable synthetic EEG through the canonical driver interface."""

    def __init__(
        self,
        config: SyntheticEEGConfig | None = None,
        *,
        realtime: bool = True,
        stream_id: str = "synthetic-eeg",
    ) -> None:
        self.generator = SyntheticEEGGenerator(config)
        self.realtime = bool(realtime)
        super().__init__(
            sampling_rate=self.generator.config.sampling_rate_hz,
            channels=len(self.generator.config.channel_names),
            stream_id=stream_id,
            modality="eeg",
        )

    @property
    def descriptor(self) -> StreamDescriptor:
        config = self.generator.config
        return StreamDescriptor(
            stream_id=self.stream_id,
            modality="eeg",
            sample_rate_hz=self.sampling_rate,
            channel_names=config.channel_names,
            channel_types=tuple("eeg" for _ in range(self.channels)),
            clock_domain=ClockDomain.HOST_MONOTONIC,
            device="SyntheticEEGGenerator",
            metadata={
                "synthetic": True,
                "units": "microvolts",
                "generator": SYNTHETIC_EEG_GENERATOR_CONTRACT,
                "artifact_scheduler": SYNTHETIC_EEG_ARTIFACT_SCHEDULER_CONTRACT,
                # Artifact/control schedules are dynamic experiment inputs and
                # are intentionally not implied by this static stream descriptor.
                # Recorded samples remain replay authority; scenario manifests
                # should persist dynamic schedules separately when regeneration
                # rather than replay is required.
                "artifact_schedule_in_descriptor": False,
                # Persist the stochastic-world inputs beside the stream identity.
                # A seed without these parameters or a generator contract is not
                # sufficient to reconstruct the same world across revisions.
                "generator_config": {
                    "sampling_rate_hz": float(config.sampling_rate_hz),
                    "channel_names": list(config.channel_names),
                    "colored_noise_uv": float(config.colored_noise_uv),
                    "white_noise_uv": float(config.white_noise_uv),
                    "alpha_frequency_hz": float(config.alpha_frequency_hz),
                    "alpha_amplitude_uv": float(config.alpha_amplitude_uv),
                    "ssvep_amplitude_uv": float(config.ssvep_amplitude_uv),
                    "first_harmonic_ratio": float(config.first_harmonic_ratio),
                    "seed": int(config.seed),
                },
            },
        )

    def set_attention(self, frequency_hz: float | None, gain: float = 1.0) -> None:
        self.generator.set_attention(frequency_hz, gain)

    def inject_artifact(
        self,
        kind: ArtifactKind,
        duration_seconds: float = 0.35,
        severity: float = 1.0,
    ) -> None:
        self.generator.inject_artifact(kind, duration_seconds, severity)

    def schedule_artifact(
        self,
        kind: ArtifactKind,
        *,
        event_id: str,
        duration_seconds: float = 0.35,
        severity: float = 1.0,
        start_sample: int | None = None,
        delay_seconds: float = 0.0,
        channels: str | int | Sequence[str | int] | None = None,
        seed: int | None = None,
    ) -> ArtifactEvent:
        return self.generator.schedule_artifact(
            kind,
            event_id=event_id,
            duration_seconds=duration_seconds,
            severity=severity,
            start_sample=start_sample,
            delay_seconds=delay_seconds,
            channels=channels,
            seed=seed,
        )

    def cancel_artifact(self, event_id: str) -> bool:
        return self.generator.cancel_artifact(event_id)

    @property
    def scheduled_artifacts(self) -> tuple[ArtifactEvent, ...]:
        return self.generator.scheduled_artifacts

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
