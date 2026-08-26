"""Reference out-of-tree neurOS source plugin."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator

import numpy as np

from neuros.contracts import ClockDomain, SignalFrame, StreamDescriptor


class SineSource:
    """Deterministic multi-channel sine source implementing ``neuros.Source``.

    The numerical samples are deterministic for a fixed configuration. Host
    receive timestamps intentionally use the host monotonic clock because an
    external source must not fabricate a device clock it does not possess.
    """

    def __init__(
        self,
        sampling_rate: float = 250.0,
        channels: int = 4,
        frequency_hz: float = 10.0,
        amplitude_uv: float = 20.0,
        chunk_samples: int = 25,
        stream_id: str = "example-eeg",
    ) -> None:
        if sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive")
        if channels <= 0:
            raise ValueError("channels must be positive")
        if frequency_hz <= 0 or frequency_hz >= sampling_rate / 2:
            raise ValueError("frequency_hz must be between 0 and Nyquist")
        if amplitude_uv < 0:
            raise ValueError("amplitude_uv must be non-negative")
        if chunk_samples <= 0:
            raise ValueError("chunk_samples must be positive")
        if not stream_id:
            raise ValueError("stream_id must be non-empty")

        self.sampling_rate = float(sampling_rate)
        self.channels = int(channels)
        self.frequency_hz = float(frequency_hz)
        self.amplitude_uv = float(amplitude_uv)
        self.chunk_samples = int(chunk_samples)
        self.stream_id = stream_id
        self._running = False
        self._sequence_id = 0
        self._sample_cursor = 0
        self._descriptor = StreamDescriptor(
            stream_id=stream_id,
            modality="eeg",
            sample_rate_hz=self.sampling_rate,
            channel_names=tuple(f"EX{i + 1}" for i in range(self.channels)),
            channel_types=("eeg",) * self.channels,
            units=("uV",) * self.channels,
            device="ExampleSineSource",
            manufacturer="neurOS example plugin",
            clock_domain=ClockDomain.HOST_MONOTONIC,
            metadata={
                "plugin": "neuros-example-plugin",
                "generator": "deterministic_sine",
                "frequency_hz": self.frequency_hz,
                "amplitude_uv": self.amplitude_uv,
                "axis_order": ("sample", "channel"),
            },
        )

    @property
    def descriptor(self) -> StreamDescriptor:
        return self._descriptor

    async def start(self) -> None:
        self._running = True

    async def stop(self) -> None:
        self._running = False

    def frames(self) -> AsyncIterator[SignalFrame]:
        return self._frames()

    async def _frames(self) -> AsyncIterator[SignalFrame]:
        if not self._running:
            raise RuntimeError("SineSource.start() must be called before frames()")

        period_s = self.chunk_samples / self.sampling_rate
        channel_phase = np.arange(self.channels, dtype=float) * (np.pi / 8.0)
        while self._running:
            sample_index = self._sample_cursor + np.arange(self.chunk_samples, dtype=float)
            phase = (2.0 * np.pi * self.frequency_hz * sample_index / self.sampling_rate)[:, None]
            data = self.amplitude_uv * np.sin(phase + channel_phase[None, :])
            now_ns = time.monotonic_ns()
            yield SignalFrame(
                stream_id=self.stream_id,
                sequence_id=self._sequence_id,
                data=data,
                sample_rate_hz=self.sampling_rate,
                host_receive_time_ns=now_ns,
                clock_domain=ClockDomain.HOST_MONOTONIC,
                metadata={
                    "axis_order": ("sample", "channel"),
                    "plugin": "neuros-example-plugin",
                    "generator": "deterministic_sine",
                },
            )
            self._sequence_id += 1
            self._sample_cursor += self.chunk_samples
            await asyncio.sleep(period_s)
