"""Deterministic SignalFrame -> NeuralWindow transforms."""

from __future__ import annotations

from enum import Enum

import numpy as np

from neuros.contracts import (
    ClockDomain,
    NeuralWindow,
    QualityFlag,
    SignalFrame,
    StreamDescriptor,
    TransformEmission,
    WindowSpec,
)


class DiscontinuityPolicy(str, Enum):
    """Behavior when frame continuity is broken."""

    ERROR = "error"
    RESET = "reset"


class SlidingWindowTransform:
    """Build decoder-ready windows from contiguous ``SignalFrame`` chunks.

    Input frames use neurOS' streaming convention: one-dimensional frames are a
    single multi-channel sample, while two-dimensional frames must declare
    ``axis_order=('sample', 'channel')``. Output windows are always
    ``(channels, time)``.

    The transform never pads, interpolates, resamples, or stitches sequence
    gaps. With ``discontinuity='error'`` (the default), gaps fail closed. The
    opt-in ``'reset'`` policy discards pending overlap and begins a new
    contiguous window segment.
    """

    def __init__(
        self,
        spec: WindowSpec,
        *,
        descriptor: StreamDescriptor | None = None,
        discontinuity: DiscontinuityPolicy | str = DiscontinuityPolicy.ERROR,
    ) -> None:
        self.spec = spec
        self.descriptor = descriptor
        self.discontinuity = DiscontinuityPolicy(discontinuity)

        self._stream_id: str | None = descriptor.stream_id if descriptor else None
        self._sample_rate_hz: float | None = (
            float(descriptor.sample_rate_hz) if descriptor else None
        )
        self._channel_names: tuple[str, ...] = (
            tuple(descriptor.channel_names) if descriptor else ()
        )
        self._channel_count: int | None = (
            len(self._channel_names) if self._channel_names else None
        )
        self._clock_domain: ClockDomain | None = (
            descriptor.clock_domain
            if descriptor and descriptor.clock_domain is not ClockDomain.UNKNOWN
            else None
        )

        self._data = np.empty((0, self._channel_count or 0), dtype=np.float64)
        self._times_ns = np.empty((0,), dtype=np.int64)
        self._sequence_per_sample = np.empty((0,), dtype=np.int64)
        self._quality_per_sample = np.empty((0,), dtype=np.int64)

        self._last_sequence_id: int | None = None
        self._next_window_id = 0
        self._discontinuity_count = 0

    @property
    def pending_samples(self) -> int:
        return int(self._data.shape[0])

    @property
    def discontinuity_count(self) -> int:
        return self._discontinuity_count

    def reset(self) -> None:
        """Discard pending overlap without changing stream geometry."""

        channel_count = self._channel_count or 0
        self._data = np.empty((0, channel_count), dtype=np.float64)
        self._times_ns = np.empty((0,), dtype=np.int64)
        self._sequence_per_sample = np.empty((0,), dtype=np.int64)
        self._quality_per_sample = np.empty((0,), dtype=np.int64)
        self._last_sequence_id = None
        self._clock_domain = (
            self.descriptor.clock_domain
            if self.descriptor
            and self.descriptor.clock_domain is not ClockDomain.UNKNOWN
            else None
        )

    @staticmethod
    def _matrix(frame: SignalFrame) -> np.ndarray:
        arr = np.asarray(frame.data)
        if arr.ndim == 1:
            matrix = arr.reshape(1, -1)
        elif arr.ndim == 2:
            axis_order = tuple(frame.metadata.get("axis_order", ()))
            if axis_order != ("sample", "channel"):
                raise ValueError(
                    "Two-dimensional SignalFrames require "
                    "metadata axis_order=('sample', 'channel') before windowing"
                )
            matrix = arr
        else:
            raise ValueError(
                "SlidingWindowTransform accepts one- or two-dimensional SignalFrames"
            )
        if matrix.shape[0] == 0 or matrix.shape[1] == 0:
            raise ValueError("SignalFrame cannot contain an empty sample/channel axis")
        if not np.isfinite(matrix).all():
            raise ValueError("SignalFrame contains NaN or infinite samples")
        return np.asarray(matrix)

    def _bind_geometry(self, frame: SignalFrame, matrix: np.ndarray) -> None:
        if self._stream_id is None:
            self._stream_id = frame.stream_id
        elif frame.stream_id != self._stream_id:
            raise ValueError(
                f"Window transform is bound to stream {self._stream_id!r}, "
                f"received {frame.stream_id!r}"
            )

        if self._sample_rate_hz is None:
            self._sample_rate_hz = float(frame.sample_rate_hz)
        elif not np.isclose(frame.sample_rate_hz, self._sample_rate_hz, rtol=0.0, atol=1e-12):
            raise ValueError("SignalFrame sample rate changed within a window stream")

        if self.descriptor is not None:
            if frame.stream_id != self.descriptor.stream_id:
                raise ValueError("SignalFrame stream_id does not match descriptor")
            if not np.isclose(
                frame.sample_rate_hz,
                self.descriptor.sample_rate_hz,
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError("SignalFrame sample rate does not match descriptor")

        metadata_names = tuple(frame.metadata.get("channel_names", ()))
        if metadata_names:
            if len(metadata_names) != matrix.shape[1]:
                raise ValueError("SignalFrame channel_names do not match channel axis")
            if self._channel_names and metadata_names != self._channel_names:
                raise ValueError("SignalFrame channel identity changed within stream")
            self._channel_names = metadata_names

        if self._channel_count is None:
            self._channel_count = int(matrix.shape[1])
            if not self._channel_names:
                self._channel_names = tuple(
                    f"ch{index}" for index in range(self._channel_count)
                )
            self._data = np.empty((0, self._channel_count), dtype=matrix.dtype)
        elif matrix.shape[1] != self._channel_count:
            raise ValueError("SignalFrame channel count changed within stream")

        if self._channel_names and len(self._channel_names) != self._channel_count:
            raise ValueError("Configured channel names do not match observed geometry")

    def _handle_continuity(self, frame: SignalFrame) -> None:
        if self._last_sequence_id is None:
            return

        expected = self._last_sequence_id + 1
        broken = frame.sequence_id != expected or bool(
            frame.quality & QualityFlag.DROPPED_SAMPLES
        )
        if not broken:
            return

        if self.discontinuity is DiscontinuityPolicy.ERROR:
            reason = (
                "dropped-sample quality flag"
                if frame.quality & QualityFlag.DROPPED_SAMPLES
                else f"sequence gap: expected {expected}, received {frame.sequence_id}"
            )
            raise ValueError(f"Cannot build a neural window across {reason}")

        self._discontinuity_count += 1
        channel_count = self._channel_count or 0
        self._data = np.empty((0, channel_count), dtype=self._data.dtype)
        self._times_ns = np.empty((0,), dtype=np.int64)
        self._sequence_per_sample = np.empty((0,), dtype=np.int64)
        self._quality_per_sample = np.empty((0,), dtype=np.int64)
        self._clock_domain = None

    def _bind_clock(self, frame: SignalFrame) -> None:
        domain = frame.clock_domain
        if self._clock_domain is None:
            self._clock_domain = domain
            return
        if domain != self._clock_domain:
            if self.pending_samples and self.discontinuity is DiscontinuityPolicy.ERROR:
                raise ValueError("Clock domain changed inside a pending neural window")
            if self.pending_samples:
                self._discontinuity_count += 1
                channel_count = self._channel_count or 0
                self._data = np.empty((0, channel_count), dtype=self._data.dtype)
                self._times_ns = np.empty((0,), dtype=np.int64)
                self._sequence_per_sample = np.empty((0,), dtype=np.int64)
                self._quality_per_sample = np.empty((0,), dtype=np.int64)
            self._clock_domain = domain

    def transform(self, frame: SignalFrame) -> TransformEmission | None:
        if not isinstance(frame, SignalFrame):
            raise TypeError("SlidingWindowTransform requires SignalFrame input")

        matrix = self._matrix(frame)
        self._bind_geometry(frame, matrix)
        self._handle_continuity(frame)
        self._bind_clock(frame)

        assert self._sample_rate_hz is not None
        assert self._stream_id is not None
        assert self._channel_count is not None

        period_ns = 1_000_000_000.0 / self._sample_rate_hz
        sample_times = np.asarray(
            [
                int(round(frame.timestamp_ns + offset * period_ns))
                for offset in range(matrix.shape[0])
            ],
            dtype=np.int64,
        )
        if self.pending_samples and sample_times[0] <= self._times_ns[-1]:
            if self.discontinuity is DiscontinuityPolicy.ERROR:
                raise ValueError("SignalFrame timestamps are not strictly increasing")
            self._discontinuity_count += 1
            self.reset()
            self._clock_domain = frame.clock_domain

        matrix = np.asarray(matrix)
        if self._data.dtype != matrix.dtype and self.pending_samples == 0:
            self._data = np.empty((0, self._channel_count), dtype=matrix.dtype)
        self._data = np.concatenate([self._data, matrix], axis=0)
        self._times_ns = np.concatenate([self._times_ns, sample_times])
        self._sequence_per_sample = np.concatenate(
            [
                self._sequence_per_sample,
                np.full(matrix.shape[0], frame.sequence_id, dtype=np.int64),
            ]
        )
        self._quality_per_sample = np.concatenate(
            [
                self._quality_per_sample,
                np.full(matrix.shape[0], int(frame.quality), dtype=np.int64),
            ]
        )
        self._last_sequence_id = frame.sequence_id

        windows: list[NeuralWindow] = []
        while self.pending_samples >= self.spec.window_samples:
            stop = self.spec.window_samples
            sample_major = self._data[:stop]
            source_sequences = tuple(
                dict.fromkeys(
                    int(value) for value in self._sequence_per_sample[:stop].tolist()
                )
            )
            quality_value = 0
            for value in self._quality_per_sample[:stop]:
                quality_value |= int(value)

            start_time_ns = int(self._times_ns[0])
            end_time_ns = int(round(self._times_ns[stop - 1] + period_ns))
            windows.append(
                NeuralWindow(
                    stream_id=self._stream_id,
                    window_id=self._next_window_id,
                    data=sample_major.T.copy(),
                    sample_rate_hz=self._sample_rate_hz,
                    start_time_ns=start_time_ns,
                    end_time_ns=end_time_ns,
                    channel_names=self._channel_names,
                    source_sequence_ids=source_sequences,
                    clock_domain=self._clock_domain or ClockDomain.UNKNOWN,
                    quality=QualityFlag(quality_value),
                    metadata={
                        "representation": "neural_window",
                        "axis_order": ("channel", "time"),
                        "window_samples": self.spec.window_samples,
                        "stride_samples": self.spec.stride_samples,
                        "overlap_samples": self.spec.overlap_samples,
                        "discontinuity_count": self._discontinuity_count,
                    },
                )
            )
            self._next_window_id += 1

            stride = self.spec.stride_samples
            self._data = self._data[stride:]
            self._times_ns = self._times_ns[stride:]
            self._sequence_per_sample = self._sequence_per_sample[stride:]
            self._quality_per_sample = self._quality_per_sample[stride:]

        return TransformEmission(tuple(windows)) if windows else None
