"""Fail-closed Lab Streaming Layer source for neurOS.

The driver resolves one unambiguous continuous LSL stream, keeps liblsl
post-processing disabled, applies the explicit LSL time-correction estimate,
and publishes canonical :class:`~neuros.contracts.SignalFrame` objects in the
synchronized clock domain.

This module deliberately does not enable LSL dejitter/monotonization flags.
Those operations alter timing semantics and should be explicit transforms in a
qualified pipeline rather than hidden acquisition behavior.
"""

from __future__ import annotations

import asyncio
import math
import time
from contextlib import suppress
from typing import Any, AsyncIterator

import numpy as np

from neuros.contracts import ClockDomain, QualityFlag, SignalFrame, StreamDescriptor
from neuros.drivers.base_driver import BaseDriver
from neuros.runtime import OverflowPolicy, put_with_policy


class LSLDriver(BaseDriver):
    """Receive one continuous numeric stream through Lab Streaming Layer.

    At least one deterministic selector must be supplied. If a selector still
    matches multiple streams, startup fails instead of attaching to whichever
    outlet happened to be discovered first.

    Parameters
    ----------
    source_id:
        Preferred LSL source identity. Use this whenever the producer exposes
        a stable source ID.
    name:
        Optional exact LSL stream name constraint.
    stream_type:
        Optional exact LSL content type constraint, for example ``"EEG"``.
    sampling_rate:
        Optional expected nominal sampling rate. It is an assertion, not a
        resampler.
    channels:
        Optional expected channel count. It is an assertion, not a selector.
    resolve_timeout:
        Maximum seconds spent resolving the primary selector at startup. The
        complete window is deliberately used to detect a second matching
        outlet before a stream is accepted.
    open_timeout:
        Maximum seconds spent opening the selected inlet.
    time_correction_timeout:
        Maximum seconds for the initial LSL clock-correction estimate.
    correction_refresh_seconds:
        Refresh cadence for LSL's clock-correction estimate. Set to ``0`` to
        retain the startup estimate for the session.
    poll_interval_seconds:
        Cooperative sleep used when no chunk is immediately available. LSL
        pulls themselves are non-blocking so the neurOS event loop is not held
        by a long acquisition call.
    max_samples:
        Maximum samples requested from one LSL chunk pull.
    max_buflen:
        liblsl inlet buffer length in seconds for regular-rate streams.
    recover:
        Request liblsl recovery. Recovery is only effective for streams that
        expose a non-empty ``source_id``.
    """

    def __init__(
        self,
        *,
        source_id: str | None = None,
        name: str | None = None,
        stream_type: str | None = None,
        sampling_rate: float | None = None,
        channels: int | None = None,
        resolve_timeout: float = 2.0,
        open_timeout: float = 2.0,
        time_correction_timeout: float = 1.0,
        correction_refresh_seconds: float = 5.0,
        poll_interval_seconds: float = 0.002,
        max_samples: int = 256,
        max_buflen: int = 5,
        recover: bool = True,
        stream_id: str | None = None,
        modality: str | None = None,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
    ) -> None:
        try:
            from pylsl import StreamInlet, resolve_byprop  # type: ignore
        except (ImportError, OSError, RuntimeError) as exc:
            raise ImportError(
                "LSLDriver requires pylsl and a loadable liblsl runtime. "
                "Install `neuros-drivers[eeg]` and follow pylsl's liblsl "
                "installation guidance for the host platform."
            ) from exc

        self._source_id = self._clean_selector(source_id)
        self._name = self._clean_selector(name)
        self._stream_type = self._clean_selector(stream_type)
        if not any((self._source_id, self._name, self._stream_type)):
            raise ValueError(
                "LSLDriver requires source_id, name, or stream_type; neurOS will "
                "not attach to an arbitrary discovered LSL stream"
            )

        if sampling_rate is not None and sampling_rate <= 0:
            raise ValueError("sampling_rate must be positive when provided")
        if channels is not None and channels <= 0:
            raise ValueError("channels must be positive when provided")
        if resolve_timeout <= 0 or open_timeout <= 0 or time_correction_timeout <= 0:
            raise ValueError("LSL startup timeouts must be positive")
        if correction_refresh_seconds < 0:
            raise ValueError("correction_refresh_seconds must be >= 0")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        if max_samples <= 0 or max_buflen <= 0:
            raise ValueError("max_samples and max_buflen must be positive")

        self._StreamInlet = StreamInlet
        self._resolve_byprop = resolve_byprop
        self._expected_sampling_rate = float(sampling_rate) if sampling_rate is not None else None
        self._expected_channels = int(channels) if channels is not None else None
        self._resolve_timeout = float(resolve_timeout)
        self._open_timeout = float(open_timeout)
        self._time_correction_timeout = float(time_correction_timeout)
        self._correction_refresh_seconds = float(correction_refresh_seconds)
        self._poll_interval_seconds = float(poll_interval_seconds)
        self._max_samples = int(max_samples)
        self._max_buflen = int(max_buflen)
        self._recover_requested = bool(recover)
        self._explicit_stream_id = stream_id
        self._explicit_modality = modality

        self._inlet: Any | None = None
        self._resolved_info: Any | None = None
        self._descriptor: StreamDescriptor | None = None
        self._time_correction_seconds: float | None = None
        self._last_correction_refresh_monotonic = 0.0
        self._sequence_id = 0

        super().__init__(
            sampling_rate=self._expected_sampling_rate or 1.0,
            channels=self._expected_channels or 1,
            stream_id=stream_id or "lsl-unresolved",
            modality=modality or (self._stream_type or "unknown").lower(),
            overflow_policy=overflow_policy,
        )

    @staticmethod
    def _clean_selector(value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("LSL selectors must not be empty strings")
        return cleaned

    @staticmethod
    def _info_value(info: Any, field: str) -> Any:
        value = getattr(info, field, None)
        return value() if callable(value) else value

    def _matches_constraints(self, info: Any) -> bool:
        constraints = {
            "source_id": self._source_id,
            "name": self._name,
            "type": self._stream_type,
        }
        for field, expected in constraints.items():
            if expected is None:
                continue
            if str(self._info_value(info, field) or "") != expected:
                return False
        return True

    def _stream_identity(self, info: Any) -> str:
        return (
            f"name={self._info_value(info, 'name')!r}, "
            f"type={self._info_value(info, 'type')!r}, "
            f"source_id={self._info_value(info, 'source_id')!r}, "
            f"uid={self._info_value(info, 'uid')!r}"
        )

    def _resolve_one(self) -> Any:
        if self._source_id is not None:
            prop, value = "source_id", self._source_id
        elif self._name is not None:
            prop, value = "name", self._name
        else:
            prop, value = "type", self._stream_type

        # pylsl's resolver may return as soon as ``minimum`` candidates are
        # found. Request two so the full timeout is spent checking for a second
        # matching outlet when only one is initially visible. This makes the
        # fail-on-ambiguity contract deterministic within the discovery window.
        candidates = list(
            self._resolve_byprop(
                prop,
                value,
                minimum=2,
                timeout=self._resolve_timeout,
            )
        )
        candidates = [item for item in candidates if self._matches_constraints(item)]
        if not candidates:
            raise RuntimeError(
                "No LSL stream matched the requested selectors within "
                f"{self._resolve_timeout:g}s"
            )
        if len(candidates) > 1:
            identities = "; ".join(self._stream_identity(item) for item in candidates[:5])
            raise RuntimeError(
                "LSL stream selection is ambiguous; refine source_id/name/type. "
                f"Matches: {identities}"
            )
        return candidates[0]

    def _extract_channel_names(self, info: Any, channel_count: int) -> tuple[str, ...]:
        fallback = tuple(f"lsl_ch{index}" for index in range(channel_count))
        try:
            node = info.desc().child("channels").child("channel")
            names: list[str] = []
            for index in range(channel_count):
                if node is None or (hasattr(node, "empty") and node.empty()):
                    return fallback
                label = str(node.child_value("label") or "").strip()
                names.append(label or fallback[index])
                node = node.next_sibling()
            return tuple(names)
        except Exception:
            return fallback

    def _validate_info(self, info: Any) -> tuple[float, int]:
        sample_rate = float(self._info_value(info, "nominal_srate") or 0.0)
        channel_count = int(self._info_value(info, "channel_count") or 0)
        if not math.isfinite(sample_rate) or sample_rate <= 0:
            raise ValueError(
                "LSLDriver v1 supports continuous regular-rate streams only; "
                f"the selected stream reports nominal_srate={sample_rate!r}"
            )
        if channel_count <= 0:
            raise ValueError("The selected LSL stream reports no channels")
        if self._expected_sampling_rate is not None and not math.isclose(
            sample_rate,
            self._expected_sampling_rate,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "LSL stream sampling rate mismatch: expected "
                f"{self._expected_sampling_rate:g} Hz, discovered {sample_rate:g} Hz"
            )
        if self._expected_channels is not None and channel_count != self._expected_channels:
            raise ValueError(
                "LSL stream channel-count mismatch: expected "
                f"{self._expected_channels}, discovered {channel_count}"
            )
        return sample_rate, channel_count

    @property
    def descriptor(self) -> StreamDescriptor:
        if self._descriptor is None:
            raise RuntimeError("LSL stream is unresolved; call `await driver.start()` first")
        return self._descriptor

    async def start(self) -> None:
        if self._running:
            return

        try:
            resolved = self._resolve_one()
            sample_rate, channel_count = self._validate_info(resolved)
            discovered_source_id = str(self._info_value(resolved, "source_id") or "")
            effective_recover = self._recover_requested and bool(discovered_source_id)

            # Keep liblsl timing post-processing disabled. neurOS records the
            # explicit raw timestamp + time_correction mapping instead.
            inlet = self._StreamInlet(
                resolved,
                max_buflen=self._max_buflen,
                max_chunklen=0,
                recover=effective_recover,
                processing_flags=0,
            )
            self._inlet = inlet
            open_stream = getattr(inlet, "open_stream", None)
            if callable(open_stream):
                open_stream(timeout=self._open_timeout)

            full_info = resolved
            info_getter = getattr(inlet, "info", None)
            if callable(info_getter):
                full_info = info_getter(timeout=self._open_timeout)
                sample_rate, channel_count = self._validate_info(full_info)

            correction = float(inlet.time_correction(timeout=self._time_correction_timeout))
            if not math.isfinite(correction):
                raise RuntimeError("LSL returned a non-finite clock-correction estimate")

            self._resolved_info = full_info
            self._time_correction_seconds = correction
            self._last_correction_refresh_monotonic = time.monotonic()
            self.sampling_rate = sample_rate
            self.channels = channel_count
            source_id = str(self._info_value(full_info, "source_id") or discovered_source_id)
            name = str(self._info_value(full_info, "name") or "unnamed")
            stream_type = str(self._info_value(full_info, "type") or "unknown")
            uid = str(self._info_value(full_info, "uid") or "")
            self.stream_id = self._explicit_stream_id or f"lsl:{source_id or uid or name}"
            self.modality = self._explicit_modality or stream_type.lower()

            self._descriptor = StreamDescriptor(
                stream_id=self.stream_id,
                modality=self.modality,
                sample_rate_hz=sample_rate,
                channel_names=self._extract_channel_names(full_info, channel_count),
                channel_types=tuple(self.modality for _ in range(channel_count)),
                clock_domain=ClockDomain.SYNCHRONIZED,
                device=name,
                metadata={
                    "transport": "lsl",
                    "lsl_name": name,
                    "lsl_type": stream_type,
                    "lsl_source_id": source_id,
                    "lsl_uid": uid,
                    "lsl_hostname": str(self._info_value(full_info, "hostname") or ""),
                    "lsl_session_id": str(self._info_value(full_info, "session_id") or ""),
                    "lsl_channel_format": str(
                        self._info_value(full_info, "channel_format") or ""
                    ),
                    "lsl_recover_requested": self._recover_requested,
                    "lsl_recover_effective": effective_recover,
                    "lsl_postprocessing_flags": 0,
                    "timing_semantics": "raw_lsl_timestamp_plus_time_correction",
                },
            )
            self._sequence_id = 0
            await super().start()
        except Exception:
            self._close_inlet()
            self._reset_resolved_state()
            raise

    async def stop(self) -> None:
        runtime_error: BaseException | None = None
        cleanup_error: BaseException | None = None
        try:
            await super().stop()
        except BaseException as exc:
            runtime_error = exc
        try:
            self._close_inlet()
        except BaseException as exc:
            cleanup_error = exc
        finally:
            self._reset_resolved_state(keep_descriptor=True)

        if runtime_error is not None and cleanup_error is not None:
            raise RuntimeError(
                "LSL acquisition failed and inlet cleanup also failed: "
                f"{cleanup_error!r}"
            ) from runtime_error
        if runtime_error is not None:
            raise runtime_error
        if cleanup_error is not None:
            raise cleanup_error

    def _close_inlet(self) -> None:
        if self._inlet is None:
            return
        inlet, self._inlet = self._inlet, None
        close_stream = getattr(inlet, "close_stream", None)
        if callable(close_stream):
            close_stream()

    def _reset_resolved_state(self, *, keep_descriptor: bool = False) -> None:
        self._resolved_info = None
        self._time_correction_seconds = None
        self._last_correction_refresh_monotonic = 0.0
        if not keep_descriptor:
            self._descriptor = None

    def _refresh_time_correction_if_due(self) -> float:
        if self._inlet is None or self._time_correction_seconds is None:
            raise RuntimeError("LSL inlet is not initialized")
        if self._correction_refresh_seconds > 0:
            now = time.monotonic()
            if now - self._last_correction_refresh_monotonic >= self._correction_refresh_seconds:
                correction = float(
                    self._inlet.time_correction(timeout=self._time_correction_timeout)
                )
                if not math.isfinite(correction):
                    raise RuntimeError("LSL returned a non-finite clock-correction estimate")
                self._time_correction_seconds = correction
                self._last_correction_refresh_monotonic = now
        return self._time_correction_seconds

    async def _frame_stream(self) -> AsyncIterator[SignalFrame]:
        if self._inlet is None:
            raise RuntimeError("LSL inlet is not initialized")

        while self._running:
            correction = self._refresh_time_correction_if_due()
            samples, timestamps = self._inlet.pull_chunk(
                timeout=0.0,
                max_samples=self._max_samples,
            )
            if not samples and not timestamps:
                await asyncio.sleep(self._poll_interval_seconds)
                continue
            if len(samples) != len(timestamps):
                raise RuntimeError(
                    "LSL returned mismatched sample/timestamp chunk lengths: "
                    f"{len(samples)} samples vs {len(timestamps)} timestamps"
                )

            host_receive_time_ns = time.monotonic_ns()
            for sample, raw_timestamp in zip(samples, timestamps):
                timestamp = float(raw_timestamp)
                if not math.isfinite(timestamp) or timestamp <= 0:
                    raise RuntimeError(f"LSL emitted invalid timestamp {timestamp!r}")
                data = np.asarray(sample, dtype=np.float64)
                if data.ndim != 1 or data.shape[0] != self.channels:
                    raise RuntimeError(
                        "LSL sample geometry does not match stream metadata: "
                        f"expected ({self.channels},), received {data.shape}"
                    )
                if not np.isfinite(data).all():
                    raise RuntimeError("LSL emitted NaN or infinite signal values")

                synchronized_timestamp = timestamp + correction
                yield SignalFrame(
                    stream_id=self.stream_id,
                    sequence_id=self._sequence_id,
                    data=data,
                    sample_rate_hz=self.sampling_rate,
                    host_receive_time_ns=host_receive_time_ns,
                    synchronized_time_ns=int(round(synchronized_timestamp * 1_000_000_000)),
                    clock_domain=ClockDomain.SYNCHRONIZED,
                    quality=QualityFlag.GOOD,
                    metadata={
                        "driver": self.__class__.__name__,
                        "lsl_raw_timestamp_seconds": timestamp,
                        "lsl_time_correction_seconds": correction,
                    },
                )
                self._sequence_id += 1
            await asyncio.sleep(0)

    async def _run(self) -> None:
        async for frame in self._frame_stream():
            try:
                await put_with_policy(
                    self._queue,
                    frame,
                    policy=self.overflow_policy,
                    stats=self._queue_stats,
                )
            except asyncio.CancelledError:
                break
            if not self._running:
                break

    async def _next_frame(self) -> SignalFrame:
        if not self._queue.empty():
            frame = self._queue.get_nowait()
            self._queue.task_done()
            return frame
        if self._task is None:
            raise RuntimeError("LSLDriver is not running")
        if self._task.done():
            await self._task
            raise StopAsyncIteration

        queue_get = asyncio.create_task(self._queue.get())
        done, _ = await asyncio.wait(
            {queue_get, self._task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if queue_get in done:
            frame = queue_get.result()
            self._queue.task_done()
            return frame

        queue_get.cancel()
        with suppress(asyncio.CancelledError):
            await queue_get
        await self._task
        raise StopAsyncIteration

    async def frames(self) -> AsyncIterator[SignalFrame]:
        """Yield synchronized canonical frames and surface acquisition failures."""
        while self._running or not self._queue.empty():
            try:
                yield await self._next_frame()
            except StopAsyncIteration:
                break

    async def __aiter__(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        """Yield legacy tuples using the corrected LSL timestamp."""
        async for frame in self.frames():
            yield frame.timestamp_seconds, frame.data

    async def _stream(self) -> AsyncIterator[tuple[float, np.ndarray]]:
        """Compatibility implementation for the BaseDriver private contract."""
        async for frame in self._frame_stream():
            yield frame.timestamp_seconds, frame.data
