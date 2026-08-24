"""Fail-closed BrainFlow EEG source for neurOS.

BrainFlow's returned matrix is board-specific. This driver therefore uses
BrainFlow's channel metadata instead of assuming EEG occupies the first rows,
drains the board ringbuffer so samples are emitted once, and never silently
substitutes synthetic data when hardware support is unavailable.
"""

from __future__ import annotations

import asyncio
import math
from typing import AsyncIterator, Optional

import numpy as np

from neuros.contracts import ClockDomain, StreamDescriptor
from neuros.drivers.base_driver import BaseDriver
from neuros.runtime import OverflowPolicy


class BrainFlowDriver(BaseDriver):
    """Stream EEG samples from a BrainFlow-supported board.

    Parameters
    ----------
    board_id:
        BrainFlow board identifier.
    sampling_rate:
        Optional *expected* prepared-session sampling rate. BrainFlow controls
        the hardware sampling rate; neurOS does not pretend this argument can
        resample the device. If supplied, startup fails when the actual rate
        differs.
    channels:
        Optional number of EEG channels to expose, selected from BrainFlow's
        declared EEG row indices. It must not exceed the board's EEG channel
        count.
    stream_id:
        neurOS stream identifier.
    overflow_policy:
        Queue overload behavior inherited from :class:`BaseDriver`.
    **params:
        BrainFlow ``BrainFlowInputParams`` fields such as ``serial_port``,
        ``ip_address``, or ``master_board``. Unknown fields are rejected so a
        misspelled hardware parameter cannot be ignored silently.

    Notes
    -----
    BrainFlow is an optional dependency. Constructing this driver without it
    installed raises an actionable :class:`ImportError`. Use ``MockDriver``
    explicitly for synthetic data.
    """

    def __init__(
        self,
        board_id: int = 0,
        sampling_rate: Optional[float] = None,
        channels: Optional[int] = None,
        *,
        stream_id: str | None = None,
        overflow_policy: OverflowPolicy = OverflowPolicy.DROP_OLDEST,
        **params,
    ) -> None:
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "BrainFlowDriver requires the optional BrainFlow dependency. "
                "Install `neuros-drivers[eeg]` or use MockDriver explicitly "
                "for synthetic data."
            ) from exc

        if channels is not None and int(channels) <= 0:
            raise ValueError("channels must be positive when provided")
        if sampling_rate is not None and float(sampling_rate) <= 0:
            raise ValueError("sampling_rate must be positive when provided")

        input_params = BrainFlowInputParams()
        unknown_params: list[str] = []
        for key, value in params.items():
            if not hasattr(input_params, key):
                unknown_params.append(key)
                continue
            setattr(input_params, key, value)
        if unknown_params:
            joined = ", ".join(sorted(unknown_params))
            raise ValueError(f"Unknown BrainFlowInputParams field(s): {joined}")

        self._BoardShim = BoardShim
        self._board_id = int(board_id)
        self._params = input_params
        self._board = BoardShim(self._board_id, input_params)
        self._requested_channels = int(channels) if channels is not None else None
        self._expected_sampling_rate = (
            float(sampling_rate) if sampling_rate is not None else None
        )
        self._session_prepared = False
        self._board_streaming = False

        self._master_board_id = self._resolve_master_board_id()
        self._eeg_channels = self._resolve_eeg_channels(self._master_board_id)
        initial_rate = float(BoardShim.get_sampling_rate(self._master_board_id))
        self._timestamp_channel = self._resolve_timestamp_channel(self._master_board_id)
        self._device_name = self._resolve_device_name(self._master_board_id)

        super().__init__(
            sampling_rate=initial_rate,
            channels=len(self._eeg_channels),
            stream_id=stream_id,
            modality="eeg",
            overflow_policy=overflow_policy,
        )

    def _resolve_master_board_id(self) -> int:
        get_board_id = getattr(self._board, "get_board_id", None)
        if callable(get_board_id):
            return int(get_board_id())
        return self._board_id

    def _resolve_eeg_channels(self, board_id: int) -> tuple[int, ...]:
        rows = tuple(int(row) for row in self._BoardShim.get_eeg_channels(board_id))
        if not rows:
            raise ValueError(f"BrainFlow board {board_id} exposes no EEG channels")
        if self._requested_channels is not None:
            if self._requested_channels > len(rows):
                raise ValueError(
                    f"Requested {self._requested_channels} EEG channels but BrainFlow "
                    f"board {board_id} exposes {len(rows)}"
                )
            rows = rows[: self._requested_channels]
        return rows

    def _resolve_timestamp_channel(self, board_id: int) -> int | None:
        try:
            return int(self._BoardShim.get_timestamp_channel(board_id))
        except Exception:
            # Some BrainFlow-compatible streams do not expose a timestamp row.
            # This is optional metadata; EEG row discovery is not optional.
            return None

    def _resolve_device_name(self, board_id: int) -> str:
        getter = getattr(self._BoardShim, "get_device_name", None)
        if callable(getter):
            try:
                return str(getter(board_id))
            except Exception:
                pass
        return f"BrainFlow board {board_id}"

    @property
    def descriptor(self) -> StreamDescriptor:
        """Describe the actual BrainFlow EEG rows exposed by this source."""
        return StreamDescriptor(
            stream_id=self.stream_id,
            modality="eeg",
            sample_rate_hz=self.sampling_rate,
            channel_names=tuple(f"eeg_{index}" for index in range(self.channels)),
            channel_types=tuple("eeg" for _ in range(self.channels)),
            device=self._device_name,
            manufacturer="BrainFlow",
            clock_domain=ClockDomain.UNKNOWN,
            metadata={
                "brainflow_board_id": self._board_id,
                "brainflow_master_board_id": self._master_board_id,
                "brainflow_eeg_rows": self._eeg_channels,
                "brainflow_timestamp_row": self._timestamp_channel,
            },
        )

    async def start(self) -> None:
        """Prepare the hardware session, validate it, and start acquisition."""
        if self._running:
            return

        try:
            self._board.prepare_session()
            self._session_prepared = True

            # Playback/streaming boards can resolve to a different master board.
            self._master_board_id = self._resolve_master_board_id()
            self._eeg_channels = self._resolve_eeg_channels(self._master_board_id)
            self.channels = len(self._eeg_channels)
            self._timestamp_channel = self._resolve_timestamp_channel(self._master_board_id)
            self._device_name = self._resolve_device_name(self._master_board_id)

            actual_rate_getter = getattr(self._board, "get_board_sampling_rate", None)
            if callable(actual_rate_getter):
                actual_rate = float(actual_rate_getter())
            else:
                actual_rate = float(self._BoardShim.get_sampling_rate(self._master_board_id))
            if actual_rate <= 0:
                raise RuntimeError(
                    f"BrainFlow reported invalid prepared-session sampling rate {actual_rate}"
                )
            if self._expected_sampling_rate is not None and not math.isclose(
                actual_rate,
                self._expected_sampling_rate,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "BrainFlow controls the device sampling rate: expected "
                    f"{self._expected_sampling_rate:g} Hz but the prepared session "
                    f"reports {actual_rate:g} Hz"
                )
            self.sampling_rate = actual_rate

            self._board.start_stream()
            self._board_streaming = True
            await super().start()
        except Exception:
            await self._cleanup_board(raise_errors=False)
            raise

    async def stop(self) -> None:
        """Stop neurOS streaming and release the BrainFlow session.

        Hardware cleanup is attempted even if the background acquisition task
        has already failed. The original acquisition failure remains the
        primary exception unless cleanup also fails.
        """
        runtime_error: BaseException | None = None
        cleanup_error: BaseException | None = None

        try:
            await super().stop()
        except BaseException as exc:
            runtime_error = exc

        try:
            await self._cleanup_board(raise_errors=True)
        except BaseException as exc:
            cleanup_error = exc

        if runtime_error is not None and cleanup_error is not None:
            raise RuntimeError(
                "BrainFlow acquisition failed and hardware cleanup also failed: "
                f"{cleanup_error!r}"
            ) from runtime_error
        if runtime_error is not None:
            raise runtime_error
        if cleanup_error is not None:
            raise cleanup_error

    async def _cleanup_board(self, *, raise_errors: bool) -> None:
        errors: list[BaseException] = []

        if self._board_streaming:
            try:
                self._board.stop_stream()
            except Exception as exc:
                errors.append(exc)
            finally:
                self._board_streaming = False

        if self._session_prepared:
            try:
                self._board.release_session()
            except Exception as exc:
                errors.append(exc)
            finally:
                self._session_prepared = False

        if errors and raise_errors:
            raise RuntimeError(
                "BrainFlow cleanup failed; the hardware session may require manual recovery"
            ) from errors[0]

    async def _stream(self) -> AsyncIterator[tuple[float, list[float]]]:
        """Drain buffered BrainFlow samples exactly once using declared EEG rows."""
        idle_sleep = min(max(1.0 / self.sampling_rate, 0.001), 0.02)

        while self._running:
            try:
                available = int(self._board.get_board_data_count())
                if available <= 0:
                    await asyncio.sleep(idle_sleep)
                    continue

                # get_board_data removes returned samples from BrainFlow's ringbuffer.
                # Using get_current_board_data here would repeatedly emit the same
                # latest sample when neurOS polls faster than the device.
                matrix = np.asarray(self._board.get_board_data(), dtype=np.float64)
                if matrix.ndim != 2:
                    raise RuntimeError(
                        f"BrainFlow returned a non-matrix payload with shape {matrix.shape}"
                    )
                if matrix.shape[1] == 0:
                    await asyncio.sleep(idle_sleep)
                    continue

                required_rows = list(self._eeg_channels)
                if self._timestamp_channel is not None:
                    required_rows.append(self._timestamp_channel)
                max_required_row = max(required_rows)
                if max_required_row >= matrix.shape[0]:
                    raise RuntimeError(
                        "BrainFlow payload row count does not match the board metadata: "
                        f"required row {max_required_row}, received {matrix.shape[0]} rows"
                    )

                for column in range(matrix.shape[1]):
                    if self._timestamp_channel is not None:
                        timestamp = float(matrix[self._timestamp_channel, column])
                        if not math.isfinite(timestamp):
                            raise RuntimeError("BrainFlow emitted a non-finite timestamp")
                    else:
                        # A host wall-clock fallback is explicit only when the board
                        # exposes no timestamp channel. Canonical host receipt timing is
                        # still recorded by SignalFrame.from_legacy/BaseDriver.frames().
                        import time

                        timestamp = time.time()

                    sample = matrix[np.asarray(self._eeg_channels, dtype=int), column]
                    if not np.isfinite(sample).all():
                        raise RuntimeError("BrainFlow emitted NaN or infinite EEG values")
                    yield timestamp, sample.tolist()

                await asyncio.sleep(0)
            except asyncio.CancelledError:
                break
