"""
BrainFlow driver for neurOS.

This module provides integration with the BrainFlow library for acquiring
data from a wide variety of consumer and research-grade biosignal devices.
If the optional ``brainflow`` package is not installed, the driver will
gracefully fall back to the :class:`MockDriver`.  Users can specify a
board ID and optional parameters such as serial port or IP address.  In
production, this driver enables neurOS to support dozens of devices
uniformly.
"""

from __future__ import annotations

import asyncio
from typing import AsyncIterator, Optional

from neuros.drivers.base_driver import BaseDriver
from neuros.drivers.mock_driver import MockDriver


class BrainFlowDriver(BaseDriver):
    """Driver that uses BrainFlow to stream neural data.

    This driver wraps the optional BrainFlow library.  When BrainFlow is
    available, it will stream data from the specified board.  If
    BrainFlow cannot be imported, a :class:`MockDriver` is used as a
    fallback so that pipelines can still run without hardware.  The
    driver exposes the standard :class:`BaseDriver` interface by
    implementing the `_stream` coroutine which is invoked by the
    base class to produce timestamped samples.
    """

    def __init__(
        self,
        board_id: int = 0,
        sampling_rate: Optional[float] = None,
        channels: Optional[int] = None,
        **params,
    ) -> None:
        """Initialise a BrainFlow driver.

        Parameters
        ----------
        board_id : int, optional
            Identifier of the BrainFlow board.  Defaults to 0 (synthetic).
        sampling_rate : float, optional
            Desired sampling rate.  If omitted, the board's native rate is
            used.
        channels : int, optional
            Number of channels to acquire.  If omitted, the number of EEG
            channels reported by the board is used.
        **params
            Additional keyword arguments are passed through to the
            BrainFlow ``BrainFlowInputParams`` object.  These allow
            specifying serial port, IP address, etc.  Unknown keys are
            ignored.
        """
        # Attempt to import BrainFlow.  If it fails, fall back to the mock
        # driver.  We defer importing BrainFlow until runtime so that
        # neurOS does not require it as a hard dependency.
        try:
            from brainflow.board_shim import BoardShim, BrainFlowInputParams  # type: ignore
        except ImportError:
            # no brainflow: set up a mock driver and inherit its config
            mock = MockDriver(
                sampling_rate=sampling_rate or 250.0,
                channels=channels or 8,
            )
            # call BaseDriver constructor to initialise queues and state
            super().__init__(sampling_rate=mock.sampling_rate, channels=mock.channels)
            self._delegate: Optional[BaseDriver] = mock
            self._board: Optional[object] = None
            return

        # brainflow available: configure the board
        # instantiate input parameters and assign any provided values
        input_params = BrainFlowInputParams()
        for k, v in params.items():
            if hasattr(input_params, k):
                setattr(input_params, k, v)
        # determine channel count and sampling rate from board if not given
        # BrainFlow provides functions to query these properties
        board_fs = BoardShim.get_sampling_rate(board_id)
        eeg_chans = BoardShim.get_eeg_channels(board_id)
        n_ch = channels or len(eeg_chans)
        fs = sampling_rate or board_fs
        # initialise base driver
        super().__init__(sampling_rate=fs, channels=n_ch)
        # store BrainFlow board
        self._board_id = board_id
        self._params = input_params
        self._board = BoardShim(board_id, input_params)
        self._delegate = None

    async def start(self) -> None:
        """Start streaming.

        If BrainFlow is available, this prepares and starts the board.
        Otherwise it delegates to the mock driver.  In all cases
        ``BaseDriver.start`` is called to spawn the streaming task.
        """
        if self._delegate is not None:
            # use mock driver
            await self._delegate.start()
            return
        # prepare and start the BrainFlow board
        # BoardShim.prepare_session and start_stream are synchronous
        self._board.prepare_session()  # type: ignore[attr-defined]
        self._board.start_stream()  # type: ignore[attr-defined]
        await super().start()

    async def stop(self) -> None:
        """Stop streaming and clean up resources."""
        if self._delegate is not None:
            await self._delegate.stop()
            return
        # call BaseDriver.stop to cancel the stream task and flush the queue
        await super().stop()
        # stop the board and release resources
        try:
            self._board.stop_stream()  # type: ignore[attr-defined]
            self._board.release_session()  # type: ignore[attr-defined]
        except Exception:
            pass

    async def _stream(self) -> AsyncIterator[tuple[float, list[float]]]:
        """Internal coroutine that yields timestamped samples.

        This implementation delegates to the mock driver when BrainFlow
        is unavailable.  Otherwise it repeatedly queries the board for
        the most recent sample and yields it along with a timestamp.
        The loop runs until the driver is stopped.
        """
        # if fallback driver is present, delegate streaming
        if self._delegate is not None:
            async for ts, data in self._delegate:
                yield ts, data
            return
        import time
        # compute sleep interval based on sampling rate to reduce busy looping
        period = 1.0 / (self.sampling_rate or 1.0)
        while True:
            try:
                # BrainFlow returns an array shape (n_channels_total, n_samples)
                data = self._board.get_current_board_data(1)  # type: ignore[attr-defined]
                if data.size != 0:
                    # extract the latest sample for the configured channels
                    ts = time.time()
                    # data[:n_channels, -1] gives the last sample for each channel
                    sample = data[: self.channels, -1]
                    # convert to Python list for consistency with MockDriver
                    yield ts, sample.tolist()
                await asyncio.sleep(period)
            except asyncio.CancelledError:
                break