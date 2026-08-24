import asyncio
import builtins
import sys
import types

import numpy as np
import pytest

from neuros.drivers.brainflow_driver import BrainFlowDriver


class FakeBrainFlowInputParams:
    def __init__(self) -> None:
        self.serial_port = ""
        self.ip_address = ""
        self.master_board = 0


class FakeBoardShim:
    instances = []

    def __init__(self, board_id, params) -> None:
        self.board_id = int(board_id)
        self.params = params
        self.prepared = False
        self.streaming = False
        self.released = False
        self._consumed = False
        FakeBoardShim.instances.append(self)

    @staticmethod
    def get_sampling_rate(board_id):
        return 250

    @staticmethod
    def get_eeg_channels(board_id):
        # BrainFlow matrices are board-specific. Deliberately choose rows that
        # are not the first N rows to catch accidental data[:channels] slicing.
        return [1, 3]

    @staticmethod
    def get_timestamp_channel(board_id):
        return 5

    @staticmethod
    def get_device_name(board_id):
        return "Fake EEG Board"

    def get_board_id(self):
        return self.board_id

    def prepare_session(self):
        self.prepared = True

    def start_stream(self):
        assert self.prepared
        self.streaming = True

    def get_board_sampling_rate(self):
        # Exercise the actual prepared-session rate rather than static metadata.
        return 200

    def get_board_data_count(self):
        return 0 if self._consumed else 2

    def get_board_data(self):
        self._consumed = True
        return np.asarray(
            [
                [100.0, 101.0],  # package/other row
                [11.0, 12.0],    # EEG row 1
                [20.0, 21.0],    # non-EEG row
                [31.0, 32.0],    # EEG row 3
                [40.0, 41.0],    # non-EEG row
                [1000.25, 1000.255],  # timestamp row
            ]
        )

    def stop_stream(self):
        self.streaming = False

    def release_session(self):
        self.released = True
        self.prepared = False


def install_fake_brainflow(monkeypatch):
    FakeBoardShim.instances.clear()
    package = types.ModuleType("brainflow")
    board_shim = types.ModuleType("brainflow.board_shim")
    board_shim.BoardShim = FakeBoardShim
    board_shim.BrainFlowInputParams = FakeBrainFlowInputParams
    package.board_shim = board_shim
    monkeypatch.setitem(sys.modules, "brainflow", package)
    monkeypatch.setitem(sys.modules, "brainflow.board_shim", board_shim)


def test_brainflow_missing_dependency_fails_closed(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "brainflow.board_shim" or name.startswith("brainflow"):
            raise ImportError("brainflow intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match=r"neuros-drivers\[eeg\].*MockDriver explicitly"):
        BrainFlowDriver()


def test_brainflow_rejects_unknown_hardware_parameters(monkeypatch):
    install_fake_brainflow(monkeypatch)

    with pytest.raises(ValueError, match="Unknown BrainFlowInputParams field.*seral_port"):
        BrainFlowDriver(seral_port="/dev/ttyUSB0")


@pytest.mark.asyncio
async def test_brainflow_uses_declared_eeg_rows_and_drains_samples_once(monkeypatch):
    install_fake_brainflow(monkeypatch)
    driver = BrainFlowDriver(board_id=7, serial_port="/dev/fake")

    # Static metadata is available before the session is prepared.
    assert driver.sampling_rate == 250
    assert driver.channels == 2
    assert driver.descriptor.metadata["brainflow_eeg_rows"] == (1, 3)

    await driver.start()
    assert driver.sampling_rate == 200

    received = []
    async for timestamp, data in driver:
        received.append((timestamp, data))
        if len(received) == 2:
            break

    await driver.stop()

    assert received == [
        (1000.25, [11.0, 31.0]),
        (1000.255, [12.0, 32.0]),
    ]
    board = FakeBoardShim.instances[-1]
    assert board._consumed is True
    assert board.streaming is False
    assert board.released is True


@pytest.mark.asyncio
async def test_brainflow_sampling_rate_is_an_assertion_not_fake_resampling(monkeypatch):
    install_fake_brainflow(monkeypatch)
    driver = BrainFlowDriver(board_id=7, sampling_rate=250)

    with pytest.raises(ValueError, match="controls the device sampling rate.*200 Hz"):
        await driver.start()

    board = FakeBoardShim.instances[-1]
    assert board.streaming is False
    assert board.released is True
    assert driver._running is False


def test_brainflow_channel_count_cannot_exceed_declared_eeg_rows(monkeypatch):
    install_fake_brainflow(monkeypatch)

    with pytest.raises(ValueError, match="Requested 3 EEG channels.*exposes 2"):
        BrainFlowDriver(board_id=7, channels=3)


@pytest.mark.asyncio
async def test_brainflow_acquisition_failure_still_releases_session(monkeypatch):
    install_fake_brainflow(monkeypatch)
    driver = BrainFlowDriver(board_id=7)
    board = FakeBoardShim.instances[-1]

    def malformed_payload():
        board._consumed = True
        # Metadata requires rows 1, 3, and 5, so this payload must fail.
        return np.zeros((2, 1), dtype=float)

    board.get_board_data = malformed_payload

    await driver.start()
    await asyncio.sleep(0.02)

    with pytest.raises(RuntimeError, match="payload row count does not match"):
        await driver.stop()

    assert board.streaming is False
    assert board.released is True
