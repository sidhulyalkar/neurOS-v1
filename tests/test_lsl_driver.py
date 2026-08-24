import builtins
import sys
import types

import pytest

from neuros.contracts import ClockDomain
from neuros.drivers.lsl_driver import LSLDriver


class FakeXMLNode:
    def __init__(self, labels=None, index=0, empty=False):
        self.labels = list(labels or [])
        self.index = index
        self._empty = empty

    def child(self, name):
        if name == "channels":
            return self
        if name == "channel":
            if not self.labels:
                return FakeXMLNode(empty=True)
            return FakeXMLNode(self.labels, 0)
        return FakeXMLNode(empty=True)

    def child_value(self, name):
        if name == "label" and self.index < len(self.labels):
            return self.labels[self.index]
        return ""

    def next_sibling(self):
        next_index = self.index + 1
        if next_index >= len(self.labels):
            return FakeXMLNode(empty=True)
        return FakeXMLNode(self.labels, next_index)

    def empty(self):
        return self._empty


class FakeInfo:
    def __init__(
        self,
        *,
        name="Fake EEG",
        stream_type="EEG",
        source_id="fake-source",
        uid="fake-uid",
        hostname="fake-host",
        session_id="fake-session",
        channel_format="float32",
        channel_count=2,
        nominal_srate=250.0,
        channel_labels=("C3", "C4"),
    ) -> None:
        self._name = name
        self._type = stream_type
        self._source_id = source_id
        self._uid = uid
        self._hostname = hostname
        self._session_id = session_id
        self._channel_format = channel_format
        self._channel_count = channel_count
        self._nominal_srate = nominal_srate
        self._channel_labels = tuple(channel_labels)

    def name(self):
        return self._name

    def type(self):
        return self._type

    def source_id(self):
        return self._source_id

    def uid(self):
        return self._uid

    def hostname(self):
        return self._hostname

    def session_id(self):
        return self._session_id

    def channel_format(self):
        return self._channel_format

    def channel_count(self):
        return self._channel_count

    def nominal_srate(self):
        return self._nominal_srate

    def desc(self):
        return FakeXMLNode(self._channel_labels)


class FakeLSLRuntime:
    streams = []
    chunks = []
    correction = 0.125
    resolve_calls = []
    inlet_instances = []

    @classmethod
    def reset(cls, *, streams=None, chunks=None, correction=0.125):
        cls.streams = list(streams or [])
        cls.chunks = list(chunks or [])
        cls.correction = correction
        cls.resolve_calls = []
        cls.inlet_instances = []


class FakeStreamInlet:
    def __init__(
        self,
        info,
        *,
        max_buflen,
        max_chunklen,
        recover,
        processing_flags,
    ) -> None:
        self._info = info
        self.max_buflen = max_buflen
        self.max_chunklen = max_chunklen
        self.recover = recover
        self.processing_flags = processing_flags
        self.opened = False
        self.closed = False
        self.time_correction_calls = 0
        self.chunks = list(FakeLSLRuntime.chunks)
        FakeLSLRuntime.inlet_instances.append(self)

    def open_stream(self, timeout):
        assert timeout > 0
        self.opened = True

    def info(self, timeout):
        assert timeout > 0
        return self._info

    def time_correction(self, timeout):
        assert timeout > 0
        self.time_correction_calls += 1
        return FakeLSLRuntime.correction

    def pull_chunk(self, *, timeout, max_samples):
        assert timeout == 0.0
        assert max_samples > 0
        if self.chunks:
            return self.chunks.pop(0)
        return [], []

    def close_stream(self):
        self.closed = True


def install_fake_pylsl(monkeypatch, *, streams=None, chunks=None, correction=0.125):
    FakeLSLRuntime.reset(streams=streams, chunks=chunks, correction=correction)
    module = types.ModuleType("pylsl")

    def resolve_byprop(prop, value, minimum=1, timeout=1.0):
        FakeLSLRuntime.resolve_calls.append((prop, value, minimum, timeout))
        return list(FakeLSLRuntime.streams)

    module.resolve_byprop = resolve_byprop
    module.StreamInlet = FakeStreamInlet
    monkeypatch.setitem(sys.modules, "pylsl", module)


def test_lsl_missing_dependency_fails_closed(monkeypatch):
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "pylsl" or name.startswith("pylsl."):
            raise ImportError("pylsl intentionally unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)

    with pytest.raises(ImportError, match=r"neuros-drivers\[eeg\].*liblsl"):
        LSLDriver(source_id="missing")


def test_lsl_requires_deterministic_selector(monkeypatch):
    install_fake_pylsl(monkeypatch)

    with pytest.raises(ValueError, match="requires source_id, name, or stream_type"):
        LSLDriver()


@pytest.mark.asyncio
async def test_lsl_rejects_ambiguous_streams(monkeypatch):
    install_fake_pylsl(
        monkeypatch,
        streams=[
            FakeInfo(name="EEG A", source_id="a"),
            FakeInfo(name="EEG B", source_id="b"),
        ],
    )
    driver = LSLDriver(stream_type="EEG")

    with pytest.raises(RuntimeError, match="selection is ambiguous"):
        await driver.start()

    assert FakeLSLRuntime.resolve_calls == [("type", "EEG", 2, 2.0)]
    assert FakeLSLRuntime.inlet_instances == []
    assert driver._running is False


@pytest.mark.asyncio
async def test_lsl_emits_explicitly_synchronized_canonical_frames(monkeypatch):
    info = FakeInfo(source_id="headset-1", channel_labels=("C3", "C4"))
    install_fake_pylsl(
        monkeypatch,
        streams=[info],
        chunks=[([[1.0, 2.0], [3.0, 4.0]], [100.0, 100.004])],
        correction=0.125,
    )
    driver = LSLDriver(
        source_id="headset-1",
        stream_type="EEG",
        sampling_rate=250,
        channels=2,
        correction_refresh_seconds=0,
    )

    await driver.start()
    descriptor = driver.descriptor
    assert descriptor.clock_domain is ClockDomain.SYNCHRONIZED
    assert descriptor.channel_names == ("C3", "C4")
    assert descriptor.manufacturer is None
    assert descriptor.metadata["transport"] == "lsl"
    assert descriptor.metadata["lsl_source_id"] == "headset-1"
    assert descriptor.metadata["lsl_postprocessing_flags"] == 0
    assert descriptor.metadata["timing_semantics"] == (
        "raw_lsl_timestamp_plus_time_correction"
    )
    assert FakeLSLRuntime.resolve_calls == [("source_id", "headset-1", 2, 2.0)]

    frames = []
    async for frame in driver.frames():
        frames.append(frame)
        if len(frames) == 2:
            break

    await driver.stop()

    assert [frame.sequence_id for frame in frames] == [0, 1]
    assert [frame.data.tolist() for frame in frames] == [[1.0, 2.0], [3.0, 4.0]]
    assert frames[0].clock_domain is ClockDomain.SYNCHRONIZED
    assert frames[0].synchronized_time_ns == 100_125_000_000
    assert frames[1].synchronized_time_ns == 100_129_000_000
    assert frames[0].metadata["lsl_raw_timestamp_seconds"] == 100.0
    assert frames[0].metadata["lsl_time_correction_seconds"] == 0.125

    inlet = FakeLSLRuntime.inlet_instances[-1]
    assert inlet.opened is True
    assert inlet.closed is True
    assert inlet.recover is True
    assert inlet.processing_flags == 0
    assert inlet.time_correction_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("driver_kwargs", "message"),
    [
        ({"sampling_rate": 200}, "sampling rate mismatch"),
        ({"channels": 8}, "channel-count mismatch"),
    ],
)
async def test_lsl_expected_geometry_is_asserted(monkeypatch, driver_kwargs, message):
    install_fake_pylsl(monkeypatch, streams=[FakeInfo(source_id="headset-1")])
    driver = LSLDriver(source_id="headset-1", **driver_kwargs)

    with pytest.raises(ValueError, match=message):
        await driver.start()

    assert FakeLSLRuntime.inlet_instances == []
    assert driver._running is False


@pytest.mark.asyncio
async def test_lsl_recovery_is_disabled_without_source_id(monkeypatch):
    install_fake_pylsl(
        monkeypatch,
        streams=[FakeInfo(name="NoSourceId", source_id="")],
        chunks=[([[1.0, 2.0]], [10.0])],
    )
    driver = LSLDriver(name="NoSourceId", recover=True, correction_refresh_seconds=0)

    await driver.start()
    inlet = FakeLSLRuntime.inlet_instances[-1]
    assert inlet.recover is False
    assert driver.descriptor.metadata["lsl_recover_requested"] is True
    assert driver.descriptor.metadata["lsl_recover_effective"] is False
    await driver.stop()


@pytest.mark.asyncio
async def test_lsl_rejects_irregular_streams(monkeypatch):
    install_fake_pylsl(
        monkeypatch,
        streams=[FakeInfo(source_id="markers", stream_type="Markers", nominal_srate=0.0)],
    )
    driver = LSLDriver(source_id="markers")

    with pytest.raises(ValueError, match="continuous regular-rate streams only"):
        await driver.start()

    assert FakeLSLRuntime.inlet_instances == []


@pytest.mark.asyncio
async def test_lsl_acquisition_failure_surfaces_to_frame_consumer_and_closes(monkeypatch):
    install_fake_pylsl(
        monkeypatch,
        streams=[FakeInfo(source_id="headset-1")],
        chunks=[([[1.0, 2.0]], [10.0, 10.004])],
    )
    driver = LSLDriver(source_id="headset-1", correction_refresh_seconds=0)
    await driver.start()

    with pytest.raises(RuntimeError, match="mismatched sample/timestamp"):
        await anext(driver.frames())

    with pytest.raises(RuntimeError, match="mismatched sample/timestamp"):
        await driver.stop()

    assert FakeLSLRuntime.inlet_instances[-1].closed is True
