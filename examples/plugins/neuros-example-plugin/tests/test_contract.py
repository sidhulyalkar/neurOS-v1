from __future__ import annotations

import numpy as np
import pytest

from neuros.contracts import ClockDomain, SignalFrame, Source, Transform
from neuros_example_plugin import GainTransform, SineSource


@pytest.mark.asyncio
async def test_sine_source_implements_public_source_contract() -> None:
    source = SineSource(
        sampling_rate=200.0,
        channels=3,
        frequency_hz=10.0,
        amplitude_uv=12.0,
        chunk_samples=20,
        stream_id="external/eeg",
    )
    assert isinstance(source, Source)
    assert source.descriptor.stream_id == "external/eeg"
    assert source.descriptor.channel_names == ("EX1", "EX2", "EX3")
    assert source.descriptor.units == ("uV", "uV", "uV")
    assert source.descriptor.clock_domain is ClockDomain.HOST_MONOTONIC

    await source.start()
    frame = await source.frames().__anext__()
    await source.stop()

    assert isinstance(frame, SignalFrame)
    assert frame.data.shape == (20, 3)
    assert frame.sample_rate_hz == 200.0
    assert frame.sequence_id == 0
    assert frame.clock_domain is ClockDomain.HOST_MONOTONIC
    assert frame.metadata["axis_order"] == ("sample", "channel")
    assert frame.device_time_ns is None
    assert frame.synchronized_time_ns is None


@pytest.mark.asyncio
async def test_sine_samples_are_deterministic_for_same_configuration() -> None:
    first = SineSource(chunk_samples=8, channels=2)
    second = SineSource(chunk_samples=8, channels=2)
    await first.start()
    await second.start()
    first_frame = await first.frames().__anext__()
    second_frame = await second.frames().__anext__()
    await first.stop()
    await second.stop()
    np.testing.assert_allclose(first_frame.data, second_frame.data)


def test_gain_transform_preserves_frame_identity_and_adds_provenance() -> None:
    transform = GainTransform(gain=0.25)
    assert isinstance(transform, Transform)
    frame = SignalFrame(
        stream_id="external/eeg",
        sequence_id=7,
        data=np.asarray([[4.0, -8.0]]),
        sample_rate_hz=250.0,
        host_receive_time_ns=123,
        clock_domain=ClockDomain.HOST_MONOTONIC,
        metadata={"axis_order": ("sample", "channel"), "upstream": "fixture"},
    )

    result = transform.transform(frame)

    assert isinstance(result, SignalFrame)
    assert result.stream_id == frame.stream_id
    assert result.sequence_id == frame.sequence_id
    assert result.host_receive_time_ns == frame.host_receive_time_ns
    assert result.metadata["upstream"] == "fixture"
    assert result.metadata["example_gain"] == 0.25
    np.testing.assert_allclose(result.data, [[1.0, -2.0]])


def test_example_configuration_rejects_invalid_options_early() -> None:
    with pytest.raises(ValueError, match="Nyquist"):
        SineSource(sampling_rate=100.0, frequency_hz=50.0)
    with pytest.raises(ValueError, match="finite"):
        GainTransform(gain=float("nan"))
