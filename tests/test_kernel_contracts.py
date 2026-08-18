import numpy as np
import pytest

from neuros.contracts import ClockDomain, DecoderOutput, QualityFlag, SignalFrame, StreamDescriptor


def test_stream_descriptor_validates_channel_metadata():
    with pytest.raises(ValueError):
        StreamDescriptor(
            stream_id="eeg",
            modality="eeg",
            sample_rate_hz=250.0,
            channel_names=("C3", "C4"),
            units=("uV",),
        )


def test_signal_frame_prefers_synchronized_timestamp():
    frame = SignalFrame(
        stream_id="eeg",
        sequence_id=4,
        data=np.zeros((2, 8)),
        sample_rate_hz=250.0,
        host_receive_time_ns=100,
        device_time_ns=200,
        synchronized_time_ns=300,
        clock_domain=ClockDomain.SYNCHRONIZED,
        quality=QualityFlag.GOOD,
    )
    assert frame.timestamp_ns == 300


def test_decoder_output_does_not_require_fake_confidence():
    output = DecoderOutput(prediction=1)
    assert output.confidence is None


def test_decoder_output_rejects_invalid_confidence():
    with pytest.raises(ValueError):
        DecoderOutput(prediction=1, confidence=1.2)
