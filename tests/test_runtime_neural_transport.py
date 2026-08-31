from __future__ import annotations

import copy
import pickle
from multiprocessing import shared_memory

import numpy as np
import pytest

from neuros.contracts import DecoderOutput, NeuralWindow, SignalFrame, TransformEmission
from neuros.runtime.transport import (
    NeuralTransportCapacityError,
    NeuralTransportError,
    NeuralTransportProtocolError,
    NeuralTransportTypeError,
    SharedMemoryMailbox,
)


def _frame(data: np.ndarray | None = None) -> SignalFrame:
    return SignalFrame(
        stream_id="eeg",
        sequence_id=7,
        data=np.arange(12, dtype=np.float32).reshape(6, 2) if data is None else data,
        sample_rate_hz=250.0,
        host_receive_time_ns=123_000,
        device_time_ns=120_000,
        metadata={
            "axis_order": ("sample", "channel"),
            "channel_names": ("C3", "C4"),
            "nested": {"session": "transport", "trial": 3},
        },
    )


def _window(data: np.ndarray | None = None) -> NeuralWindow:
    return NeuralWindow(
        stream_id="eeg",
        window_id=11,
        data=np.arange(16, dtype=np.float32).reshape(2, 8) if data is None else data,
        sample_rate_hz=250.0,
        start_time_ns=1_000,
        end_time_ns=33_000_000,
        channel_names=("C3", "C4"),
        source_sequence_ids=(6, 7),
        metadata={"nested": {"source": "window"}},
    )


def test_neural_window_detaches_array_and_nested_metadata_from_caller_state():
    source = np.arange(8, dtype=np.float32).reshape(2, 4)
    nested = {"labels": ["left", "right"]}
    window = NeuralWindow(
        stream_id="eeg",
        window_id=1,
        data=source,
        sample_rate_hz=100.0,
        start_time_ns=1,
        end_time_ns=40_000_001,
        channel_names=["C3", "C4"],
        source_sequence_ids=[1, 2],
        metadata=nested,
    )

    source[:] = -1
    nested["labels"].append("mutated")

    assert window.data.tolist() == [[0.0, 1.0, 2.0, 3.0], [4.0, 5.0, 6.0, 7.0]]
    assert not window.data.flags.writeable
    assert window.channel_names == ("C3", "C4")
    assert window.source_sequence_ids == (1, 2)
    assert window.metadata["labels"] == ("left", "right")
    with pytest.raises(ValueError):
        window.data[0, 0] = 99


def test_generic_pickle_rejects_canonical_frame_but_shared_mailbox_round_trips_it():
    frame = _frame()
    with pytest.raises((TypeError, pickle.PicklingError)):
        pickle.dumps(frame, protocol=pickle.HIGHEST_PROTOCOL)

    box = SharedMemoryMailbox(16 * 1024)
    try:
        envelope = box.encode(frame, lease_id=1)
        decoded = box.decode(envelope, expected_lease_id=1)
    finally:
        box.close_and_unlink()

    assert isinstance(decoded, SignalFrame)
    assert decoded.stream_id == frame.stream_id
    assert decoded.sequence_id == frame.sequence_id
    assert decoded.sample_rate_hz == frame.sample_rate_hz
    assert decoded.metadata["nested"]["session"] == "transport"
    assert np.array_equal(decoded.data, frame.data)
    assert not decoded.data.flags.writeable


def test_codec_recursively_packs_multiple_arrays_and_canonical_payloads():
    window = _window()
    output = DecoderOutput(
        prediction=np.array([2], dtype=np.int64),
        confidence=0.8,
        probabilities=np.array([0.1, 0.2, 0.7], dtype=np.float32),
        logits=np.array([-2.0, -1.0, 1.5], dtype=np.float32),
        embedding=np.arange(6, dtype=np.float32).reshape(2, 3),
        model_id="transport-model",
        metadata={"window": 11},
    )
    payload = TransformEmission((window, output, {"aux": np.arange(5, dtype=np.int16)}))

    box = SharedMemoryMailbox(32 * 1024)
    try:
        envelope = box.encode(payload, lease_id=9)
        decoded = box.decode(envelope, expected_lease_id=9)
    finally:
        box.close_and_unlink()

    assert isinstance(decoded, TransformEmission)
    decoded_window, decoded_output, decoded_aux = decoded.items
    assert isinstance(decoded_window, NeuralWindow)
    assert np.array_equal(decoded_window.data, window.data)
    assert isinstance(decoded_output, DecoderOutput)
    assert np.array_equal(decoded_output.probabilities, output.probabilities)
    assert np.array_equal(decoded_output.logits, output.logits)
    assert np.array_equal(decoded_output.embedding, output.embedding)
    assert np.array_equal(decoded_aux["aux"], np.arange(5, dtype=np.int16))
    assert envelope["bytes_used"] >= (
        window.data.nbytes
        + output.prediction.nbytes
        + output.probabilities.nbytes
        + output.logits.nbytes
        + output.embedding.nbytes
    )


def test_capacity_and_unsupported_dtype_fail_closed_without_partial_decode():
    small = SharedMemoryMailbox(64)
    try:
        with pytest.raises(NeuralTransportCapacityError, match="capacity"):
            small.encode(np.arange(100, dtype=np.float64), lease_id=1)
        with pytest.raises(NeuralTransportTypeError, match="boolean or numeric"):
            small.encode(np.array([object()], dtype=object), lease_id=2)
    finally:
        small.close_and_unlink()


def test_stale_lease_and_overlapping_array_descriptors_are_rejected():
    box = SharedMemoryMailbox(4096)
    try:
        envelope = box.encode(
            (np.arange(4, dtype=np.float32), np.arange(4, dtype=np.float32) + 10),
            lease_id=4,
        )
        with pytest.raises(NeuralTransportProtocolError, match="stale"):
            box.decode(envelope, expected_lease_id=5)

        corrupted = copy.deepcopy(envelope)
        first, second = corrupted["manifest"]["items"]
        second["offset"] = first["offset"]
        with pytest.raises(NeuralTransportProtocolError, match="overlap"):
            box.decode(corrupted, expected_lease_id=4)
    finally:
        box.close_and_unlink()


def test_decoded_array_is_independent_of_subsequent_mailbox_reuse():
    box = SharedMemoryMailbox(4096)
    try:
        first = box.decode(
            box.encode(np.arange(8, dtype=np.float32), lease_id=1),
            expected_lease_id=1,
        )
        box.encode(np.full(8, 99, dtype=np.float32), lease_id=2)
        assert np.array_equal(first, np.arange(8, dtype=np.float32))
    finally:
        box.close_and_unlink()


def test_owner_cleanup_unlinks_named_region_and_attached_peer_cannot_claim_authority():
    owner = SharedMemoryMailbox(1024)
    name = owner.name
    peer = SharedMemoryMailbox.attach(name, 1024)
    try:
        with pytest.raises(NeuralTransportError, match="owner"):
            peer.unlink()
        peer.close()
        owner.close_and_unlink()
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)
    finally:
        peer.close()
        owner.close_and_unlink()
