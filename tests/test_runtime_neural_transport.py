from __future__ import annotations

import asyncio
import copy
import os
import pickle
import time
from multiprocessing import shared_memory

import numpy as np
import pytest

from neuros.contracts import DecoderOutput, NeuralWindow, SignalFrame, TransformEmission
from neuros.runtime.process_worker import (
    ProcessWorkerTerminationError,
    ProcessWorkerTimeoutError,
    ProcessWorkerTransportError,
)
from neuros.runtime.shared_process_worker import SharedMemoryProcessWorker
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


def _assert_names_exist(names: dict[str, str]) -> None:
    for name in names.values():
        probe = shared_memory.SharedMemory(name=name, create=False)
        probe.close()


def _assert_names_absent(names: dict[str, str]) -> None:
    for name in names.values():
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)


class IdentityTransform:
    def transform(self, item):
        return item


class RetainingTransform:
    def __init__(self):
        self.first = None

    def transform(self, item):
        if self.first is None:
            self.first = item
            return item
        return self.first


class DecoderOutputTransform:
    def transform(self, item):
        value = float(np.asarray(item).reshape(-1)[0])
        return DecoderOutput(
            prediction=np.array([int(value)], dtype=np.int64),
            probabilities=np.array([0.25, 0.75], dtype=np.float32),
            logits=np.array([-0.5, 0.5], dtype=np.float32),
            embedding=np.arange(6, dtype=np.float32).reshape(2, 3),
            model_id="shared-worker",
            metadata={"value": value},
        )


class LargeResultTransform:
    def transform(self, item):
        return np.arange(1024, dtype=np.float64)


class SleepTransform:
    def transform(self, item):
        time.sleep(2.0)
        return item


class CrashTransform:
    def transform(self, item):
        os._exit(23)


class ImmortalProcess:
    def __init__(self):
        self.terminate_calls = 0
        self.kill_calls = 0
        self.join_calls = 0

    def is_alive(self):
        return True

    def terminate(self):
        self.terminate_calls += 1

    def kill(self):
        self.kill_calls += 1

    def join(self, timeout=None):
        self.join_calls += 1

    def close(self):
        raise AssertionError("live process handle must not be closed")


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
        with pytest.raises(NeuralTransportTypeError, match="keys must be strings"):
            small.encode({"valid": 1, 2: "invalid"}, lease_id=3)
    finally:
        small.close_and_unlink()


def test_transport_preserves_generic_numeric_values_without_inventing_policy():
    payload = {
        "array": np.array([np.nan, np.inf, -np.inf], dtype=np.float64),
        "float": float("nan"),
        "complex": complex(1.5, -2.25),
        "complex_array": np.array([1 + 2j, np.nan + 1j], dtype=np.complex128),
    }
    box = SharedMemoryMailbox(4096)
    try:
        decoded = box.decode(box.encode(payload, lease_id=8), expected_lease_id=8)
    finally:
        box.close_and_unlink()

    assert np.isnan(decoded["array"][0])
    assert np.isposinf(decoded["array"][1])
    assert np.isneginf(decoded["array"][2])
    assert np.isnan(decoded["float"])
    assert decoded["complex"] == complex(1.5, -2.25)
    assert decoded["complex_array"][0] == 1 + 2j
    assert np.isnan(decoded["complex_array"][1].real)


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


@pytest.mark.asyncio
async def test_shared_worker_round_trips_canonical_frame_and_unlinks_after_close():
    worker = SharedMemoryProcessWorker(
        "transform",
        IdentityTransform(),
        execution_timeout_s=2.0,
        request_capacity_bytes=16 * 1024,
        response_capacity_bytes=16 * 1024,
    )
    call = await worker.invoke("transform", _frame())
    names = worker.shared_memory_names
    assert names is not None
    _assert_names_exist(names)

    assert isinstance(call.result, SignalFrame)
    assert call.result.metadata["nested"]["trial"] == 3
    assert np.array_equal(call.result.data, _frame().data)
    assert call.receipt.outcome == "success"

    worker.close()
    assert not worker.is_alive
    assert worker.shared_memory_names is None
    _assert_names_absent(names)


@pytest.mark.asyncio
async def test_shared_worker_retained_input_survives_mailbox_reuse():
    worker = SharedMemoryProcessWorker(
        "transform",
        RetainingTransform(),
        execution_timeout_s=2.0,
        request_capacity_bytes=4096,
        response_capacity_bytes=4096,
    )
    try:
        first_value = np.arange(8, dtype=np.float32)
        first = await worker.invoke("transform", first_value)
        second = await worker.invoke(
            "transform", np.full(8, 99, dtype=np.float32)
        )
        assert np.array_equal(first.result, first_value)
        assert np.array_equal(second.result, first_value)
    finally:
        worker.close()


@pytest.mark.asyncio
async def test_shared_worker_returns_decoder_output_with_multiple_arrays():
    worker = SharedMemoryProcessWorker(
        "transform",
        DecoderOutputTransform(),
        execution_timeout_s=2.0,
        request_capacity_bytes=4096,
        response_capacity_bytes=16 * 1024,
    )
    try:
        call = await worker.invoke("transform", np.array([4], dtype=np.float32))
        output = call.result
        assert isinstance(output, DecoderOutput)
        assert np.array_equal(output.prediction, np.array([4], dtype=np.int64))
        assert np.array_equal(output.probabilities, np.array([0.25, 0.75], dtype=np.float32))
        assert np.array_equal(output.embedding, np.arange(6, dtype=np.float32).reshape(2, 3))
        assert output.metadata["value"] == 4.0
    finally:
        worker.close()


@pytest.mark.asyncio
async def test_shared_worker_request_capacity_failure_is_terminal_and_leak_free():
    worker = SharedMemoryProcessWorker(
        "transform",
        IdentityTransform(),
        execution_timeout_s=2.0,
        request_capacity_bytes=64,
        response_capacity_bytes=4096,
    )
    await worker.heartbeat()
    names = worker.shared_memory_names
    assert names is not None
    _assert_names_exist(names)

    with pytest.raises(ProcessWorkerTransportError, match="capacity"):
        await worker.invoke("transform", np.arange(100, dtype=np.float64))

    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "transport_error"
    assert not worker.is_alive
    _assert_names_absent(names)


@pytest.mark.asyncio
async def test_shared_worker_result_capacity_failure_is_terminal_and_leak_free():
    worker = SharedMemoryProcessWorker(
        "transform",
        LargeResultTransform(),
        execution_timeout_s=2.0,
        request_capacity_bytes=4096,
        response_capacity_bytes=64,
    )
    await worker.heartbeat()
    names = worker.shared_memory_names
    assert names is not None

    with pytest.raises(ProcessWorkerTransportError, match="result transport failed"):
        await worker.invoke("transform", 1)

    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "transport_error"
    assert not worker.is_alive
    _assert_names_absent(names)


@pytest.mark.asyncio
async def test_shared_worker_timeout_terminates_child_and_unlinks_transport():
    worker = SharedMemoryProcessWorker(
        "transform",
        SleepTransform(),
        execution_timeout_s=0.05,
        request_capacity_bytes=4096,
        response_capacity_bytes=4096,
    )
    await worker.heartbeat()
    names = worker.shared_memory_names
    assert names is not None

    with pytest.raises(ProcessWorkerTimeoutError):
        await worker.invoke("transform", 1)

    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "timeout"
    assert not worker.is_alive
    _assert_names_absent(names)


@pytest.mark.asyncio
async def test_shared_worker_crash_admits_no_result_and_unlinks_transport():
    worker = SharedMemoryProcessWorker(
        "transform",
        CrashTransform(),
        execution_timeout_s=2.0,
        request_capacity_bytes=4096,
        response_capacity_bytes=4096,
    )
    await worker.heartbeat()
    names = worker.shared_memory_names
    assert names is not None

    with pytest.raises(Exception, match="worker"):
        await worker.invoke("transform", 1)

    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "crashed"
    assert not worker.is_alive
    _assert_names_absent(names)


@pytest.mark.asyncio
async def test_shared_worker_asyncio_cancellation_terminates_and_unlinks():
    worker = SharedMemoryProcessWorker(
        "transform",
        SleepTransform(),
        execution_timeout_s=5.0,
        request_capacity_bytes=4096,
        response_capacity_bytes=4096,
    )
    await worker.heartbeat()
    names = worker.shared_memory_names
    assert names is not None

    task = asyncio.create_task(worker.invoke("transform", 1))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "cancelled"
    assert not worker.is_alive
    _assert_names_absent(names)


def test_shared_worker_does_not_unlink_if_direct_child_survives_escalation():
    worker = SharedMemoryProcessWorker(
        "transform",
        IdentityTransform(),
        execution_timeout_s=1.0,
        request_capacity_bytes=4096,
        response_capacity_bytes=4096,
    )
    worker._prepare_payload_transport()
    names = worker.shared_memory_names
    assert names is not None
    process = ImmortalProcess()
    worker._process = process

    try:
        with pytest.raises(ProcessWorkerTerminationError, match="remained alive"):
            worker.abort()
        assert process.terminate_calls == 1
        assert process.kill_calls == 1
        assert process.join_calls == 2
        assert worker._process is process
        _assert_names_exist(names)
    finally:
        worker._process = None
        worker._cleanup_payload_transport()

    _assert_names_absent(names)
