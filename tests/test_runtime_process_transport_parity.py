from __future__ import annotations

import asyncio
import os
import time

import numpy as np
import pytest

from neuros.runtime.process_worker import (
    PersistentProcessWorker,
    ProcessWorkerProtocolError,
    ProcessWorkerRemoteError,
    ProcessWorkerTimeoutError,
    ProcessWorkerTransportError,
    _PayloadTransportFailure,
    _PickleParentTransport,
)
from neuros.runtime.shared_process_worker import SharedMemoryProcessWorker


class CounterTransform:
    def __init__(self):
        self.calls = 0

    def transform(self, item):
        self.calls += 1
        return self.calls, np.asarray(item) + self.calls


class FailingTransform:
    def transform(self, item):
        raise LookupError(f"rejected {int(np.asarray(item).reshape(-1)[0])}")


class SleepTransform:
    def transform(self, item):
        time.sleep(2.0)
        return item


class CrashTransform:
    def transform(self, item):
        os._exit(31)


def _worker(transport: str, operator, *, timeout_s: float = 2.0):
    if transport == "pickle":
        return PersistentProcessWorker(
            "transform", operator, execution_timeout_s=timeout_s
        )
    return SharedMemoryProcessWorker(
        "transform",
        operator,
        execution_timeout_s=timeout_s,
        request_capacity_bytes=64 * 1024,
        response_capacity_bytes=64 * 1024,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["pickle", "shared_memory"])
async def test_process_transports_share_stateful_lifecycle_and_receipt_semantics(transport):
    worker = _worker(transport, CounterTransform())
    try:
        first = await worker.invoke("transform", np.array([10], dtype=np.int64))
        second = await worker.invoke("transform", np.array([20], dtype=np.int64))
        assert first.result[0] == 1
        assert second.result[0] == 2
        assert np.array_equal(first.result[1], np.array([11], dtype=np.int64))
        assert np.array_equal(second.result[1], np.array([22], dtype=np.int64))
        assert second.receipt.request_id == 2
        assert second.receipt.outcome == "success"
    finally:
        worker.close()
    assert not worker.is_alive


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["pickle", "shared_memory"])
async def test_process_transports_preserve_remote_exception_type_without_killing_worker(transport):
    worker = _worker(transport, FailingTransform())
    try:
        with pytest.raises(ProcessWorkerRemoteError) as caught:
            await worker.invoke("transform", np.array([7], dtype=np.int64))
        assert caught.value.remote_error_type == "LookupError"
        assert worker.last_receipt is not None
        assert worker.last_receipt.outcome == "error"
        assert worker.last_receipt.remote_error_type == "LookupError"
        assert worker.is_alive
    finally:
        worker.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["pickle", "shared_memory"])
async def test_process_transports_timeout_with_same_terminal_authority(transport):
    worker = _worker(transport, SleepTransform(), timeout_s=0.05)
    with pytest.raises(ProcessWorkerTimeoutError):
        await worker.invoke("transform", np.array([1], dtype=np.int64))
    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "timeout"
    assert not worker.is_alive


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["pickle", "shared_memory"])
async def test_process_transports_cancel_with_same_terminal_authority(transport):
    worker = _worker(transport, SleepTransform(), timeout_s=5.0)
    await worker.heartbeat()
    task = asyncio.create_task(
        worker.invoke("transform", np.array([1], dtype=np.int64))
    )
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "cancelled"
    assert not worker.is_alive


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["pickle", "shared_memory"])
async def test_process_transports_crash_with_same_terminal_authority(transport):
    worker = _worker(transport, CrashTransform())
    with pytest.raises(Exception, match="worker"):
        await worker.invoke("transform", np.array([1], dtype=np.int64))
    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "crashed"
    assert not worker.is_alive


@pytest.mark.parametrize("transport", ["pickle", "shared_memory"])
def test_common_control_plane_rejects_bool_as_integer_identity(transport):
    worker = _worker(transport, CounterTransform())
    base = {
        "protocol": 1,
        "node_id": "transform",
        "generation": 0,
        "kind": "heartbeat",
    }

    corrupted = {**base, "protocol": True}
    with pytest.raises(ProcessWorkerProtocolError, match="exact integer"):
        worker._identity(corrupted, None)

    corrupted = {**base, "generation": False}
    with pytest.raises(ProcessWorkerProtocolError, match="exact integer"):
        worker._identity(corrupted, None)

    corrupted = {**base, "request_id": 1}
    with pytest.raises(ProcessWorkerProtocolError, match="unexpectedly carries"):
        worker._identity(corrupted, None)

    call = {
        "protocol": 1,
        "node_id": "transform",
        "generation": 0,
        "request_id": True,
        "kind": "result",
    }
    with pytest.raises(ProcessWorkerProtocolError, match="exact integer"):
        worker._identity(call, 1)


class FailOnceCleanupTransport(_PickleParentTransport):
    def __init__(self):
        self.cleanup_calls = 0

    def cleanup(self):
        self.cleanup_calls += 1
        if self.cleanup_calls == 1:
            raise _PayloadTransportFailure("synthetic unlink failure")


@pytest.mark.asyncio
async def test_cleanup_failure_after_proven_child_death_does_not_mask_timeout():
    transport = FailOnceCleanupTransport()
    worker = PersistentProcessWorker(
        "transform",
        SleepTransform(),
        execution_timeout_s=0.05,
        _payload_transport=transport,
    )

    with pytest.raises(ProcessWorkerTimeoutError):
        await worker.invoke("transform", np.array([1], dtype=np.int64))

    assert worker.last_receipt is not None
    assert worker.last_receipt.outcome == "timeout"
    assert isinstance(worker.last_cleanup_error, ProcessWorkerTransportError)
    assert "synthetic unlink failure" in str(worker.last_cleanup_error)
    assert not worker.is_alive

    # Executor-owned close authority can retry a cleanup that failed after death.
    worker.close()
    assert transport.cleanup_calls >= 2


def test_cleanup_failure_without_primary_operation_is_not_silenced():
    transport = FailOnceCleanupTransport()
    worker = PersistentProcessWorker(
        "transform",
        CounterTransform(),
        execution_timeout_s=1.0,
        _payload_transport=transport,
    )
    with pytest.raises(ProcessWorkerTransportError, match="synthetic unlink failure"):
        worker.close()


class FailingChildSpecTransport(_PickleParentTransport):
    def __init__(self):
        self.prepared = False
        self.cleanup_calls = 0

    def prepare(self):
        self.prepared = True

    def child_spec(self):
        raise _PayloadTransportFailure("synthetic child spec failure")

    def cleanup(self):
        self.cleanup_calls += 1
        self.prepared = False


@pytest.mark.asyncio
async def test_startup_child_spec_failure_releases_prepared_transport():
    transport = FailingChildSpecTransport()
    worker = PersistentProcessWorker(
        "transform",
        CounterTransform(),
        execution_timeout_s=1.0,
        _payload_transport=transport,
    )
    with pytest.raises(ProcessWorkerTransportError, match="child payload transport specification"):
        await worker.heartbeat()
    assert transport.cleanup_calls >= 1
    assert not transport.prepared
    assert not worker.is_alive
