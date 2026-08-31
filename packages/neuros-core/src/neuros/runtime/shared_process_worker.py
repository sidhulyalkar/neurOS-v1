"""Persistent direct-child worker using canonical shared-memory payload transport.

This worker is deliberately separate from :mod:`neuros.runtime.process_worker`
while Phase C transport semantics are being qualified. The already-qualified
pickle worker remains untouched. Both workers expose the same execution receipt
and fail-closed direct-child authority, while this implementation owns one
request and one response shared-memory mailbox for each worker generation.

The multiprocessing pipe remains a small control plane. Array bytes and
canonical neural payloads live in shared memory and are materialized into local
memory before arbitrary operator callbacks execute. This is not a zero-copy
callback contract.
"""
from __future__ import annotations

import asyncio
import inspect
import multiprocessing as mp
import pickle
from multiprocessing.connection import Connection
from typing import Any

from .process_worker import (
    ProcessCallResult,
    ProcessExecutionReceipt,
    ProcessWorkerCrashedError,
    ProcessWorkerError,
    ProcessWorkerProtocolError,
    ProcessWorkerRemoteError,
    ProcessWorkerSerializationError,
    ProcessWorkerTerminationError,
    ProcessWorkerTimeoutError,
)
from .transport import NeuralTransportError, SharedMemoryMailbox

_PROTOCOL = 1


class ProcessWorkerTransportError(ProcessWorkerError):
    """Shared-memory creation, codec, identity, or cleanup authority failure."""


def _run_callback(func: Any, item: Any) -> Any:
    value = func(item)
    if not inspect.isawaitable(value):
        return value

    async def _await() -> Any:
        return await value

    return asyncio.run(_await())


def _send(conn: Connection, envelope: dict[str, Any]) -> None:
    conn.send_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))


def _child(
    conn: Connection,
    operator: Any,
    node_id: str,
    generation: int,
    request_mailbox_name: str,
    request_capacity_bytes: int,
    response_mailbox_name: str,
    response_capacity_bytes: int,
) -> None:
    base = {"protocol": _PROTOCOL, "node_id": node_id, "generation": generation}
    request_mailbox: SharedMemoryMailbox | None = None
    response_mailbox: SharedMemoryMailbox | None = None
    try:
        try:
            request_mailbox = SharedMemoryMailbox.attach(
                request_mailbox_name, request_capacity_bytes
            )
            response_mailbox = SharedMemoryMailbox.attach(
                response_mailbox_name, response_capacity_bytes
            )
        except Exception as exc:
            try:
                _send(
                    conn,
                    {
                        **base,
                        "kind": "transport_error",
                        "error_type": type(exc).__name__,
                        "message": f"child shared-memory attach failed: {exc}",
                    },
                )
            finally:
                return

        _send(conn, {**base, "kind": "ready"})
        while True:
            try:
                payload = conn.recv_bytes()
            except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
                return
            try:
                request = pickle.loads(payload)
            except Exception as exc:
                try:
                    _send(
                        conn,
                        {**base, "kind": "protocol_error", "message": str(exc)},
                    )
                except (EOFError, ConnectionResetError, BrokenPipeError, OSError):
                    pass
                return

            if not isinstance(request, dict) or any(
                (
                    request.get("protocol") != _PROTOCOL,
                    request.get("node_id") != node_id,
                    request.get("generation") != generation,
                )
            ):
                _send(
                    conn,
                    {**base, "kind": "protocol_error", "message": "identity mismatch"},
                )
                return

            command = request.get("command")
            if command == "heartbeat":
                _send(conn, {**base, "kind": "heartbeat"})
                continue
            if command == "shutdown":
                _send(conn, {**base, "kind": "shutdown"})
                return

            request_id = request.get("request_id")
            method = request.get("method")
            item_envelope = request.get("item")
            call_base = {**base, "request_id": request_id}
            if (
                command != "call"
                or not isinstance(request_id, int)
                or isinstance(request_id, bool)
                or request_id <= 0
                or not isinstance(method, str)
                or not method
            ):
                _send(
                    conn,
                    {**call_base, "kind": "protocol_error", "message": "malformed call"},
                )
                return

            try:
                assert request_mailbox is not None
                item = request_mailbox.decode(
                    item_envelope, expected_lease_id=request_id
                )
            except NeuralTransportError as exc:
                _send(
                    conn,
                    {
                        **call_base,
                        "kind": "transport_error",
                        "error_type": type(exc).__name__,
                        "message": f"input transport failed: {exc}",
                    },
                )
                return
            except Exception as exc:
                _send(
                    conn,
                    {
                        **call_base,
                        "kind": "transport_error",
                        "error_type": type(exc).__name__,
                        "message": f"input transport decoding failed: {exc}",
                    },
                )
                return

            try:
                result = _run_callback(getattr(operator, method), item)
            except BaseException as exc:
                _send(
                    conn,
                    {
                        **call_base,
                        "kind": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                continue

            try:
                assert response_mailbox is not None
                result_envelope = response_mailbox.encode(
                    result, lease_id=request_id
                )
            except NeuralTransportError as exc:
                _send(
                    conn,
                    {
                        **call_base,
                        "kind": "transport_error",
                        "error_type": type(exc).__name__,
                        "message": f"result transport failed: {exc}",
                    },
                )
                return
            except Exception as exc:
                _send(
                    conn,
                    {
                        **call_base,
                        "kind": "transport_error",
                        "error_type": type(exc).__name__,
                        "message": f"result transport encoding failed: {exc}",
                    },
                )
                return

            _send(
                conn,
                {**call_base, "kind": "result", "result": result_envelope},
            )
    finally:
        if request_mailbox is not None:
            try:
                request_mailbox.close()
            except Exception:
                pass
        if response_mailbox is not None:
            try:
                response_mailbox.close()
            except Exception:
                pass
        conn.close()


class SharedMemoryProcessWorker:
    """One persistent operator child with parent-owned shared-memory mailboxes."""

    def __init__(
        self,
        node_id: str,
        operator: Any,
        *,
        execution_timeout_s: float,
        request_capacity_bytes: int,
        response_capacity_bytes: int,
        generation: int = 0,
        startup_timeout_s: float = 5.0,
        termination_grace_s: float = 0.25,
    ) -> None:
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if min(execution_timeout_s, startup_timeout_s, termination_grace_s) <= 0:
            raise ValueError("worker timeouts must be positive")
        if generation < 0:
            raise ValueError("generation must be non-negative")
        for field_name, capacity in (
            ("request_capacity_bytes", request_capacity_bytes),
            ("response_capacity_bytes", response_capacity_bytes),
        ):
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        self.node_id = node_id
        self.operator = operator
        self.execution_timeout_s = float(execution_timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.termination_grace_s = float(termination_grace_s)
        self.request_capacity_bytes = request_capacity_bytes
        self.response_capacity_bytes = response_capacity_bytes
        self.generation = int(generation)
        self._ctx = mp.get_context("spawn")
        self._conn: Connection | None = None
        self._process: mp.Process | None = None
        self._request_mailbox: SharedMemoryMailbox | None = None
        self._response_mailbox: SharedMemoryMailbox | None = None
        self._request_id = 0
        self._last_receipt: ProcessExecutionReceipt | None = None
        self._lock = asyncio.Lock()
        self._terminal = False

    @property
    def last_receipt(self) -> ProcessExecutionReceipt | None:
        return self._last_receipt

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def is_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

    @property
    def shared_memory_names(self) -> dict[str, str] | None:
        if self._request_mailbox is None or self._response_mailbox is None:
            return None
        return {
            "request": self._request_mailbox.name,
            "response": self._response_mailbox.name,
        }

    def _receipt(
        self, request_id: int, outcome: str, error_type: str | None = None
    ) -> None:
        self._last_receipt = ProcessExecutionReceipt(
            self.node_id,
            self.generation,
            request_id,
            outcome,
            error_type,
        )

    def _identity(self, response: dict[str, Any], request_id: int | None) -> None:
        if (
            response.get("protocol") != _PROTOCOL
            or response.get("node_id") != self.node_id
            or response.get("generation") != self.generation
            or (request_id is not None and response.get("request_id") != request_id)
        ):
            raise ProcessWorkerProtocolError(
                self.node_id,
                f"stale or mismatched process response for request {request_id}",
            )

    def _recv(self, timeout_s: float, request_id: int | None) -> dict[str, Any]:
        if self._conn is None:
            raise ProcessWorkerCrashedError(self.node_id, "worker IPC is closed")
        if not self._conn.poll(timeout_s):
            if not self.is_alive:
                raise ProcessWorkerCrashedError(
                    self.node_id, "worker exited before response"
                )
            raise ProcessWorkerTimeoutError(
                self.node_id,
                f"request {request_id} exceeded {timeout_s:.6f}s hard execution timeout",
            )
        try:
            response = pickle.loads(self._conn.recv_bytes())
        except EOFError as exc:
            raise ProcessWorkerCrashedError(
                self.node_id, "worker EOF before response"
            ) from exc
        except Exception as exc:
            raise ProcessWorkerProtocolError(
                self.node_id, f"invalid worker response: {exc}"
            ) from exc
        if not isinstance(response, dict):
            raise ProcessWorkerProtocolError(
                self.node_id, "worker response is not a mapping"
            )
        self._identity(response, request_id)
        return response

    def _send_control(self, command: str) -> None:
        if self._conn is None:
            raise ProcessWorkerCrashedError(self.node_id, "worker IPC is closed")
        try:
            _send(
                self._conn,
                {
                    "protocol": _PROTOCOL,
                    "node_id": self.node_id,
                    "generation": self.generation,
                    "command": command,
                },
            )
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise ProcessWorkerCrashedError(
                self.node_id, f"worker IPC failed while sending {command}"
            ) from exc

    def _create_transport(self) -> None:
        if self._request_mailbox is not None or self._response_mailbox is not None:
            if self._request_mailbox is None or self._response_mailbox is None:
                raise ProcessWorkerTransportError(
                    self.node_id, "shared-memory mailbox ownership is incomplete"
                )
            return

        try:
            request_mailbox = SharedMemoryMailbox(self.request_capacity_bytes)
        except Exception as exc:
            raise ProcessWorkerTransportError(
                self.node_id,
                "request shared-memory mailbox creation failed: "
                f"{type(exc).__name__}: {exc}",
            ) from exc

        try:
            response_mailbox = SharedMemoryMailbox(self.response_capacity_bytes)
        except Exception as exc:
            cleanup_suffix = ""
            try:
                request_mailbox.close_and_unlink()
            except Exception as cleanup_exc:
                cleanup_suffix = (
                    "; request mailbox cleanup also failed: "
                    f"{type(cleanup_exc).__name__}: {cleanup_exc}"
                )
            raise ProcessWorkerTransportError(
                self.node_id,
                "response shared-memory mailbox creation failed: "
                f"{type(exc).__name__}: {exc}{cleanup_suffix}",
            ) from exc

        self._request_mailbox = request_mailbox
        self._response_mailbox = response_mailbox

    def _cleanup_transport(self) -> None:
        errors: list[str] = []
        for attr_name in ("_request_mailbox", "_response_mailbox"):
            mailbox = getattr(self, attr_name)
            if mailbox is None:
                continue
            try:
                mailbox.close_and_unlink()
            except Exception as exc:
                errors.append(f"{attr_name}: {type(exc).__name__}: {exc}")
                continue
            setattr(self, attr_name, None)
        if errors:
            raise ProcessWorkerTransportError(
                self.node_id,
                "shared-memory cleanup authority failed: " + "; ".join(errors),
            )

    def _start(self) -> None:
        if self._terminal:
            raise ProcessWorkerError(
                self.node_id, "worker is terminal; create a new generation"
            )
        if self._process is not None:
            if self._process.is_alive():
                return
            raise ProcessWorkerCrashedError(self.node_id, "worker exited")

        self._create_transport()
        assert self._request_mailbox is not None
        assert self._response_mailbox is not None
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_child,
            args=(
                child,
                self.operator,
                self.node_id,
                self.generation,
                self._request_mailbox.name,
                self.request_capacity_bytes,
                self._response_mailbox.name,
                self.response_capacity_bytes,
            ),
            name=f"neuros-shm-process:{self.node_id}:g{self.generation}",
        )
        self._conn, self._process = parent, process
        try:
            process.start()
        except Exception as exc:
            parent.close()
            self._conn = self._process = None
            cleanup_suffix = ""
            try:
                self._cleanup_transport()
            except ProcessWorkerTransportError as cleanup_exc:
                cleanup_suffix = f"; transport cleanup also failed: {cleanup_exc}"
            raise ProcessWorkerSerializationError(
                self.node_id,
                f"operator/start serialization failed: {exc}{cleanup_suffix}",
            ) from exc
        finally:
            child.close()

        ready = self._recv(self.startup_timeout_s, None)
        if ready.get("kind") == "transport_error":
            raise ProcessWorkerTransportError(
                self.node_id, str(ready.get("message") or "child transport startup failed")
            )
        if ready.get("kind") != "ready":
            raise ProcessWorkerProtocolError(
                self.node_id, "worker did not become ready"
            )
        self._send_control("heartbeat")
        heartbeat = self._recv(self.startup_timeout_s, None)
        if heartbeat.get("kind") != "heartbeat":
            raise ProcessWorkerProtocolError(self.node_id, "worker heartbeat failed")

    async def invoke(self, method: str, item: Any) -> ProcessCallResult:
        async with self._lock:
            request_id = self._request_id + 1
            try:
                await asyncio.to_thread(self._start)
                self._request_id = request_id
                if self._request_mailbox is None:
                    raise ProcessWorkerTransportError(
                        self.node_id, "request mailbox is unavailable"
                    )
                try:
                    item_envelope = self._request_mailbox.encode(
                        item, lease_id=request_id
                    )
                except NeuralTransportError as exc:
                    raise ProcessWorkerTransportError(
                        self.node_id, f"input transport failed: {exc}"
                    ) from exc
                except Exception as exc:
                    raise ProcessWorkerTransportError(
                        self.node_id,
                        f"input transport encoding failed: {type(exc).__name__}: {exc}",
                    ) from exc

                if self._conn is None:
                    raise ProcessWorkerCrashedError(
                        self.node_id, "worker IPC is closed"
                    )
                try:
                    await asyncio.to_thread(
                        _send,
                        self._conn,
                        {
                            "protocol": _PROTOCOL,
                            "node_id": self.node_id,
                            "generation": self.generation,
                            "command": "call",
                            "request_id": request_id,
                            "method": method,
                            "item": item_envelope,
                        },
                    )
                except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
                    raise ProcessWorkerCrashedError(
                        self.node_id,
                        f"worker IPC failed while sending request {request_id}",
                    ) from exc

                response = await asyncio.to_thread(
                    self._recv, self.execution_timeout_s, request_id
                )
            except asyncio.CancelledError:
                self._receipt(request_id, "cancelled")
                await asyncio.shield(asyncio.to_thread(self.abort))
                raise
            except ProcessWorkerTimeoutError:
                self._receipt(request_id, "timeout")
                await asyncio.to_thread(self.abort)
                raise
            except ProcessWorkerCrashedError:
                self._receipt(request_id, "crashed")
                await asyncio.to_thread(self.abort)
                raise
            except ProcessWorkerProtocolError:
                self._receipt(request_id, "protocol_error")
                await asyncio.to_thread(self.abort)
                raise
            except ProcessWorkerTransportError:
                self._request_id = request_id
                self._receipt(request_id, "transport_error")
                await asyncio.to_thread(self.abort)
                raise
            except ProcessWorkerSerializationError:
                self._request_id = request_id
                self._receipt(request_id, "serialization_error")
                await asyncio.to_thread(self.abort)
                raise
            except Exception as exc:
                self._request_id = request_id
                self._receipt(request_id, "serialization_error")
                await asyncio.to_thread(self.abort)
                raise ProcessWorkerSerializationError(
                    self.node_id, f"input/request serialization failed: {exc}"
                ) from exc

            kind = response.get("kind")
            if kind == "result":
                if self._response_mailbox is None:
                    self._receipt(request_id, "transport_error")
                    await asyncio.to_thread(self.abort)
                    raise ProcessWorkerTransportError(
                        self.node_id, "response mailbox is unavailable"
                    )
                try:
                    result = self._response_mailbox.decode(
                        response.get("result"), expected_lease_id=request_id
                    )
                except NeuralTransportError as exc:
                    self._receipt(request_id, "transport_error")
                    await asyncio.to_thread(self.abort)
                    raise ProcessWorkerTransportError(
                        self.node_id, f"result transport failed: {exc}"
                    ) from exc
                except Exception as exc:
                    self._receipt(request_id, "transport_error")
                    await asyncio.to_thread(self.abort)
                    raise ProcessWorkerTransportError(
                        self.node_id,
                        f"result transport decoding failed: {type(exc).__name__}: {exc}",
                    ) from exc

                receipt = ProcessExecutionReceipt(
                    self.node_id, self.generation, request_id, "success"
                )
                self._last_receipt = receipt
                return ProcessCallResult(result, receipt)

            if kind == "error":
                error_type = str(response.get("error_type") or "RemoteError")
                self._receipt(request_id, "error", error_type)
                raise ProcessWorkerRemoteError(
                    self.node_id,
                    error_type,
                    str(response.get("message") or ""),
                )
            if kind == "transport_error":
                self._receipt(request_id, "transport_error")
                await asyncio.to_thread(self.abort)
                raise ProcessWorkerTransportError(
                    self.node_id, str(response.get("message") or "transport failure")
                )
            if kind == "serialization_error":
                self._receipt(request_id, "serialization_error")
                await asyncio.to_thread(self.abort)
                raise ProcessWorkerSerializationError(
                    self.node_id,
                    str(response.get("message") or "serialization failure"),
                )

            self._receipt(request_id, "protocol_error")
            await asyncio.to_thread(self.abort)
            raise ProcessWorkerProtocolError(
                self.node_id, f"unexpected worker response kind {kind!r}"
            )

    async def heartbeat(self) -> None:
        async with self._lock:
            try:
                await asyncio.to_thread(self._start)
                await asyncio.to_thread(self._send_control, "heartbeat")
                response = await asyncio.to_thread(
                    self._recv, self.startup_timeout_s, None
                )
                if response.get("kind") != "heartbeat":
                    raise ProcessWorkerProtocolError(
                        self.node_id, "worker heartbeat failed"
                    )
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.to_thread(self.abort))
                raise
            except Exception:
                await asyncio.to_thread(self.abort)
                raise

    def _terminate_owned_process(self, process: mp.Process) -> None:
        """Prove the direct child is dead or retain its live handle and fail."""

        if process.is_alive():
            process.terminate()
            process.join(self.termination_grace_s)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(self.termination_grace_s)
        if process.is_alive():
            self._process = process
            raise ProcessWorkerTerminationError(
                self.node_id,
                "direct child remained alive after terminate/join/kill escalation",
            )
        process.close()
        self._process = None

    def abort(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        self._conn = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if process is None:
            self._process = None
            self._cleanup_transport()
            return
        self._terminate_owned_process(process)
        # Shared-memory names are unlinked only after direct-child death has
        # been proven. If termination raises, these handles intentionally stay.
        self._cleanup_transport()

    def close(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        if process is None:
            if conn is not None:
                conn.close()
            self._conn = None
            self._cleanup_transport()
            return

        if process.is_alive() and conn is not None:
            try:
                self._send_control("shutdown")
                if conn.poll(self.termination_grace_s):
                    response = pickle.loads(conn.recv_bytes())
                    self._identity(response, None)
            except Exception:
                pass

        process.join(self.termination_grace_s)
        if conn is not None:
            conn.close()
        self._conn = None
        if process.is_alive():
            self._terminate_owned_process(process)
            self._cleanup_transport()
            return

        process.close()
        self._process = None
        self._cleanup_transport()
