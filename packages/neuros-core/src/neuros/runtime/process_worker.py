"""Persistent direct-child authority for process-executed runtime nodes.

Trusted Python objects cross a multiprocessing boundary, so this is fault
isolation, not a security sandbox. Termination authority covers only the direct
child created here, never arbitrary descendants.

The process lifecycle is intentionally transport-agnostic. Pickle remains the
qualified default payload transport. Alternate transports plug into the same
startup, identity, receipt, timeout, cancellation, crash, and termination state
machine instead of duplicating process authority.
"""
from __future__ import annotations

import asyncio
import inspect
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any, Callable

_PROTOCOL = 1


class ProcessWorkerError(RuntimeError):
    def __init__(self, node_id: str, message: str) -> None:
        self.node_id = node_id
        super().__init__(message)


class ProcessWorkerTimeoutError(ProcessWorkerError):
    pass


class ProcessWorkerCrashedError(ProcessWorkerError):
    pass


class ProcessWorkerProtocolError(ProcessWorkerError):
    pass


class ProcessWorkerSerializationError(ProcessWorkerError):
    pass


class ProcessWorkerTransportError(ProcessWorkerError):
    """Payload transport creation, codec, identity, or cleanup failure."""


class ProcessWorkerTerminationError(ProcessWorkerError):
    pass


class ProcessWorkerRemoteError(ProcessWorkerError):
    def __init__(self, node_id: str, error_type: str, message: str) -> None:
        self.remote_error_type = error_type
        self.runtime_error_type = error_type
        super().__init__(node_id, message)


@dataclass(frozen=True, slots=True)
class ProcessExecutionReceipt:
    node_id: str
    generation: int
    request_id: int
    outcome: str
    remote_error_type: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "generation": self.generation,
            "request_id": self.request_id,
            "outcome": self.outcome,
            "remote_error_type": self.remote_error_type,
        }


@dataclass(frozen=True, slots=True)
class ProcessCallResult:
    result: Any
    receipt: ProcessExecutionReceipt


class _PayloadSerializationFailure(Exception):
    pass


class _PayloadTransportFailure(Exception):
    def __init__(self, message: str, *, error_type: str | None = None) -> None:
        self.error_type = error_type
        super().__init__(message)


class _PayloadProtocolFailure(Exception):
    pass


class _PickleParentTransport:
    """Parent-side payload adapter preserving the Phase B pickle semantics."""

    @property
    def shared_memory_names(self) -> None:
        return None

    def prepare(self) -> None:
        return None

    def child_spec(self) -> None:
        return None

    def encode_request(self, value: Any, lease_id: int) -> bytes:
        del lease_id
        try:
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise _PayloadSerializationFailure(
                f"input/request serialization failed: {exc}"
            ) from exc

    def decode_result(self, payload: Any, lease_id: int) -> Any:
        del lease_id
        if not isinstance(payload, bytes):
            raise _PayloadSerializationFailure(
                "result deserialization failed: result payload is not bytes"
            )
        try:
            return pickle.loads(payload)
        except Exception as exc:
            raise _PayloadSerializationFailure(
                f"result deserialization failed: {exc}"
            ) from exc

    def cleanup(self) -> None:
        return None


class _PickleChildTransport:
    def decode_request(self, payload: Any, lease_id: int) -> Any:
        del lease_id
        if not isinstance(payload, bytes):
            raise _PayloadProtocolFailure("malformed call")
        try:
            return pickle.loads(payload)
        except Exception as exc:
            raise _PayloadSerializationFailure(
                f"input deserialization failed: {exc}"
            ) from exc

    def encode_result(self, value: Any, lease_id: int) -> bytes:
        del lease_id
        try:
            return pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as exc:
            raise _PayloadSerializationFailure(
                f"result serialization failed: {exc}"
            ) from exc

    def close(self) -> None:
        return None


def _make_pickle_child_transport(spec: Any) -> _PickleChildTransport:
    if spec is not None:
        raise _PayloadProtocolFailure("pickle child transport spec must be null")
    return _PickleChildTransport()


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
    child_transport_factory: Callable[[Any], Any],
    child_transport_spec: Any,
) -> None:
    base = {"protocol": _PROTOCOL, "node_id": node_id, "generation": generation}
    transport: Any | None = None
    try:
        try:
            transport = child_transport_factory(child_transport_spec)
        except _PayloadTransportFailure as exc:
            try:
                _send(
                    conn,
                    {
                        **base,
                        "kind": "transport_error",
                        "error_type": exc.error_type or type(exc).__name__,
                        "message": str(exc),
                    },
                )
            finally:
                return
        except _PayloadProtocolFailure as exc:
            try:
                _send(conn, {**base, "kind": "protocol_error", "message": str(exc)})
            finally:
                return
        except Exception as exc:
            try:
                _send(
                    conn,
                    {
                        **base,
                        "kind": "transport_error",
                        "error_type": type(exc).__name__,
                        "message": f"child payload transport startup failed: {exc}",
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
            item_payload = request.get("item")
            call_base = {**base, "request_id": request_id}
            if (
                command != "call"
                or isinstance(request_id, bool)
                or not isinstance(request_id, int)
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
                item = transport.decode_request(item_payload, request_id)
            except _PayloadProtocolFailure as exc:
                _send(
                    conn,
                    {**call_base, "kind": "protocol_error", "message": str(exc)},
                )
                return
            except _PayloadSerializationFailure as exc:
                _send(
                    conn,
                    {**call_base, "kind": "serialization_error", "message": str(exc)},
                )
                return
            except _PayloadTransportFailure as exc:
                _send(
                    conn,
                    {
                        **call_base,
                        "kind": "transport_error",
                        "error_type": exc.error_type or type(exc).__name__,
                        "message": str(exc),
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
                        "message": f"input payload decoding failed: {exc}",
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
                result_payload = transport.encode_result(result, request_id)
            except _PayloadSerializationFailure as exc:
                _send(
                    conn,
                    {**call_base, "kind": "serialization_error", "message": str(exc)},
                )
                return
            except _PayloadTransportFailure as exc:
                _send(
                    conn,
                    {
                        **call_base,
                        "kind": "transport_error",
                        "error_type": exc.error_type or type(exc).__name__,
                        "message": str(exc),
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
                        "message": f"result payload encoding failed: {exc}",
                    },
                )
                return

            _send(conn, {**call_base, "kind": "result", "result": result_payload})
    finally:
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass
        conn.close()


class PersistentProcessWorker:
    """One operator instance in one spawn-safe direct child, with no auto-retry."""

    def __init__(
        self,
        node_id: str,
        operator: Any,
        *,
        execution_timeout_s: float,
        generation: int = 0,
        startup_timeout_s: float = 5.0,
        termination_grace_s: float = 0.25,
        _payload_transport: Any | None = None,
        _child_transport_factory: Callable[[Any], Any] = _make_pickle_child_transport,
        _process_name_prefix: str = "neuros-process",
    ) -> None:
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if min(execution_timeout_s, startup_timeout_s, termination_grace_s) <= 0:
            raise ValueError("worker timeouts must be positive")
        if generation < 0:
            raise ValueError("generation must be non-negative")
        if not _process_name_prefix:
            raise ValueError("_process_name_prefix must be non-empty")
        self.node_id = node_id
        self.operator = operator
        self.execution_timeout_s = float(execution_timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.termination_grace_s = float(termination_grace_s)
        self.generation = int(generation)
        self._ctx = mp.get_context("spawn")
        self._conn: Connection | None = None
        self._process: mp.Process | None = None
        self._request_id = 0
        self._last_receipt: ProcessExecutionReceipt | None = None
        self._lock = asyncio.Lock()
        self._terminal = False
        self._payload_transport = _payload_transport or _PickleParentTransport()
        self._child_transport_factory = _child_transport_factory
        self._process_name_prefix = _process_name_prefix

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
        return self._payload_transport.shared_memory_names

    def _receipt(
        self, request_id: int, outcome: str, error_type: str | None = None
    ) -> None:
        self._last_receipt = ProcessExecutionReceipt(
            self.node_id, self.generation, request_id, outcome, error_type
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

    def _cleanup_payload_transport(self) -> None:
        try:
            self._payload_transport.cleanup()
        except _PayloadTransportFailure as exc:
            raise ProcessWorkerTransportError(self.node_id, str(exc)) from exc
        except ProcessWorkerTransportError:
            raise
        except Exception as exc:
            raise ProcessWorkerTransportError(
                self.node_id,
                f"payload transport cleanup authority failed: {type(exc).__name__}: {exc}",
            ) from exc

    def _prepare_payload_transport(self) -> None:
        try:
            self._payload_transport.prepare()
        except _PayloadTransportFailure as exc:
            raise ProcessWorkerTransportError(self.node_id, str(exc)) from exc
        except ProcessWorkerTransportError:
            raise
        except Exception as exc:
            raise ProcessWorkerTransportError(
                self.node_id,
                f"payload transport preparation failed: {type(exc).__name__}: {exc}",
            ) from exc

    def _start(self) -> None:
        if self._terminal:
            raise ProcessWorkerError(
                self.node_id, "worker is terminal; create a new generation"
            )
        if self._process is not None:
            if self._process.is_alive():
                return
            raise ProcessWorkerCrashedError(self.node_id, "worker exited")

        self._prepare_payload_transport()
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_child,
            args=(
                child,
                self.operator,
                self.node_id,
                self.generation,
                self._child_transport_factory,
                self._payload_transport.child_spec(),
            ),
            name=f"{self._process_name_prefix}:{self.node_id}:g{self.generation}",
        )
        self._conn, self._process = parent, process
        try:
            process.start()
        except Exception as exc:
            parent.close()
            self._conn = self._process = None
            cleanup_suffix = ""
            try:
                self._cleanup_payload_transport()
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
                self.node_id,
                str(ready.get("message") or "child payload transport startup failed"),
            )
        if ready.get("kind") != "ready":
            raise ProcessWorkerProtocolError(self.node_id, "worker did not become ready")
        self._send_control("heartbeat")
        heartbeat = self._recv(self.startup_timeout_s, None)
        if heartbeat.get("kind") != "heartbeat":
            raise ProcessWorkerProtocolError(self.node_id, "worker heartbeat failed")

    def _encode_request(self, item: Any, request_id: int) -> Any:
        try:
            return self._payload_transport.encode_request(item, request_id)
        except _PayloadSerializationFailure as exc:
            raise ProcessWorkerSerializationError(self.node_id, str(exc)) from exc
        except _PayloadTransportFailure as exc:
            raise ProcessWorkerTransportError(self.node_id, str(exc)) from exc
        except Exception as exc:
            raise ProcessWorkerTransportError(
                self.node_id,
                f"input payload encoding failed: {type(exc).__name__}: {exc}",
            ) from exc

    def _decode_result(self, payload: Any, request_id: int) -> Any:
        try:
            return self._payload_transport.decode_result(payload, request_id)
        except _PayloadSerializationFailure as exc:
            raise ProcessWorkerSerializationError(self.node_id, str(exc)) from exc
        except _PayloadTransportFailure as exc:
            raise ProcessWorkerTransportError(self.node_id, str(exc)) from exc
        except Exception as exc:
            raise ProcessWorkerTransportError(
                self.node_id,
                f"result payload decoding failed: {type(exc).__name__}: {exc}",
            ) from exc

    async def invoke(self, method: str, item: Any) -> ProcessCallResult:
        async with self._lock:
            request_id = self._request_id + 1
            try:
                await asyncio.to_thread(self._start)
                self._request_id = request_id
                item_payload = self._encode_request(item, request_id)
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
                            "item": item_payload,
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
                try:
                    result = self._decode_result(response.get("result"), request_id)
                except ProcessWorkerSerializationError:
                    self._receipt(request_id, "serialization_error")
                    await asyncio.to_thread(self.abort)
                    raise
                except ProcessWorkerTransportError:
                    self._receipt(request_id, "transport_error")
                    await asyncio.to_thread(self.abort)
                    raise
                receipt = ProcessExecutionReceipt(
                    self.node_id, self.generation, request_id, "success"
                )
                self._last_receipt = receipt
                return ProcessCallResult(result, receipt)
            if kind == "error":
                error_type = str(response.get("error_type") or "RemoteError")
                self._receipt(request_id, "error", error_type)
                raise ProcessWorkerRemoteError(
                    self.node_id, error_type, str(response.get("message") or "")
                )
            if kind == "serialization_error":
                self._receipt(request_id, "serialization_error")
                await asyncio.to_thread(self.abort)
                raise ProcessWorkerSerializationError(
                    self.node_id,
                    str(response.get("message") or "serialization failure"),
                )
            if kind == "transport_error":
                self._receipt(request_id, "transport_error")
                await asyncio.to_thread(self.abort)
                raise ProcessWorkerTransportError(
                    self.node_id, str(response.get("message") or "transport failure")
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
        """Prove the direct child is dead or fail closed with its handle retained."""

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
            self._cleanup_payload_transport()
            return
        self._terminate_owned_process(process)
        # Resource cleanup happens only after direct-child death is proven.
        self._cleanup_payload_transport()

    def close(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        if process is None:
            if conn is not None:
                conn.close()
            self._conn = None
            self._cleanup_payload_transport()
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
            self._cleanup_payload_transport()
            return
        process.close()
        self._process = None
        self._cleanup_payload_transport()
