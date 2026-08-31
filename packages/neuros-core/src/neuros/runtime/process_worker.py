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
    """Payload/control transport creation, codec, identity, or cleanup failure."""


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


def _is_exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _control_int(mapping: dict[str, Any], key: str) -> int:
    if key not in mapping or not _is_exact_int(mapping[key]):
        raise _PayloadProtocolFailure(f"control field {key} must be an exact integer")
    return mapping[key]


def _control_str(mapping: dict[str, Any], key: str) -> str:
    if key not in mapping or not isinstance(mapping[key], str):
        raise _PayloadProtocolFailure(f"control field {key} must be a string")
    return mapping[key]


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
            raise _PayloadProtocolFailure("malformed call payload")
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

            if type(request) is not dict:
                _send(
                    conn,
                    {**base, "kind": "protocol_error", "message": "request is not an exact mapping"},
                )
                return
            try:
                protocol = _control_int(request, "protocol")
                request_node_id = _control_str(request, "node_id")
                request_generation = _control_int(request, "generation")
                command = _control_str(request, "command")
            except _PayloadProtocolFailure as exc:
                _send(conn, {**base, "kind": "protocol_error", "message": str(exc)})
                return
            if (
                protocol != _PROTOCOL
                or request_node_id != node_id
                or request_generation != generation
            ):
                _send(
                    conn,
                    {**base, "kind": "protocol_error", "message": "identity mismatch"},
                )
                return

            if command == "heartbeat":
                if "request_id" in request:
                    _send(
                        conn,
                        {**base, "kind": "protocol_error", "message": "heartbeat cannot carry request identity"},
                    )
                    return
                _send(conn, {**base, "kind": "heartbeat"})
                continue
            if command == "shutdown":
                if "request_id" in request:
                    _send(
                        conn,
                        {**base, "kind": "protocol_error", "message": "shutdown cannot carry request identity"},
                    )
                    return
                _send(conn, {**base, "kind": "shutdown"})
                return
            if command != "call":
                _send(
                    conn,
                    {**base, "kind": "protocol_error", "message": "unknown command"},
                )
                return

            try:
                request_id = _control_int(request, "request_id")
                method = _control_str(request, "method")
            except _PayloadProtocolFailure as exc:
                _send(conn, {**base, "kind": "protocol_error", "message": str(exc)})
                return
            if request_id <= 0 or not method:
                _send(
                    conn,
                    {**base, "kind": "protocol_error", "message": "malformed call"},
                )
                return

            item_payload = request.get("item")
            call_base = {**base, "request_id": request_id}
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
        if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
            raise ValueError("generation must be a non-negative integer")
        if not _process_name_prefix:
            raise ValueError("_process_name_prefix must be non-empty")
        self.node_id = node_id
        self.operator = operator
        self.execution_timeout_s = float(execution_timeout_s)
        self.startup_timeout_s = float(startup_timeout_s)
        self.termination_grace_s = float(termination_grace_s)
        self.generation = generation
        self._ctx = mp.get_context("spawn")
        self._conn: Connection | None = None
        self._process: mp.Process | None = None
        self._request_id = 0
        self._last_receipt: ProcessExecutionReceipt | None = None
        self._last_cleanup_error: ProcessWorkerTransportError | None = None
        self._lock = asyncio.Lock()
        self._terminal = False
        self._payload_transport = (
            _payload_transport if _payload_transport is not None else _PickleParentTransport()
        )
        self._child_transport_factory = _child_transport_factory
        self._process_name_prefix = _process_name_prefix

    @property
    def last_receipt(self) -> ProcessExecutionReceipt | None:
        return self._last_receipt

    @property
    def last_cleanup_error(self) -> ProcessWorkerTransportError | None:
        """Most recent payload-resource cleanup degradation, if one occurred."""
        return self._last_cleanup_error

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
        try:
            protocol = _control_int(response, "protocol")
            response_node_id = _control_str(response, "node_id")
            response_generation = _control_int(response, "generation")
        except _PayloadProtocolFailure as exc:
            raise ProcessWorkerProtocolError(self.node_id, str(exc)) from exc
        if (
            protocol != _PROTOCOL
            or response_node_id != self.node_id
            or response_generation != self.generation
        ):
            raise ProcessWorkerProtocolError(
                self.node_id,
                f"stale or mismatched process response for request {request_id}",
            )
        if request_id is None:
            if "request_id" in response:
                raise ProcessWorkerProtocolError(
                    self.node_id,
                    "control response unexpectedly carries request identity",
                )
            return
        try:
            response_request_id = _control_int(response, "request_id")
        except _PayloadProtocolFailure as exc:
            raise ProcessWorkerProtocolError(self.node_id, str(exc)) from exc
        if response_request_id != request_id:
            raise ProcessWorkerProtocolError(
                self.node_id,
                f"stale or mismatched process response for request {request_id}",
            )

    def _recv(self, timeout_s: float, request_id: int | None) -> dict[str, Any]:
        if self._conn is None:
            raise ProcessWorkerCrashedError(self.node_id, "worker IPC is closed")
        if not self._conn.poll(timeout_s):
            if not self.is_alive:
                raise ProcessWorkerCrashedError(self.node_id, "worker exited before response")
            raise ProcessWorkerTimeoutError(
                self.node_id,
                f"request {request_id} exceeded {timeout_s:.6f}s hard execution timeout",
            )
        try:
            response = pickle.loads(self._conn.recv_bytes())
        except (EOFError, ConnectionResetError, BrokenPipeError, OSError) as exc:
            raise ProcessWorkerCrashedError(
                self.node_id, "worker IPC closed before response"
            ) from exc
        except Exception as exc:
            raise ProcessWorkerProtocolError(
                self.node_id, f"invalid worker response: {exc}"
            ) from exc
        if type(response) is not dict:
            raise ProcessWorkerProtocolError(
                self.node_id, "worker response is not an exact mapping"
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

    def _cleanup_suffix(self) -> str:
        try:
            self._cleanup_payload_transport()
        except ProcessWorkerTransportError as exc:
            self._last_cleanup_error = exc
            return f"; payload cleanup also failed: {exc}"
        return ""

    def _abort_after_primary_failure(self) -> None:
        """Contain a failed call without allowing cleanup noise to rewrite its cause.

        Failure to prove direct-child death remains authority-critical and is
        therefore allowed to supersede the original operation failure. Payload
        cleanup degradation after child death is retained as secondary evidence
        and retried by later executor-owned close authority.
        """
        try:
            self.abort()
        except ProcessWorkerTerminationError:
            raise
        except ProcessWorkerTransportError as exc:
            self._last_cleanup_error = exc

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
        try:
            child_transport_spec = self._payload_transport.child_spec()
        except _PayloadTransportFailure as exc:
            suffix = self._cleanup_suffix()
            raise ProcessWorkerTransportError(
                self.node_id, f"child payload transport specification failed: {exc}{suffix}"
            ) from exc
        except Exception as exc:
            suffix = self._cleanup_suffix()
            raise ProcessWorkerTransportError(
                self.node_id,
                "child payload transport specification failed: "
                f"{type(exc).__name__}: {exc}{suffix}",
            ) from exc

        try:
            parent, child = self._ctx.Pipe(duplex=True)
        except Exception as exc:
            suffix = self._cleanup_suffix()
            raise ProcessWorkerTransportError(
                self.node_id,
                f"worker IPC creation failed: {type(exc).__name__}: {exc}{suffix}",
            ) from exc

        try:
            process = self._ctx.Process(
                target=_child,
                args=(
                    child,
                    self.operator,
                    self.node_id,
                    self.generation,
                    self._child_transport_factory,
                    child_transport_spec,
                ),
                name=f"{self._process_name_prefix}:{self.node_id}:g{self.generation}",
            )
        except Exception as exc:
            parent.close()
            child.close()
            suffix = self._cleanup_suffix()
            raise ProcessWorkerTransportError(
                self.node_id,
                f"worker process construction failed: {type(exc).__name__}: {exc}{suffix}",
            ) from exc

        self._conn, self._process = parent, process
        try:
            process.start()
        except Exception as exc:
            parent.close()
            self._conn = self._process = None
            suffix = self._cleanup_suffix()
            raise ProcessWorkerSerializationError(
                self.node_id,
                f"operator/start serialization failed: {exc}{suffix}",
            ) from exc
        finally:
            child.close()

        ready = self._recv(self.startup_timeout_s, None)
        try:
            ready_kind = _control_str(ready, "kind")
        except _PayloadProtocolFailure as exc:
            raise ProcessWorkerProtocolError(self.node_id, str(exc)) from exc
        if ready_kind == "transport_error":
            message = ready.get("message")
            error_type = ready.get("error_type")
            if not isinstance(message, str) or not isinstance(error_type, str):
                raise ProcessWorkerProtocolError(
                    self.node_id, "malformed child transport startup error"
                )
            raise ProcessWorkerTransportError(self.node_id, message)
        if ready_kind != "ready":
            raise ProcessWorkerProtocolError(self.node_id, "worker did not become ready")
        self._send_control("heartbeat")
        heartbeat = self._recv(self.startup_timeout_s, None)
        try:
            heartbeat_kind = _control_str(heartbeat, "kind")
        except _PayloadProtocolFailure as exc:
            raise ProcessWorkerProtocolError(self.node_id, str(exc)) from exc
        if heartbeat_kind != "heartbeat":
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

    async def _contain_primary_failure(self) -> None:
        await asyncio.to_thread(self._abort_after_primary_failure)

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
                # Defense in depth: do not make correctness depend on _recv being
                # the only response source. Tests and future transports may wrap
                # or replace it; every admitted call result is re-verified here.
                if type(response) is not dict:
                    raise ProcessWorkerProtocolError(
                        self.node_id, "worker response is not an exact mapping"
                    )
                self._identity(response, request_id)
            except asyncio.CancelledError:
                self._receipt(request_id, "cancelled")
                await asyncio.shield(self._contain_primary_failure())
                raise
            except ProcessWorkerTimeoutError:
                self._receipt(request_id, "timeout")
                await self._contain_primary_failure()
                raise
            except ProcessWorkerCrashedError:
                self._receipt(request_id, "crashed")
                await self._contain_primary_failure()
                raise
            except ProcessWorkerProtocolError:
                self._receipt(request_id, "protocol_error")
                await self._contain_primary_failure()
                raise
            except ProcessWorkerTransportError:
                self._request_id = request_id
                self._receipt(request_id, "transport_error")
                await self._contain_primary_failure()
                raise
            except ProcessWorkerSerializationError:
                self._request_id = request_id
                self._receipt(request_id, "serialization_error")
                await self._contain_primary_failure()
                raise
            except Exception as exc:
                self._request_id = request_id
                self._receipt(request_id, "serialization_error")
                wrapped = ProcessWorkerSerializationError(
                    self.node_id, f"input/request serialization failed: {exc}"
                )
                await self._contain_primary_failure()
                raise wrapped from exc

            try:
                kind = _control_str(response, "kind")
            except _PayloadProtocolFailure as exc:
                error = ProcessWorkerProtocolError(self.node_id, str(exc))
                self._receipt(request_id, "protocol_error")
                await self._contain_primary_failure()
                raise error from exc

            if kind == "result":
                try:
                    result = self._decode_result(response.get("result"), request_id)
                except ProcessWorkerSerializationError:
                    self._receipt(request_id, "serialization_error")
                    await self._contain_primary_failure()
                    raise
                except ProcessWorkerTransportError:
                    self._receipt(request_id, "transport_error")
                    await self._contain_primary_failure()
                    raise
                receipt = ProcessExecutionReceipt(
                    self.node_id, self.generation, request_id, "success"
                )
                self._last_receipt = receipt
                return ProcessCallResult(result, receipt)

            if kind == "error":
                error_type = response.get("error_type")
                message = response.get("message")
                if not isinstance(error_type, str) or not error_type or not isinstance(message, str):
                    error = ProcessWorkerProtocolError(
                        self.node_id, "malformed remote error response"
                    )
                    self._receipt(request_id, "protocol_error")
                    await self._contain_primary_failure()
                    raise error
                self._receipt(request_id, "error", error_type)
                raise ProcessWorkerRemoteError(self.node_id, error_type, message)

            if kind == "serialization_error":
                message = response.get("message")
                if not isinstance(message, str):
                    error = ProcessWorkerProtocolError(
                        self.node_id, "malformed serialization error response"
                    )
                    self._receipt(request_id, "protocol_error")
                    await self._contain_primary_failure()
                    raise error
                self._receipt(request_id, "serialization_error")
                error = ProcessWorkerSerializationError(self.node_id, message)
                await self._contain_primary_failure()
                raise error

            if kind == "transport_error":
                message = response.get("message")
                error_type = response.get("error_type")
                if not isinstance(message, str) or not isinstance(error_type, str):
                    error = ProcessWorkerProtocolError(
                        self.node_id, "malformed transport error response"
                    )
                    self._receipt(request_id, "protocol_error")
                    await self._contain_primary_failure()
                    raise error
                self._receipt(request_id, "transport_error")
                error = ProcessWorkerTransportError(self.node_id, message)
                await self._contain_primary_failure()
                raise error

            self._receipt(request_id, "protocol_error")
            error = ProcessWorkerProtocolError(
                self.node_id, f"unexpected worker response kind {kind!r}"
            )
            await self._contain_primary_failure()
            raise error

    async def heartbeat(self) -> None:
        async with self._lock:
            try:
                await asyncio.to_thread(self._start)
                await asyncio.to_thread(self._send_control, "heartbeat")
                response = await asyncio.to_thread(
                    self._recv, self.startup_timeout_s, None
                )
                try:
                    kind = _control_str(response, "kind")
                except _PayloadProtocolFailure as exc:
                    raise ProcessWorkerProtocolError(self.node_id, str(exc)) from exc
                if kind != "heartbeat":
                    raise ProcessWorkerProtocolError(self.node_id, "worker heartbeat failed")
            except asyncio.CancelledError:
                await asyncio.shield(self._contain_primary_failure())
                raise
            except Exception:
                await self._contain_primary_failure()
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
                    if type(response) is not dict:
                        raise ProcessWorkerProtocolError(
                            self.node_id, "shutdown response is not an exact mapping"
                        )
                    self._identity(response, None)
                    if _control_str(response, "kind") != "shutdown":
                        raise ProcessWorkerProtocolError(
                            self.node_id, "worker shutdown acknowledgement failed"
                        )
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
