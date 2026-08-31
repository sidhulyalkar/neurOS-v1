"""Persistent direct-child authority for process-executed runtime nodes.

Trusted Python objects cross a multiprocessing boundary, so this is fault
isolation, not a security sandbox. Termination authority covers only the direct
child created here, never arbitrary descendants.
"""
from __future__ import annotations

import asyncio
import inspect
import multiprocessing as mp
import pickle
from dataclasses import dataclass
from multiprocessing.connection import Connection
from typing import Any

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


def _run_callback(func: Any, item: Any) -> Any:
    value = func(item)
    if not inspect.isawaitable(value):
        return value

    async def _await() -> Any:
        return await value

    return asyncio.run(_await())


def _send(conn: Connection, envelope: dict[str, Any]) -> None:
    conn.send_bytes(pickle.dumps(envelope, protocol=pickle.HIGHEST_PROTOCOL))


def _child(conn: Connection, operator: Any, node_id: str, generation: int) -> None:
    base = {"protocol": _PROTOCOL, "node_id": node_id, "generation": generation}
    try:
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
                _send(conn, {**base, "kind": "protocol_error", "message": "identity mismatch"})
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
            item_bytes = request.get("item")
            call_base = {**base, "request_id": request_id}
            if (
                command != "call"
                or not isinstance(request_id, int)
                or request_id <= 0
                or not isinstance(method, str)
                or not isinstance(item_bytes, bytes)
            ):
                _send(conn, {**call_base, "kind": "protocol_error", "message": "malformed call"})
                return
            try:
                item = pickle.loads(item_bytes)
            except Exception as exc:
                _send(conn, {**call_base, "kind": "serialization_error", "message": f"input deserialization failed: {exc}"})
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
                result_bytes = pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
            except Exception as exc:
                _send(conn, {**call_base, "kind": "serialization_error", "message": f"result serialization failed: {exc}"})
                return
            _send(conn, {**call_base, "kind": "result", "result": result_bytes})
    finally:
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
    ) -> None:
        if not node_id:
            raise ValueError("node_id must be non-empty")
        if min(execution_timeout_s, startup_timeout_s, termination_grace_s) <= 0:
            raise ValueError("worker timeouts must be positive")
        if generation < 0:
            raise ValueError("generation must be non-negative")
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

    @property
    def last_receipt(self) -> ProcessExecutionReceipt | None:
        return self._last_receipt

    @property
    def pid(self) -> int | None:
        return None if self._process is None else self._process.pid

    @property
    def is_alive(self) -> bool:
        return bool(self._process is not None and self._process.is_alive())

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
                raise ProcessWorkerCrashedError(self.node_id, "worker exited before response")
            raise ProcessWorkerTimeoutError(
                self.node_id,
                f"request {request_id} exceeded {timeout_s:.6f}s hard execution timeout",
            )
        try:
            response = pickle.loads(self._conn.recv_bytes())
        except EOFError as exc:
            raise ProcessWorkerCrashedError(self.node_id, "worker EOF before response") from exc
        except Exception as exc:
            raise ProcessWorkerProtocolError(self.node_id, f"invalid worker response: {exc}") from exc
        if not isinstance(response, dict):
            raise ProcessWorkerProtocolError(self.node_id, "worker response is not a mapping")
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

    def _start(self) -> None:
        if self._terminal:
            raise ProcessWorkerError(self.node_id, "worker is terminal; create a new generation")
        if self._process is not None:
            if self._process.is_alive():
                return
            raise ProcessWorkerCrashedError(self.node_id, "worker exited")
        parent, child = self._ctx.Pipe(duplex=True)
        process = self._ctx.Process(
            target=_child,
            args=(child, self.operator, self.node_id, self.generation),
            name=f"neuros-process:{self.node_id}:g{self.generation}",
        )
        self._conn, self._process = parent, process
        try:
            process.start()
        except Exception as exc:
            parent.close()
            self._conn = self._process = None
            raise ProcessWorkerSerializationError(
                self.node_id, f"operator/start serialization failed: {exc}"
            ) from exc
        finally:
            child.close()
        ready = self._recv(self.startup_timeout_s, None)
        if ready.get("kind") != "ready":
            raise ProcessWorkerProtocolError(self.node_id, "worker did not become ready")
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
                item_bytes = pickle.dumps(item, protocol=pickle.HIGHEST_PROTOCOL)
                if self._conn is None:
                    raise ProcessWorkerCrashedError(self.node_id, "worker IPC is closed")
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
                            "item": item_bytes,
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
                if not isinstance(response, dict):
                    raise ProcessWorkerProtocolError(
                        self.node_id, "worker response is not a mapping"
                    )
                self._identity(response, request_id)
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
                    result = pickle.loads(response["result"])
                except Exception as exc:
                    self._receipt(request_id, "serialization_error")
                    await asyncio.to_thread(self.abort)
                    raise ProcessWorkerSerializationError(
                        self.node_id, f"result deserialization failed: {exc}"
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
                    self.node_id, error_type, str(response.get("message") or "")
                )
            if kind == "serialization_error":
                self._receipt(request_id, "serialization_error")
                await asyncio.to_thread(self.abort)
                raise ProcessWorkerSerializationError(
                    self.node_id, str(response.get("message") or "serialization failure")
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
                    raise ProcessWorkerProtocolError(self.node_id, "worker heartbeat failed")
            except asyncio.CancelledError:
                await asyncio.shield(asyncio.to_thread(self.abort))
                raise
            except Exception:
                await asyncio.to_thread(self.abort)
                raise

    def abort(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        self._conn = self._process = None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(self.termination_grace_s)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(self.termination_grace_s)
        if not process.is_alive():
            process.close()

    def close(self) -> None:
        self._terminal = True
        conn, process = self._conn, self._process
        if process is None:
            if conn is not None:
                conn.close()
            self._conn = None
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
        if process.is_alive():
            process.terminate()
            process.join(self.termination_grace_s)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(self.termination_grace_s)
        if conn is not None:
            conn.close()
        self._conn = self._process = None
        if not process.is_alive():
            process.close()
