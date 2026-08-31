"""Shared-memory payload adapter for the common persistent process authority.

The process lifecycle lives in :mod:`neuros.runtime.process_worker`. This module
owns only shared-memory payload representation and resource ownership. Array
bytes cross fixed-capacity parent-owned mailboxes while the multiprocessing pipe
remains the control plane. Decoded callback inputs are materialized into local
memory, so this is not a zero-copy callback contract.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .process_worker import (
    PersistentProcessWorker,
    _PayloadProtocolFailure,
    _PayloadTransportFailure,
)
from .transport import NeuralTransportError, SharedMemoryMailbox


@dataclass(frozen=True, slots=True)
class _SharedMemoryChildSpec:
    request_name: str
    request_capacity_bytes: int
    response_name: str
    response_capacity_bytes: int


class _SharedMemoryParentTransport:
    def __init__(self, request_capacity_bytes: int, response_capacity_bytes: int) -> None:
        self.request_capacity_bytes = request_capacity_bytes
        self.response_capacity_bytes = response_capacity_bytes
        self._request_mailbox: SharedMemoryMailbox | None = None
        self._response_mailbox: SharedMemoryMailbox | None = None

    @property
    def shared_memory_names(self) -> dict[str, str] | None:
        if self._request_mailbox is None or self._response_mailbox is None:
            return None
        return {
            "request": self._request_mailbox.name,
            "response": self._response_mailbox.name,
        }

    def prepare(self) -> None:
        if self._request_mailbox is not None or self._response_mailbox is not None:
            if self._request_mailbox is None or self._response_mailbox is None:
                raise _PayloadTransportFailure(
                    "shared-memory mailbox ownership is incomplete"
                )
            return

        try:
            request_mailbox = SharedMemoryMailbox(self.request_capacity_bytes)
        except Exception as exc:
            raise _PayloadTransportFailure(
                "request shared-memory mailbox creation failed: "
                f"{type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
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
            raise _PayloadTransportFailure(
                "response shared-memory mailbox creation failed: "
                f"{type(exc).__name__}: {exc}{cleanup_suffix}",
                error_type=type(exc).__name__,
            ) from exc

        self._request_mailbox = request_mailbox
        self._response_mailbox = response_mailbox

    def child_spec(self) -> _SharedMemoryChildSpec:
        if self._request_mailbox is None or self._response_mailbox is None:
            raise _PayloadTransportFailure(
                "shared-memory transport must be prepared before child startup"
            )
        return _SharedMemoryChildSpec(
            request_name=self._request_mailbox.name,
            request_capacity_bytes=self.request_capacity_bytes,
            response_name=self._response_mailbox.name,
            response_capacity_bytes=self.response_capacity_bytes,
        )

    def encode_request(self, value: Any, lease_id: int) -> dict[str, Any]:
        if self._request_mailbox is None:
            raise _PayloadTransportFailure("request mailbox is unavailable")
        try:
            return self._request_mailbox.encode(value, lease_id=lease_id)
        except NeuralTransportError as exc:
            raise _PayloadTransportFailure(
                f"input transport failed: {exc}", error_type=type(exc).__name__
            ) from exc
        except Exception as exc:
            raise _PayloadTransportFailure(
                f"input transport encoding failed: {type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            ) from exc

    def decode_result(self, payload: Any, lease_id: int) -> Any:
        if self._response_mailbox is None:
            raise _PayloadTransportFailure("response mailbox is unavailable")
        try:
            return self._response_mailbox.decode(
                payload, expected_lease_id=lease_id
            )
        except NeuralTransportError as exc:
            raise _PayloadTransportFailure(
                f"result transport failed: {exc}", error_type=type(exc).__name__
            ) from exc
        except Exception as exc:
            raise _PayloadTransportFailure(
                f"result transport decoding failed: {type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            ) from exc

    def cleanup(self) -> None:
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
            raise _PayloadTransportFailure(
                "shared-memory cleanup authority failed: " + "; ".join(errors)
            )


class _SharedMemoryChildTransport:
    def __init__(
        self,
        request_mailbox: SharedMemoryMailbox,
        response_mailbox: SharedMemoryMailbox,
    ) -> None:
        self._request_mailbox = request_mailbox
        self._response_mailbox = response_mailbox

    def decode_request(self, payload: Any, lease_id: int) -> Any:
        try:
            return self._request_mailbox.decode(
                payload, expected_lease_id=lease_id
            )
        except NeuralTransportError as exc:
            raise _PayloadTransportFailure(
                f"input transport failed: {exc}", error_type=type(exc).__name__
            ) from exc
        except Exception as exc:
            raise _PayloadTransportFailure(
                f"input transport decoding failed: {type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            ) from exc

    def encode_result(self, value: Any, lease_id: int) -> dict[str, Any]:
        try:
            return self._response_mailbox.encode(value, lease_id=lease_id)
        except NeuralTransportError as exc:
            raise _PayloadTransportFailure(
                f"result transport failed: {exc}", error_type=type(exc).__name__
            ) from exc
        except Exception as exc:
            raise _PayloadTransportFailure(
                f"result transport encoding failed: {type(exc).__name__}: {exc}",
                error_type=type(exc).__name__,
            ) from exc

    def close(self) -> None:
        errors: list[str] = []
        for label, mailbox in (
            ("request", self._request_mailbox),
            ("response", self._response_mailbox),
        ):
            try:
                mailbox.close()
            except Exception as exc:
                errors.append(f"{label}: {type(exc).__name__}: {exc}")
        if errors:
            raise _PayloadTransportFailure(
                "child shared-memory close failed: " + "; ".join(errors)
            )


def _make_shared_memory_child_transport(
    spec: Any,
) -> _SharedMemoryChildTransport:
    if not isinstance(spec, _SharedMemoryChildSpec):
        raise _PayloadProtocolFailure("invalid shared-memory child transport spec")

    request_mailbox: SharedMemoryMailbox | None = None
    try:
        request_mailbox = SharedMemoryMailbox.attach(
            spec.request_name, spec.request_capacity_bytes
        )
        response_mailbox = SharedMemoryMailbox.attach(
            spec.response_name, spec.response_capacity_bytes
        )
    except Exception as exc:
        if request_mailbox is not None:
            try:
                request_mailbox.close()
            except Exception:
                pass
        raise _PayloadTransportFailure(
            f"child shared-memory attach failed: {type(exc).__name__}: {exc}",
            error_type=type(exc).__name__,
        ) from exc

    return _SharedMemoryChildTransport(request_mailbox, response_mailbox)


class SharedMemoryProcessWorker(PersistentProcessWorker):
    """Persistent process worker using parent-owned shared-memory mailboxes."""

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
        for field_name, capacity in (
            ("request_capacity_bytes", request_capacity_bytes),
            ("response_capacity_bytes", response_capacity_bytes),
        ):
            if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
                raise ValueError(f"{field_name} must be a positive integer")

        self.request_capacity_bytes = request_capacity_bytes
        self.response_capacity_bytes = response_capacity_bytes
        super().__init__(
            node_id,
            operator,
            execution_timeout_s=execution_timeout_s,
            generation=generation,
            startup_timeout_s=startup_timeout_s,
            termination_grace_s=termination_grace_s,
            _payload_transport=_SharedMemoryParentTransport(
                request_capacity_bytes, response_capacity_bytes
            ),
            _child_transport_factory=_make_shared_memory_child_transport,
            _process_name_prefix="neuros-shm-process",
        )
