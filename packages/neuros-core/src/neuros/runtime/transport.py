"""Shared-memory transport for canonical neurOS runtime payloads.

The transport deliberately keeps control metadata on the multiprocessing pipe
and moves numeric array bytes through a fixed-capacity shared-memory mailbox.
Decoded arrays are materialized into independent local memory before arbitrary
operator code is invoked. This is therefore a shared-memory transport, not a
zero-copy callback contract.

Transport is representation authority, not scientific-value authority. Generic
numeric payloads preserve their existing values, including non-finite floating
values when the surrounding contract permits them. Canonical contracts such as
``SignalFrame`` and ``NeuralWindow`` continue to enforce their own stricter
scientific invariants during construction.
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

import numpy as np

from neuros.contracts import (
    ClockDomain,
    DecoderOutput,
    NeuralWindow,
    QualityFlag,
    SignalFrame,
    TransformEmission,
)

_SCHEMA = "neuros.shared_memory_payload.v1"
_ALIGNMENT = 64
_SUPPORTED_ARRAY_KINDS = frozenset("biufc")


class NeuralTransportError(RuntimeError):
    """Base class for fail-closed neural transport errors."""


class NeuralTransportCapacityError(NeuralTransportError):
    """Raised when a logical payload cannot fit in the declared mailbox."""


class NeuralTransportProtocolError(NeuralTransportError):
    """Raised when a transport manifest is stale, malformed, or inconsistent."""


class NeuralTransportTypeError(NeuralTransportError, TypeError):
    """Raised when an unsupported Python payload crosses shared transport."""


@dataclass(frozen=True, slots=True)
class SharedPayloadEnvelope:
    """Small control-plane description of bytes stored in one mailbox lease."""

    lease_id: int
    bytes_used: int
    manifest: Any

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": _SCHEMA,
            "lease_id": self.lease_id,
            "bytes_used": self.bytes_used,
            "manifest": self.manifest,
        }


def _manifest_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise NeuralTransportProtocolError(
            f"transport field {field_name} must be an exact integer"
        )
    return value


def _manifest_optional_int(value: Any, field_name: str) -> int | None:
    if value is None:
        return None
    return _manifest_int(value, field_name)


def _manifest_str(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise NeuralTransportProtocolError(
            f"transport field {field_name} must be a string"
        )
    return value


def _manifest_real(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NeuralTransportProtocolError(
            f"transport field {field_name} must be a real numeric scalar"
        )
    return float(value)


class _MailboxWriter:
    def __init__(self, buffer: memoryview, capacity_bytes: int) -> None:
        self.buffer = buffer
        self.capacity_bytes = capacity_bytes
        self.offset = 0

    @staticmethod
    def _aligned(offset: int) -> int:
        remainder = offset % _ALIGNMENT
        return offset if remainder == 0 else offset + (_ALIGNMENT - remainder)

    def put_array(self, value: np.ndarray) -> dict[str, Any]:
        array = np.asarray(value)
        if array.dtype.kind not in _SUPPORTED_ARRAY_KINDS:
            raise NeuralTransportTypeError(
                "shared-memory arrays must use boolean or numeric dtype; "
                f"received {array.dtype}"
            )
        array = np.ascontiguousarray(array)
        offset = self._aligned(self.offset)
        end = offset + int(array.nbytes)
        if end > self.capacity_bytes:
            raise NeuralTransportCapacityError(
                f"payload requires {end} bytes but mailbox capacity is "
                f"{self.capacity_bytes} bytes"
            )
        if array.nbytes:
            self.buffer[offset:end] = memoryview(array).cast("B")
        self.offset = end
        return {
            "type": "ndarray",
            "offset": offset,
            "nbytes": int(array.nbytes),
            "dtype": array.dtype.str,
            "shape": list(array.shape),
        }


class _MailboxReader:
    def __init__(self, buffer: memoryview, bytes_used: int) -> None:
        self.buffer = buffer
        self.bytes_used = bytes_used
        self._ranges: list[tuple[int, int]] = []

    def get_array(self, node: Mapping[str, Any]) -> np.ndarray:
        try:
            offset = _manifest_int(node["offset"], "ndarray.offset")
            nbytes = _manifest_int(node["nbytes"], "ndarray.nbytes")
            dtype_value = node["dtype"]
            if not isinstance(dtype_value, str):
                raise NeuralTransportProtocolError(
                    "transport field ndarray.dtype must be a string"
                )
            dtype = np.dtype(dtype_value)
            raw_shape = node["shape"]
            if not isinstance(raw_shape, list):
                raise NeuralTransportProtocolError(
                    "transport field ndarray.shape must be a list"
                )
            shape = tuple(
                _manifest_int(dim, f"ndarray.shape[{index}]")
                for index, dim in enumerate(raw_shape)
            )
        except NeuralTransportProtocolError:
            raise
        except Exception as exc:
            raise NeuralTransportProtocolError(
                f"malformed ndarray transport descriptor: {exc}"
            ) from exc
        if offset < 0 or nbytes < 0 or any(dim < 0 for dim in shape):
            raise NeuralTransportProtocolError("negative shared-memory array geometry")
        if offset % _ALIGNMENT != 0:
            raise NeuralTransportProtocolError(
                f"shared-memory ndarray offset must be {_ALIGNMENT}-byte aligned"
            )
        if dtype.kind not in _SUPPORTED_ARRAY_KINDS:
            raise NeuralTransportProtocolError(
                f"unsupported shared-memory dtype {dtype}"
            )
        expected = int(math.prod(shape)) * int(dtype.itemsize)
        if expected != nbytes:
            raise NeuralTransportProtocolError(
                "shared-memory ndarray byte length does not match dtype/shape"
            )
        end = offset + nbytes
        if end > self.bytes_used:
            raise NeuralTransportProtocolError(
                "shared-memory ndarray exceeds declared payload boundary"
            )
        for start, stop in self._ranges:
            if offset < stop and start < end:
                raise NeuralTransportProtocolError(
                    "shared-memory ndarray regions overlap"
                )
        self._ranges.append((offset, end))
        view = np.ndarray(shape=shape, dtype=dtype, buffer=self.buffer, offset=offset)
        return np.array(view, copy=True, subok=False)


def _encode(value: Any, writer: _MailboxWriter) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return {"type": "scalar", "value": value}
    if isinstance(value, complex):
        return {
            "type": "complex_scalar",
            "real": float(value.real),
            "imag": float(value.imag),
        }
    if isinstance(value, np.generic):
        return _encode(value.item(), writer)
    if isinstance(value, np.ndarray):
        return writer.put_array(value)
    if isinstance(value, SignalFrame):
        return {
            "type": "signal_frame",
            "stream_id": value.stream_id,
            "sequence_id": value.sequence_id,
            "data": _encode(value.data, writer),
            "sample_rate_hz": value.sample_rate_hz,
            "host_receive_time_ns": value.host_receive_time_ns,
            "device_time_ns": value.device_time_ns,
            "synchronized_time_ns": value.synchronized_time_ns,
            "clock_domain": value.clock_domain.value,
            "quality": int(value.quality),
            "metadata": _encode(dict(value.metadata), writer),
        }
    if isinstance(value, NeuralWindow):
        return {
            "type": "neural_window",
            "stream_id": value.stream_id,
            "window_id": value.window_id,
            "data": _encode(value.data, writer),
            "sample_rate_hz": value.sample_rate_hz,
            "start_time_ns": value.start_time_ns,
            "end_time_ns": value.end_time_ns,
            "channel_names": _encode(value.channel_names, writer),
            "source_sequence_ids": _encode(value.source_sequence_ids, writer),
            "clock_domain": value.clock_domain.value,
            "quality": int(value.quality),
            "metadata": _encode(dict(value.metadata), writer),
        }
    if isinstance(value, DecoderOutput):
        return {
            "type": "decoder_output",
            "prediction": _encode(value.prediction, writer),
            "confidence": _encode(value.confidence, writer),
            "uncertainty": _encode(value.uncertainty, writer),
            "probabilities": _encode(value.probabilities, writer),
            "logits": _encode(value.logits, writer),
            "embedding": _encode(value.embedding, writer),
            "model_id": _encode(value.model_id, writer),
            "model_version": _encode(value.model_version, writer),
            "inference_time_ns": _encode(value.inference_time_ns, writer),
            "metadata": _encode(dict(value.metadata), writer),
        }
    if isinstance(value, TransformEmission):
        return {"type": "transform_emission", "items": _encode(value.items, writer)}
    if isinstance(value, tuple):
        return {"type": "tuple", "items": [_encode(item, writer) for item in value]}
    if isinstance(value, list):
        return {"type": "list", "items": [_encode(item, writer) for item in value]}
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise NeuralTransportTypeError("transport mapping keys must be strings")
        return {
            "type": "mapping",
            "items": [[key, _encode(value[key], writer)] for key in sorted(keys)],
        }
    raise NeuralTransportTypeError(
        "unsupported shared-memory payload type "
        f"{type(value).__module__}.{type(value).__qualname__}"
    )


def _decode_items(node: Mapping[str, Any], node_type: str) -> list[Any]:
    items = node.get("items")
    if not isinstance(items, list):
        raise NeuralTransportProtocolError(
            f"{node_type} transport manifest items must be a list"
        )
    return items


def _decode(node: Any, reader: _MailboxReader) -> Any:
    if not isinstance(node, Mapping):
        raise NeuralTransportProtocolError("transport manifest node is not a mapping")
    node_type = node.get("type")
    if node_type == "scalar":
        value = node.get("value")
        if value is not None and not isinstance(value, (str, bool, int, float)):
            raise NeuralTransportProtocolError("invalid transport scalar")
        return value
    if node_type == "complex_scalar":
        try:
            return complex(
                _manifest_real(node["real"], "complex.real"),
                _manifest_real(node["imag"], "complex.imag"),
            )
        except NeuralTransportProtocolError:
            raise
        except Exception as exc:
            raise NeuralTransportProtocolError("invalid complex transport scalar") from exc
    if node_type == "ndarray":
        return reader.get_array(node)
    if node_type == "tuple":
        return tuple(_decode(item, reader) for item in _decode_items(node, "tuple"))
    if node_type == "list":
        return [_decode(item, reader) for item in _decode_items(node, "list")]
    if node_type == "mapping":
        result: dict[str, Any] = {}
        items = _decode_items(node, "mapping")
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2 or not isinstance(pair[0], str):
                raise NeuralTransportProtocolError("malformed mapping manifest item")
            key = pair[0]
            if key in result:
                raise NeuralTransportProtocolError("duplicate transport mapping key")
            result[key] = _decode(pair[1], reader)
        return result
    if node_type == "signal_frame":
        try:
            return SignalFrame(
                stream_id=_manifest_str(node["stream_id"], "signal_frame.stream_id"),
                sequence_id=_manifest_int(node["sequence_id"], "signal_frame.sequence_id"),
                data=_decode(node["data"], reader),
                sample_rate_hz=_manifest_real(
                    node["sample_rate_hz"], "signal_frame.sample_rate_hz"
                ),
                host_receive_time_ns=_manifest_int(
                    node["host_receive_time_ns"], "signal_frame.host_receive_time_ns"
                ),
                device_time_ns=_manifest_optional_int(
                    node.get("device_time_ns"), "signal_frame.device_time_ns"
                ),
                synchronized_time_ns=_manifest_optional_int(
                    node.get("synchronized_time_ns"),
                    "signal_frame.synchronized_time_ns",
                ),
                clock_domain=ClockDomain(
                    _manifest_str(node["clock_domain"], "signal_frame.clock_domain")
                ),
                quality=QualityFlag(
                    _manifest_int(node["quality"], "signal_frame.quality")
                ),
                metadata=_decode(node["metadata"], reader),
            )
        except NeuralTransportProtocolError:
            raise
        except Exception as exc:
            raise NeuralTransportProtocolError(
                f"invalid SignalFrame transport payload: {exc}"
            ) from exc
    if node_type == "neural_window":
        try:
            channel_names = _decode(node["channel_names"], reader)
            source_sequence_ids = _decode(node["source_sequence_ids"], reader)
            if not isinstance(channel_names, tuple):
                raise NeuralTransportProtocolError(
                    "NeuralWindow channel_names must decode to tuple"
                )
            if not isinstance(source_sequence_ids, tuple):
                raise NeuralTransportProtocolError(
                    "NeuralWindow source_sequence_ids must decode to tuple"
                )
            return NeuralWindow(
                stream_id=_manifest_str(node["stream_id"], "neural_window.stream_id"),
                window_id=_manifest_int(node["window_id"], "neural_window.window_id"),
                data=_decode(node["data"], reader),
                sample_rate_hz=_manifest_real(
                    node["sample_rate_hz"], "neural_window.sample_rate_hz"
                ),
                start_time_ns=_manifest_int(
                    node["start_time_ns"], "neural_window.start_time_ns"
                ),
                end_time_ns=_manifest_int(
                    node["end_time_ns"], "neural_window.end_time_ns"
                ),
                channel_names=channel_names,
                source_sequence_ids=source_sequence_ids,
                clock_domain=ClockDomain(
                    _manifest_str(node["clock_domain"], "neural_window.clock_domain")
                ),
                quality=QualityFlag(
                    _manifest_int(node["quality"], "neural_window.quality")
                ),
                metadata=_decode(node["metadata"], reader),
            )
        except NeuralTransportProtocolError:
            raise
        except Exception as exc:
            raise NeuralTransportProtocolError(
                f"invalid NeuralWindow transport payload: {exc}"
            ) from exc
    if node_type == "decoder_output":
        try:
            return DecoderOutput(
                prediction=_decode(node["prediction"], reader),
                confidence=_decode(node["confidence"], reader),
                uncertainty=_decode(node["uncertainty"], reader),
                probabilities=_decode(node["probabilities"], reader),
                logits=_decode(node["logits"], reader),
                embedding=_decode(node["embedding"], reader),
                model_id=_decode(node["model_id"], reader),
                model_version=_decode(node["model_version"], reader),
                inference_time_ns=_decode(node["inference_time_ns"], reader),
                metadata=_decode(node["metadata"], reader),
            )
        except NeuralTransportProtocolError:
            raise
        except Exception as exc:
            raise NeuralTransportProtocolError(
                f"invalid DecoderOutput transport payload: {exc}"
            ) from exc
    if node_type == "transform_emission":
        items = _decode(node["items"], reader)
        if not isinstance(items, tuple):
            raise NeuralTransportProtocolError("TransformEmission items must decode to tuple")
        try:
            return TransformEmission(items)
        except Exception as exc:
            raise NeuralTransportProtocolError(
                f"invalid TransformEmission transport payload: {exc}"
            ) from exc
    raise NeuralTransportProtocolError(f"unknown transport manifest node type {node_type!r}")


class SharedMemoryMailbox:
    """One fixed-capacity shared-memory mailbox with explicit lease identity."""

    def __init__(
        self,
        capacity_bytes: int,
        *,
        name: str | None = None,
        create: bool = True,
        owner: bool | None = None,
    ) -> None:
        if isinstance(capacity_bytes, bool) or not isinstance(capacity_bytes, int):
            raise TypeError("capacity_bytes must be an integer")
        if capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        self.capacity_bytes = capacity_bytes
        self._owner = create if owner is None else bool(owner)
        self._shm = shared_memory.SharedMemory(
            name=name,
            create=create,
            size=capacity_bytes if create else 0,
        )
        if self._shm.size < capacity_bytes:
            self._shm.close()
            raise NeuralTransportProtocolError(
                "attached shared-memory region is smaller than declared capacity"
            )
        self._closed = False
        self._unlinked = False

    @classmethod
    def attach(cls, name: str, capacity_bytes: int) -> "SharedMemoryMailbox":
        return cls(capacity_bytes, name=name, create=False, owner=False)

    @property
    def name(self) -> str:
        return self._shm.name

    def encode(self, value: Any, *, lease_id: int) -> dict[str, Any]:
        if self._closed:
            raise NeuralTransportError("shared-memory mailbox is closed")
        if isinstance(lease_id, bool) or not isinstance(lease_id, int) or lease_id <= 0:
            raise ValueError("lease_id must be a positive integer")
        writer = _MailboxWriter(self._shm.buf, self.capacity_bytes)
        manifest = _encode(value, writer)
        return SharedPayloadEnvelope(lease_id, writer.offset, manifest).as_dict()

    def decode(self, envelope: Any, *, expected_lease_id: int) -> Any:
        if self._closed:
            raise NeuralTransportError("shared-memory mailbox is closed")
        if not isinstance(envelope, Mapping):
            raise NeuralTransportProtocolError("transport envelope is not a mapping")
        if envelope.get("schema") != _SCHEMA:
            raise NeuralTransportProtocolError("shared-memory payload schema mismatch")
        try:
            lease_id = _manifest_int(envelope["lease_id"], "lease_id")
            bytes_used = _manifest_int(envelope["bytes_used"], "bytes_used")
        except NeuralTransportProtocolError:
            raise
        except Exception as exc:
            raise NeuralTransportProtocolError("malformed transport envelope identity") from exc
        if lease_id != expected_lease_id:
            raise NeuralTransportProtocolError(
                f"stale shared-memory lease {lease_id}; expected {expected_lease_id}"
            )
        if bytes_used < 0 or bytes_used > self.capacity_bytes:
            raise NeuralTransportProtocolError(
                "shared-memory payload bytes_used exceeds mailbox capacity"
            )
        reader = _MailboxReader(self._shm.buf, bytes_used)
        return _decode(envelope.get("manifest"), reader)

    def close(self) -> None:
        if not self._closed:
            self._shm.close()
            self._closed = True

    def unlink(self) -> None:
        if not self._owner:
            raise NeuralTransportError("only the mailbox owner may unlink shared memory")
        if self._unlinked:
            return
        try:
            self._shm.unlink()
        except FileNotFoundError:
            pass
        self._unlinked = True

    def close_and_unlink(self) -> None:
        if not self._owner:
            self.close()
            return
        self.unlink()
        self.close()
