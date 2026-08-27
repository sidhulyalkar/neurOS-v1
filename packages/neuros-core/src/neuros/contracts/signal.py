"""Canonical neural signal contracts used across neurOS.

These dataclasses intentionally live in ``neuros-core`` so drivers, runtimes,
models, storage backends, and ORION can exchange neural data without importing
one another's concrete implementations.

The contracts are deliberately fail-closed at the software boundary. They do
not claim that a physical device clock, sampling rate, or signal is accurate;
they ensure that the *representation* of those observations is internally
unambiguous and cannot be silently mutated after construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from enum import Enum, IntFlag, auto
from numbers import Integral, Real
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from numpy.typing import NDArray


class ClockDomain(str, Enum):
    """Clock used as the authoritative timestamp for a frame."""

    DEVICE = "device"
    HOST_MONOTONIC = "host_monotonic"
    SYNCHRONIZED = "synchronized"
    UNKNOWN = "unknown"


class QualityFlag(IntFlag):
    """Composable signal-quality flags."""

    GOOD = 0
    DROPPED_SAMPLES = auto()
    CLOCK_UNCERTAIN = auto()
    SATURATED = auto()
    CLIPPED = auto()
    DISCONNECTED_CHANNEL = auto()
    ARTIFACT_SUSPECTED = auto()


def _nonempty_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _finite_positive_real(value: Any, *, field_name: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{field_name} must be a real numeric scalar")
    resolved = float(value)
    if not math.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return resolved


def _nonnegative_integer(value: Any, *, field_name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{field_name} must be an integer")
    resolved = int(value)
    if resolved < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return resolved


def _optional_nonnegative_integer(value: Any, *, field_name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_integer(value, field_name=field_name)


def _freeze_metadata(value: Any, *, path: str = "metadata") -> Any:
    """Recursively detach provenance values from caller-owned mutable state.

    The metadata contract intentionally remains JSON-like. That is stricter
    than accepting arbitrary Python objects, but it guarantees that a static
    stream identity has the same scientific meaning before and after archival
    serialization. NumPy arrays are accepted as convenient input and frozen as
    nested tuples of values; dtype-specific identity belongs in a dedicated
    typed contract rather than an incidental metadata container.
    """

    if isinstance(value, Enum):
        return _freeze_metadata(value.value, path=path)
    if isinstance(value, np.generic):
        return _freeze_metadata(value.item(), path=path)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, bytes):
        raise TypeError(f"{path} cannot contain bytes; encode them explicitly as text")
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "O":
            raise TypeError(f"{path} cannot contain object-dtype arrays")
        if value.dtype.kind in "fc" and not np.isfinite(value).all():
            raise ValueError(f"{path} contains a non-finite array")
        return _freeze_metadata(value.tolist(), path=path)
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            frozen[key] = _freeze_metadata(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_metadata(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if isinstance(value, (set, frozenset)):
        raise TypeError(f"{path} cannot contain unordered set values")
    raise TypeError(
        f"{path} contains unsupported value type {type(value).__module__}."
        f"{type(value).__qualname__}; use deterministic provenance primitives"
    )


def _canonical_value(value: Any) -> Any:
    """Convert frozen contract values into deterministic JSON values."""

    if isinstance(value, Enum):
        return _canonical_value(value.value)
    if isinstance(value, np.generic):
        return _canonical_value(value.item())
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("canonical identity cannot contain NaN or infinity")
        return value
    if isinstance(value, np.ndarray):
        return _canonical_value(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _canonical_value(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise TypeError(f"Unsupported canonical identity value: {type(value)!r}")


def _sha256_identity(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _canonical_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class StreamDescriptor:
    """Static metadata describing a neural or behavioral stream.

    ``sample_rate_hz`` is the declared/nominal stream rate. Evidence about a
    measured physical clock, drift, or effective sample interval belongs in
    timing/qualification evidence rather than being inferred from this field.
    """

    stream_id: str
    modality: str
    sample_rate_hz: float
    channel_names: tuple[str, ...] = ()
    channel_types: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    device: str | None = None
    manufacturer: str | None = None
    clock_domain: ClockDomain = ClockDomain.UNKNOWN
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _nonempty_string(self.stream_id, field_name="stream_id"))
        object.__setattr__(self, "modality", _nonempty_string(self.modality, field_name="modality"))
        object.__setattr__(
            self,
            "sample_rate_hz",
            _finite_positive_real(self.sample_rate_hz, field_name="sample_rate_hz"),
        )

        names = tuple(self.channel_names)
        types = tuple(self.channel_types)
        units = tuple(self.units)
        for index, name in enumerate(names):
            _nonempty_string(name, field_name=f"channel_names[{index}]")
        if len(set(names)) != len(names):
            raise ValueError("channel_names must be unique")
        if types and len(types) != len(names):
            raise ValueError("channel_types must match channel_names length")
        if units and len(units) != len(names):
            raise ValueError("units must match channel_names length")
        for index, channel_type in enumerate(types):
            _nonempty_string(channel_type, field_name=f"channel_types[{index}]")
        for index, unit in enumerate(units):
            _nonempty_string(unit, field_name=f"units[{index}]")
        object.__setattr__(self, "channel_names", names)
        object.__setattr__(self, "channel_types", types)
        object.__setattr__(self, "units", units)

        if self.device is not None:
            _nonempty_string(self.device, field_name="device")
        if self.manufacturer is not None:
            _nonempty_string(self.manufacturer, field_name="manufacturer")
        object.__setattr__(self, "clock_domain", ClockDomain(self.clock_domain))
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def nominal_sample_rate_hz(self) -> float:
        """Explicit alias emphasizing that the descriptor rate is nominal."""

        return self.sample_rate_hz

    def identity_payload(self) -> dict[str, Any]:
        """Return the canonical static stream identity used for provenance."""

        return {
            "schema": "neuros.stream_descriptor.v1",
            "stream_id": self.stream_id,
            "modality": self.modality,
            "sample_rate_hz": self.sample_rate_hz,
            "channel_names": self.channel_names,
            "channel_types": self.channel_types,
            "units": self.units,
            "device": self.device,
            "manufacturer": self.manufacturer,
            "clock_domain": self.clock_domain.value,
            "metadata": self.metadata,
        }

    def fingerprint(self) -> str:
        """SHA-256 identity over canonical static descriptor metadata."""

        return _sha256_identity(self.identity_payload())


@dataclass(frozen=True, slots=True)
class SignalFrame:
    """A timestamped chunk of neural data with explicit clock semantics.

    Sample buffers are copied at construction and stored read-only. This makes
    a frame an immutable software observation boundary rather than a view into
    caller-owned mutable memory.
    """

    stream_id: str
    sequence_id: int
    data: NDArray[np.generic]
    sample_rate_hz: float
    host_receive_time_ns: int
    device_time_ns: int | None = None
    synchronized_time_ns: int | None = None
    clock_domain: ClockDomain = ClockDomain.UNKNOWN
    quality: QualityFlag = QualityFlag.GOOD
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "stream_id", _nonempty_string(self.stream_id, field_name="stream_id"))
        object.__setattr__(
            self,
            "sequence_id",
            _nonnegative_integer(self.sequence_id, field_name="sequence_id"),
        )
        object.__setattr__(
            self,
            "sample_rate_hz",
            _finite_positive_real(self.sample_rate_hz, field_name="sample_rate_hz"),
        )
        object.__setattr__(
            self,
            "host_receive_time_ns",
            _nonnegative_integer(self.host_receive_time_ns, field_name="host_receive_time_ns"),
        )
        object.__setattr__(
            self,
            "device_time_ns",
            _optional_nonnegative_integer(self.device_time_ns, field_name="device_time_ns"),
        )
        object.__setattr__(
            self,
            "synchronized_time_ns",
            _optional_nonnegative_integer(
                self.synchronized_time_ns, field_name="synchronized_time_ns"
            ),
        )

        domain = ClockDomain(self.clock_domain)
        if domain is ClockDomain.DEVICE and self.device_time_ns is None:
            raise ValueError("clock_domain='device' requires device_time_ns")
        if domain is ClockDomain.SYNCHRONIZED and self.synchronized_time_ns is None:
            raise ValueError("clock_domain='synchronized' requires synchronized_time_ns")
        object.__setattr__(self, "clock_domain", domain)

        quality_value = _nonnegative_integer(self.quality, field_name="quality")
        object.__setattr__(self, "quality", QualityFlag(quality_value))

        arr = np.array(self.data, copy=True, subok=False)
        if arr.ndim == 0:
            raise ValueError("SignalFrame.data must have at least one dimension")
        if arr.size == 0:
            raise ValueError("SignalFrame.data cannot be empty")
        if arr.dtype.kind not in "biufc":
            raise TypeError(
                "SignalFrame.data must use a boolean or numeric dtype; "
                f"received {arr.dtype}"
            )
        if arr.dtype.kind in "fc" and not np.isfinite(arr).all():
            raise ValueError(
                "SignalFrame.data must be finite; represent dropouts/artifacts with "
                "explicit samples and QualityFlag provenance"
            )
        arr.setflags(write=False)
        object.__setattr__(self, "data", arr)
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))

    @property
    def timestamp_ns(self) -> int:
        """Authoritative timestamp in the declared clock domain.

        ``UNKNOWN`` retains the historical best-available fallback without
        pretending that the resulting value belongs to a qualified clock.
        """

        if self.clock_domain is ClockDomain.SYNCHRONIZED:
            assert self.synchronized_time_ns is not None
            return self.synchronized_time_ns
        if self.clock_domain is ClockDomain.DEVICE:
            assert self.device_time_ns is not None
            return self.device_time_ns
        if self.clock_domain is ClockDomain.HOST_MONOTONIC:
            return self.host_receive_time_ns
        if self.synchronized_time_ns is not None:
            return self.synchronized_time_ns
        if self.device_time_ns is not None:
            return self.device_time_ns
        return self.host_receive_time_ns

    @property
    def timestamp_seconds(self) -> float:
        return self.timestamp_ns / 1_000_000_000.0

    @classmethod
    def from_legacy(
        cls,
        *,
        stream_id: str,
        sequence_id: int,
        timestamp_seconds: float,
        data: NDArray[np.generic],
        sample_rate_hz: float,
        clock_domain: ClockDomain = ClockDomain.UNKNOWN,
        metadata: Mapping[str, Any] | None = None,
    ) -> "SignalFrame":
        """Convert the legacy ``(timestamp_seconds, ndarray)`` representation.

        The legacy scalar timestamp has no independently measured uncertainty.
        Its destination field therefore follows the explicitly supplied clock
        domain. ``UNKNOWN`` keeps it as an unqualified source/device timestamp.
        """

        if isinstance(timestamp_seconds, (bool, np.bool_)) or not isinstance(
            timestamp_seconds, Real
        ):
            raise TypeError("timestamp_seconds must be a real numeric scalar")
        timestamp_value = float(timestamp_seconds)
        if not math.isfinite(timestamp_value) or timestamp_value < 0:
            raise ValueError("timestamp_seconds must be finite and >= 0")
        timestamp_ns = int(round(timestamp_value * 1_000_000_000))
        domain = ClockDomain(clock_domain)
        host_receive_time_ns = time.monotonic_ns()
        device_time_ns: int | None = None
        synchronized_time_ns: int | None = None
        if domain is ClockDomain.HOST_MONOTONIC:
            host_receive_time_ns = timestamp_ns
        elif domain is ClockDomain.SYNCHRONIZED:
            synchronized_time_ns = timestamp_ns
        else:
            device_time_ns = timestamp_ns

        return cls(
            stream_id=stream_id,
            sequence_id=sequence_id,
            data=data,
            sample_rate_hz=sample_rate_hz,
            host_receive_time_ns=host_receive_time_ns,
            device_time_ns=device_time_ns,
            synchronized_time_ns=synchronized_time_ns,
            clock_domain=domain,
            metadata=metadata or {},
        )


def frame_channel_count(frame: SignalFrame) -> int:
    """Resolve the explicit channel axis of a canonical frame.

    One-dimensional streaming frames are one multi-channel sample. For arrays
    with more than one dimension, ``metadata['axis_order']`` must identify
    exactly one ``'channel'`` axis. neurOS refuses to guess from shape alone.
    """

    if not isinstance(frame, SignalFrame):
        raise TypeError("frame_channel_count requires a SignalFrame")
    if frame.data.ndim == 1:
        return int(frame.data.shape[0])

    axis_order = tuple(frame.metadata.get("axis_order", ()))
    if len(axis_order) != frame.data.ndim:
        raise ValueError(
            "Multi-dimensional SignalFrames require axis_order metadata with one "
            "entry per data dimension"
        )
    if axis_order.count("channel") != 1:
        raise ValueError(
            "Multi-dimensional SignalFrames require exactly one 'channel' axis"
        )
    return int(frame.data.shape[axis_order.index("channel")])


def validate_frame_against_descriptor(
    descriptor: StreamDescriptor,
    frame: SignalFrame,
    *,
    sample_rate_atol_hz: float = 1e-12,
) -> None:
    """Fail closed when a frame contradicts its registered stream identity."""

    if not isinstance(descriptor, StreamDescriptor):
        raise TypeError("descriptor must be a StreamDescriptor")
    if not isinstance(frame, SignalFrame):
        raise TypeError("frame must be a SignalFrame")
    if not isinstance(sample_rate_atol_hz, Real) or isinstance(
        sample_rate_atol_hz, (bool, np.bool_)
    ):
        raise TypeError("sample_rate_atol_hz must be a real numeric scalar")
    tolerance = float(sample_rate_atol_hz)
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("sample_rate_atol_hz must be finite and >= 0")

    if frame.stream_id != descriptor.stream_id:
        raise ValueError("SignalFrame stream_id does not match StreamDescriptor")
    if not math.isclose(
        frame.sample_rate_hz,
        descriptor.sample_rate_hz,
        rel_tol=0.0,
        abs_tol=tolerance,
    ):
        raise ValueError("SignalFrame sample_rate_hz does not match StreamDescriptor")
    if (
        descriptor.clock_domain is not ClockDomain.UNKNOWN
        and frame.clock_domain is not descriptor.clock_domain
    ):
        raise ValueError("SignalFrame clock_domain does not match StreamDescriptor")

    if descriptor.channel_names:
        observed_channels = frame_channel_count(frame)
        if observed_channels != len(descriptor.channel_names):
            raise ValueError("SignalFrame channel geometry does not match StreamDescriptor")

    metadata_names = tuple(frame.metadata.get("channel_names", ()))
    if metadata_names and descriptor.channel_names and metadata_names != descriptor.channel_names:
        raise ValueError("SignalFrame channel_names contradict StreamDescriptor")
    metadata_types = tuple(frame.metadata.get("channel_types", ()))
    if metadata_types and descriptor.channel_types and metadata_types != descriptor.channel_types:
        raise ValueError("SignalFrame channel_types contradict StreamDescriptor")
    metadata_units = tuple(frame.metadata.get("units", ()))
    if metadata_units and descriptor.units and metadata_units != descriptor.units:
        raise ValueError("SignalFrame units contradict StreamDescriptor")
    metadata_modality = frame.metadata.get("modality")
    if metadata_modality is not None and str(metadata_modality) != descriptor.modality:
        raise ValueError("SignalFrame modality contradicts StreamDescriptor")
