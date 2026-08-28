"""Descriptor-preserving MOABB epoch collection for scientific evidence.

The historical :func:`collect_moabb` adapter intentionally returns only arrays,
labels, and deployment-unit metadata. That is sufficient for model fitting, but
not for a promoted qualification study that must freeze the *processed signal
contract* seen by every external method.

MOABB supports ``return_epochs=True``. This module uses that upstream surface to
retain the processed MNE Epochs channel order, channel types, sampling rate,
epoch timing, and event mapping before converting the samples into the existing
``GroupedEvaluationData`` authority.

No raw-data checksum or physical-channel claim is inferred from these fields.
They describe the processed MOABB/MNE object supplied to the benchmark.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .real_world import GroupedEvaluationData


def _canonical_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"schema": schema, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _nonempty_strings(name: str, values: Sequence[Any]) -> tuple[str, ...]:
    result = tuple(str(value).strip() for value in values)
    if not result or any(not value for value in result):
        raise ValueError(f"{name} must contain non-empty strings")
    return result


@dataclass(frozen=True, slots=True)
class MOABBEpochDescriptor:
    """Observed processed MNE Epochs geometry used by a MOABB study.

    ``signal_contract_sha256`` intentionally excludes ``n_trials`` so the same
    preprocessing/channel contract can be compared across participants with
    different numbers of available trials. ``sha256`` includes the participant-
    specific trial count and therefore identifies the complete descriptor.
    """

    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    sampling_rate_hz: float
    n_times: int
    epoch_start_s: float
    epoch_end_s: float
    event_id: tuple[tuple[str, int], ...]
    n_trials: int
    array_axes: tuple[str, str, str] = ("trial", "channel", "time")
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("MOABBEpochDescriptor schema_version must be 1")
        channels = _nonempty_strings("channel_names", self.channel_names)
        types = _nonempty_strings("channel_types", self.channel_types)
        if len(channels) != len(types):
            raise ValueError("channel_names and channel_types must have identical length")
        if len(set(channels)) != len(channels):
            raise ValueError("processed MOABB channel names cannot contain duplicates")
        rate = float(self.sampling_rate_hz)
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError("sampling_rate_hz must be finite and positive")
        if isinstance(self.n_times, bool) or int(self.n_times) <= 0:
            raise ValueError("n_times must be a positive integer")
        if isinstance(self.n_trials, bool) or int(self.n_trials) <= 0:
            raise ValueError("n_trials must be a positive integer")
        start = float(self.epoch_start_s)
        end = float(self.epoch_end_s)
        if not math.isfinite(start) or not math.isfinite(end) or end < start:
            raise ValueError("epoch timing must be finite with end >= start")
        events = tuple((str(name).strip(), int(code)) for name, code in self.event_id)
        if any(not name for name, _ in events):
            raise ValueError("event_id names must be non-empty")
        if len({name for name, _ in events}) != len(events):
            raise ValueError("event_id cannot repeat names")
        axes = tuple(str(value).strip() for value in self.array_axes)
        if axes != ("trial", "channel", "time"):
            raise ValueError("MOABB evidence arrays must use trial/channel/time axes")
        object.__setattr__(self, "channel_names", channels)
        object.__setattr__(self, "channel_types", types)
        object.__setattr__(self, "sampling_rate_hz", rate)
        object.__setattr__(self, "n_times", int(self.n_times))
        object.__setattr__(self, "n_trials", int(self.n_trials))
        object.__setattr__(self, "epoch_start_s", start)
        object.__setattr__(self, "epoch_end_s", end)
        object.__setattr__(self, "event_id", tuple(sorted(events)))
        object.__setattr__(self, "array_axes", axes)

    def signal_contract_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "array_axes": list(self.array_axes),
            "channel_names": list(self.channel_names),
            "channel_types": list(self.channel_types),
            "sampling_rate_hz": self.sampling_rate_hz,
            "n_times": self.n_times,
            "epoch_start_s": self.epoch_start_s,
            "epoch_end_s": self.epoch_end_s,
            "event_id": [[name, code] for name, code in self.event_id],
        }

    @property
    def signal_contract_sha256(self) -> str:
        return _canonical_sha256(
            "neuros.moabb_epoch_signal_contract.v1",
            self.signal_contract_dict(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.signal_contract_dict(),
            "n_trials": self.n_trials,
            "signal_contract_sha256": self.signal_contract_sha256,
        }

    @property
    def sha256(self) -> str:
        payload = self.to_dict()
        payload.pop("signal_contract_sha256")
        return _canonical_sha256("neuros.moabb_epoch_descriptor.v1", payload)


def _epoch_descriptor(epochs: Any, array: np.ndarray) -> MOABBEpochDescriptor:
    if array.ndim != 3:
        raise ValueError(
            "processed MOABB evidence must be a 3-D trial/channel/time array; "
            f"observed shape={array.shape}"
        )
    channel_names = getattr(epochs, "ch_names", None)
    if channel_names is None:
        raise ValueError("processed MNE Epochs object does not expose channel names")
    get_types = getattr(epochs, "get_channel_types", None)
    if not callable(get_types):
        raise ValueError("processed MNE Epochs object does not expose channel types")
    info = getattr(epochs, "info", None)
    if info is None:
        raise ValueError("processed MNE Epochs object does not expose info")
    try:
        sampling_rate_hz = float(info["sfreq"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("processed MNE Epochs info has no valid sfreq") from exc
    times = np.asarray(getattr(epochs, "times", ()), dtype=np.float64)
    if times.ndim != 1 or len(times) != array.shape[2] or len(times) == 0:
        raise ValueError("processed MNE Epochs time axis differs from sample array")
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("processed MNE Epochs times must be finite and increasing")
    raw_event_id = getattr(epochs, "event_id", {})
    if not isinstance(raw_event_id, Mapping):
        raise ValueError("processed MNE Epochs event_id must be a mapping")
    descriptor = MOABBEpochDescriptor(
        channel_names=tuple(channel_names),
        channel_types=tuple(get_types()),
        sampling_rate_hz=sampling_rate_hz,
        n_times=int(array.shape[2]),
        epoch_start_s=float(times[0]),
        epoch_end_s=float(times[-1]),
        event_id=tuple((str(name), int(code)) for name, code in raw_event_id.items()),
        n_trials=int(array.shape[0]),
    )
    if len(descriptor.channel_names) != array.shape[1]:
        raise ValueError(
            "processed MNE Epochs channel order length differs from sample array"
        )
    return descriptor


def collect_moabb_epochs(
    dataset: Any,
    paradigm: Any,
    *,
    subjects: Sequence[int] | None = None,
    dataset_id: str,
    **get_data_kwargs: Any,
) -> tuple[GroupedEvaluationData, MOABBEpochDescriptor]:
    """Collect MOABB data while preserving its processed MNE Epochs contract.

    The caller supplies ``dataset_id`` explicitly because promoted evidence must
    not infer scientific identity from a Python class name. The upstream
    paradigm must honor MOABB's documented ``return_epochs=True`` contract.
    """

    if not isinstance(dataset_id, str) or not dataset_id.strip():
        raise ValueError("dataset_id must be a non-empty explicit evidence identity")
    getter = getattr(paradigm, "get_data", None)
    if not callable(getter):
        raise TypeError("paradigm must provide a callable get_data method")
    if "return_epochs" in get_data_kwargs:
        raise ValueError("collect_moabb_epochs owns return_epochs=True")
    result = getter(
        dataset=dataset,
        subjects=subjects,
        return_epochs=True,
        **get_data_kwargs,
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError("MOABB epoch result must be (epochs, labels, metadata)")
    epochs, labels, metadata = result
    epoch_get_data = getattr(epochs, "get_data", None)
    if not callable(epoch_get_data):
        raise TypeError(
            "MOABB return_epochs=True did not return an MNE Epochs-like object"
        )
    array = np.asarray(epoch_get_data())
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("processed MOABB epoch data must be finite numeric values")
    descriptor = _epoch_descriptor(epochs, array)
    data = GroupedEvaluationData.from_moabb_result(
        (array, labels, metadata),
        dataset_id=dataset_id.strip(),
    )
    if tuple(data.X.shape) != tuple(array.shape):
        raise RuntimeError("GroupedEvaluationData changed processed epoch geometry")
    return data, descriptor


__all__ = ["MOABBEpochDescriptor", "collect_moabb_epochs"]
