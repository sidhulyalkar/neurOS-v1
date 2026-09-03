"""High-level dataset API backed by the optional neurOS Rust data plane.

Single-modality streaming remains the v0 execution contract. Multimodal v1
starts with an explicit exact-clock planning authority: neurOS can prove a
cross-modal frame mapping before it is allowed to execute one. Interpolation,
nearest-neighbor matching, phase correction, and resampling remain separate
future policies rather than implicit behavior.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import neuros_runtime_native as _native
except ImportError:  # pragma: no cover - depends on optional native wheel
    _native = None


class NativeRuntimeUnavailable(ImportError):
    """Raised when a native Dataset operation is requested without the Rust wheel."""


@dataclass(frozen=True, slots=True)
class AlignmentPlan:
    """Compact, deterministic exact-clock multimodal alignment authority.

    The plan is metadata-only. It binds the dataset manifest, acquisition group,
    common clock boundary, window duration/stride, and per-modality frame mapping
    without opening or copying source arrays. Execution will consume a qualified
    plan in a later runtime layer rather than recomputing synchronization ad hoc.
    """

    _native_plan: Any

    @property
    def dataset_id(self) -> str:
        return str(self._native_plan.dataset_id)

    @property
    def manifest_sha256(self) -> str:
        return str(self._native_plan.manifest_sha256)

    @property
    def sync_group(self) -> str:
        return str(self._native_plan.sync_group)

    @property
    def start_ns(self) -> int:
        return int(self._native_plan.start_ns)

    @property
    def overlap_end_ns(self) -> int:
        return int(self._native_plan.overlap_end_ns)

    @property
    def duration_ns(self) -> int:
        return int(self._native_plan.duration_ns)

    @property
    def stride_ns(self) -> int:
        return int(self._native_plan.stride_ns)

    @property
    def window_count(self) -> int:
        return int(self._native_plan.window_count)

    @property
    def sha256(self) -> str:
        """Domain-separated SHA-256 identity of the exact serialized plan."""

        return str(self._native_plan.sha256)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete stable-ordered plan payload."""

        payload = json.loads(self._native_plan.to_json())
        if not isinstance(payload, dict):  # pragma: no cover - native invariant
            raise RuntimeError("native alignment plan did not serialize to an object")
        return payload

    @property
    def entries(self) -> tuple[dict[str, Any], ...]:
        payload = self.to_dict()
        return tuple(dict(entry) for entry in payload["entries"])

    def window(self, index: int) -> dict[str, Any]:
        """Materialize one window's exact time/frame mapping for inspection.

        This does not read source data. It derives the selected frame intervals
        from the compact plan and is useful for independent validation, logging,
        and experiment provenance.
        """

        if index < 0 or index >= self.window_count:
            raise IndexError(
                f"alignment window index {index} outside [0, {self.window_count})"
            )
        start_ns = self.start_ns + index * self.stride_ns
        slices: list[dict[str, Any]] = []
        for entry in self.entries:
            start_frame = int(entry["start_frame"]) + index * int(entry["frame_stride"])
            frame_count = int(entry["frames_per_window"])
            slices.append(
                {
                    "record_id": str(entry["record_id"]),
                    "subject": str(entry["subject"]),
                    "modality": str(entry["modality"]),
                    "clock_id": str(entry["clock_id"]),
                    "period_ns": int(entry["period_ns"]),
                    "start_frame": start_frame,
                    "stop_frame": start_frame + frame_count,
                    "frame_count": frame_count,
                }
            )
        return {
            "plan_sha256": self.sha256,
            "window_index": index,
            "start_ns": start_ns,
            "end_ns": start_ns + self.duration_ns,
            "slices": slices,
        }


@dataclass(frozen=True, slots=True)
class DataWindow:
    """One deterministic time window over one modality.

    ``values`` is an Arrow-compatible array backed by the source memory map. The
    flattened storage shape is available in ``shape`` so consumers can reshape
    according to model needs without neurOS guessing tensor-library semantics.
    """

    _native_window: Any

    @property
    def record_id(self) -> str:
        return str(self._native_window.record_id)

    @property
    def subject(self) -> str:
        return str(self._native_window.subject)

    @property
    def modality(self) -> str:
        return str(self._native_window.modality)

    @property
    def start_frame(self) -> int:
        return int(self._native_window.start_frame)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self._native_window.shape)

    @property
    def sampling_hz(self) -> float | None:
        value = self._native_window.sampling_hz
        return None if value is None else float(value)

    @property
    def values(self) -> Any:
        """Return a zero-copy ``arro3.core.Array`` over the mapped source bytes."""

        return self._native_window.to_arrow()

    def to_pyarrow(self) -> Any:
        """Return a zero-copy ``pyarrow.Array`` when PyArrow is installed."""

        return self._native_window.to_pyarrow()

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "subject": self.subject,
            "modality": self.modality,
            "start_frame": self.start_frame,
            "shape": self.shape,
            "sampling_hz": self.sampling_hz,
            "manifest_sha256": str(self._native_window.manifest_sha256),
            "source_size_bytes": int(self._native_window.source_size_bytes),
        }

    def __getattr__(self, name: str) -> Any:
        """Allow the ergonomic ``window.fmri`` form for a matching modality."""

        normalized = self.modality.replace("-", "_").lower()
        if name.lower() == normalized:
            return self.values
        raise AttributeError(name)


class Dataset:
    """A validated study directory opened through the neurOS native data plane."""

    def __init__(self, native_dataset: Any, root: Path) -> None:
        self._native_dataset = native_dataset
        self.root = root

    @classmethod
    def open(cls, root: str | Path) -> "Dataset":
        native = _require_native()
        resolved = Path(root).expanduser().resolve()
        return cls(native.NativeDataset.open(resolved), resolved)

    @property
    def dataset_id(self) -> str:
        return str(self._native_dataset.dataset_id)

    @property
    def manifest_sha256(self) -> str:
        return str(self._native_dataset.manifest_sha256)

    @property
    def record_count(self) -> int:
        return int(self._native_dataset.record_count)

    def plan_aligned(
        self,
        *,
        sync_group: str,
        modalities: Sequence[str],
        duration_ns: int,
        stride_ns: int | None = None,
        policy: str = "exact",
    ) -> AlignmentPlan:
        """Prove an exact multimodal frame mapping without opening source arrays.

        ``policy="exact"`` is currently the only accepted policy. Every window
        boundary must be representable on every selected modality's integer clock.
        Requests that would require interpolation, extrapolation, tolerance-based
        nearest-neighbor matching, or implicit phase correction are rejected.
        """

        if policy != "exact":
            raise ValueError(
                "only policy='exact' is implemented; resampling policies must be explicit"
            )
        selected = tuple(str(modality) for modality in modalities)
        if len(selected) < 2:
            raise ValueError("plan_aligned requires at least two modalities")
        if len(set(selected)) != len(selected):
            raise ValueError("plan_aligned modalities cannot contain duplicates")
        if duration_ns <= 0:
            raise ValueError("duration_ns must be positive")
        if stride_ns is not None and stride_ns <= 0:
            raise ValueError("stride_ns must be positive when supplied")

        native_plan = self._native_dataset.plan_aligned(
            sync_group=sync_group,
            modalities=list(selected),
            duration_ns=duration_ns,
            stride_ns=stride_ns,
        )
        return AlignmentPlan(native_plan)

    def stream(
        self,
        *,
        subjects: Sequence[str] | None = None,
        modalities: Sequence[str] | None = None,
        window: int,
        stride: int | None = None,
        prefetch: int = 8,
    ) -> Iterator[DataWindow]:
        """Stream deterministic windows with bounded native prefetch.

        v0 requires exactly one selected modality. This is an intentional
        scientific guardrail: fMRI, behavior, EEG, and video clocks cannot be
        safely aligned merely by zipping sample indices. Use :meth:`plan_aligned`
        to preflight a multimodal exact-clock mapping.
        """

        native = _require_native()
        selected_modalities = tuple(modalities or ())
        if len(selected_modalities) != 1:
            raise ValueError(
                "neuros-runtime v0 requires exactly one modality per stream; "
                "use plan_aligned() to establish an explicit multimodal clock contract"
            )
        native.require_single_modality(list(selected_modalities))
        stream = self._native_dataset.stream(
            subjects=None if subjects is None else list(subjects),
            modalities=list(selected_modalities),
            window=window,
            stride=stride,
            prefetch=prefetch,
        )
        for native_window in stream:
            yield DataWindow(native_window)


def native_runtime_available() -> bool:
    return _native is not None


def _require_native() -> Any:
    if _native is None:
        raise NativeRuntimeUnavailable(
            "The neurOS Rust data plane is not installed. Build "
            "rust/neuros-runtime-py with `maturin develop --release` or install "
            "the `neuros-runtime-native` wheel."
        )
    return _native


__all__ = [
    "AlignmentPlan",
    "DataWindow",
    "Dataset",
    "NativeRuntimeUnavailable",
    "native_runtime_available",
]
