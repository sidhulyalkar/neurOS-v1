"""High-level dataset API backed by the optional neurOS Rust data plane.

The v0 contract deliberately supports one modality per stream. Cross-modal
synchronization is not inferred from array position; a future runtime revision
will require explicit clock and resampling policy before returning aligned
multimodal batches.
"""

from __future__ import annotations

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
        safely aligned merely by zipping sample indices.
        """

        native = _require_native()
        selected_modalities = tuple(modalities or ())
        if len(selected_modalities) != 1:
            raise ValueError(
                "neuros-runtime v0 requires exactly one modality per stream; "
                "explicit multimodal clock synchronization is the next runtime contract"
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
    "DataWindow",
    "Dataset",
    "NativeRuntimeUnavailable",
    "native_runtime_available",
]
