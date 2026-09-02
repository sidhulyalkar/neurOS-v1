"""High-level dataset API backed by the optional neurOS Rust data plane.

The v0 contract deliberately supports one modality per stream. Cross-modal
synchronization is not inferred from array position; a future runtime revision
will require explicit clock and resampling policy before returning aligned
multimodal batches.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
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
    def end_frame_exclusive(self) -> int:
        return int(self._native_window.end_frame_exclusive)

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
        """Return explicit source, dataset, and interval identity for this window.

        ``verified_at_open`` means the complete mapped regular file matched the
        SHA-256 declared by the dataset manifest when the native runtime verified
        that mapping. It does not claim the surrounding filesystem is immutable
        against later external writers.
        """

        declared_source = self._native_window.declared_source_sha256
        verified_source = self._native_window.verified_source_sha256
        declared_dataset = self._native_window.declared_dataset_content_sha256
        verified_dataset = self._native_window.verified_dataset_content_sha256
        return {
            "record_id": self.record_id,
            "subject": self.subject,
            "modality": self.modality,
            "shape": self.shape,
            "sampling_hz": self.sampling_hz,
            "manifest_sha256": str(self._native_window.manifest_sha256),
            "source_size_bytes": int(self._native_window.source_size_bytes),
            "declared_source_sha256": (
                None if declared_source is None else str(declared_source)
            ),
            "verified_source_sha256": (
                None if verified_source is None else str(verified_source)
            ),
            "source_verification_state": str(
                self._native_window.source_verification_state
            ),
            "declared_dataset_content_sha256": (
                None if declared_dataset is None else str(declared_dataset)
            ),
            "verified_dataset_content_sha256": (
                None if verified_dataset is None else str(verified_dataset)
            ),
            "record_byte_interval": {
                "start": int(self._native_window.record_byte_start),
                "end_exclusive": int(self._native_window.record_byte_end_exclusive),
            },
            "window_frame_interval": {
                "start": self.start_frame,
                "end_exclusive": self.end_frame_exclusive,
            },
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
    def declared_content_sha256(self) -> str | None:
        value = self._native_dataset.declared_dataset_content_sha256
        return None if value is None else str(value)

    @property
    def verified_content_sha256(self) -> str | None:
        value = self._native_dataset.verified_dataset_content_sha256
        return None if value is None else str(value)

    @property
    def record_count(self) -> int:
        return int(self._native_dataset.record_count)

    def verify_content(self) -> str | None:
        """Verify every source needed by the canonical dataset content identity.

        Returns ``None`` when at least one manifest record does not declare a
        source hash. In that case neurOS intentionally refuses to invent a
        partially verified dataset content identity.
        """

        value = self._native_dataset.verify_content()
        return None if value is None else str(value)

    def to_orion_lineage(
        self,
        *,
        upstream_source: str,
        version: str | None = None,
        revision: str | None = None,
        parent_dataset_ids: Sequence[str] = (),
        identity_sets: Sequence[Any] = (),
        preprocessing_history: Sequence[str] = (),
        sampling_assumptions: Mapping[str, Any] | None = None,
        license: str | None = None,
        citation: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Any:
        """Bridge fully verified native content into conservative ORION lineage.

        The bridge always emits ``LineageCompleteness.UNKNOWN``. Content hashes
        establish the local bytes/interpretation consumed by neurOS, not upstream
        acquisition, preprocessing ancestry, or entity-identity completeness.
        Callers with stronger external evidence may construct a richer ORION
        ``DatasetLineage`` separately.
        """

        content_sha256 = self.verified_content_sha256
        if content_sha256 is None:
            raise ValueError(
                "ORION lineage requires a fully verified native dataset; call "
                "Dataset.verify_content() and require a non-None digest first"
            )
        try:
            from orion.scientific import DatasetLineage, LineageCompleteness
        except ImportError as exc:  # pragma: no cover - optional extra
            raise ImportError(
                "The ORION bridge requires the optional `neuros[orion]` extra."
            ) from exc

        bridge_metadata = dict(metadata or {})
        if "neuros_runtime" in bridge_metadata:
            raise ValueError("metadata key 'neuros_runtime' is reserved by neurOS")
        bridge_metadata["neuros_runtime"] = {
            "manifest_sha256": self.manifest_sha256,
            "declared_dataset_content_sha256": self.declared_content_sha256,
            "verified_dataset_content_sha256": content_sha256,
            "content_verification": "verified_at_open",
            "lineage_boundary": (
                "local content verification does not establish upstream lineage completeness"
            ),
        }
        return DatasetLineage(
            dataset_id=self.dataset_id,
            upstream_source=upstream_source,
            version=version,
            revision=revision,
            content_sha256=content_sha256,
            parent_dataset_ids=tuple(parent_dataset_ids),
            identity_sets=tuple(identity_sets),
            preprocessing_history=tuple(preprocessing_history),
            sampling_assumptions=dict(sampling_assumptions or {}),
            license=license,
            citation=citation,
            lineage_completeness=LineageCompleteness.UNKNOWN,
            metadata=bridge_metadata,
        )

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
