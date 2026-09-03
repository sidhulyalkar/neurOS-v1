"""High-level dataset API backed by the optional neurOS Rust data plane.

Single-modality streaming remains the v0 execution contract. Multimodal v1 uses
a provenance-bound exact-clock planning authority followed by an executor that
consumes that exact plan. Interpolation, nearest-neighbor matching, phase
correction, and resampling remain separate future policies rather than implicit
behavior.
"""

from __future__ import annotations

import json
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
class AlignmentPlan:
    """Compact, deterministic, provenance-bound exact-clock alignment authority.

    Creating a plan verifies the complete dataset content identity but does not
    return source arrays. The plan binds verified dataset/source identity, the
    exact manifest, acquisition group, clock mapping, and derived frame arithmetic.
    Aligned execution consumes this exact authority rather than silently
    recomputing synchronization.
    """

    _native_plan: Any

    @property
    def dataset_id(self) -> str:
        return str(self._native_plan.dataset_id)

    @property
    def dataset_content_sha256(self) -> str:
        """Verified canonical dataset byte/interpretation identity."""

        return str(self._native_plan.dataset_content_sha256)

    @property
    def manifest_sha256(self) -> str:
        """Exact serialized manifest identity that authorized this plan."""

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
        """Domain-separated identity of this exact temporal execution plan."""

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

    @property
    def provenance(self) -> dict[str, Any]:
        """Return plan-level identities without materializing a window."""

        return {
            "plan_sha256": self.sha256,
            "dataset_id": self.dataset_id,
            "dataset_content_sha256": self.dataset_content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "sync_group": self.sync_group,
            "policy": "exact",
            "start_ns": self.start_ns,
            "overlap_end_ns": self.overlap_end_ns,
            "duration_ns": self.duration_ns,
            "stride_ns": self.stride_ns,
            "window_count": self.window_count,
        }

    def window(self, index: int) -> dict[str, Any]:
        """Materialize one window's exact time/frame mapping for inspection.

        This does not read source data. It derives selected frame intervals from
        the compact plan and carries the source/record descriptors needed to audit
        the later execution result independently.
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
                    "source_sha256": str(entry["source_sha256"]),
                    "offset_bytes": int(entry["offset_bytes"]),
                    "dtype": str(entry["dtype"]),
                    "shape": tuple(int(value) for value in entry["shape"]),
                    "clock_id": str(entry["clock_id"]),
                    "clock_start_ns": int(entry["clock_start_ns"]),
                    "period_ns": int(entry["period_ns"]),
                    "start_frame": start_frame,
                    "stop_frame": start_frame + frame_count,
                    "frame_count": frame_count,
                }
            )
        return {
            "plan_sha256": self.sha256,
            "dataset_content_sha256": self.dataset_content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "sync_group": self.sync_group,
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
            "start_frame": self.start_frame,
            "end_frame_exclusive": self.end_frame_exclusive,
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


@dataclass(frozen=True, slots=True)
class AlignedWindow:
    """One exact multimodal execution window produced from a frozen plan.

    Each modality remains a separate :class:`DataWindow` backed by its own mmap.
    neurOS does not concatenate, interpolate, resample, or otherwise transform the
    scientific arrays when constructing this envelope.
    """

    _native_window: Any

    @property
    def plan_sha256(self) -> str:
        return str(self._native_window.plan_sha256)

    @property
    def dataset_content_sha256(self) -> str:
        return str(self._native_window.dataset_content_sha256)

    @property
    def manifest_sha256(self) -> str:
        return str(self._native_window.manifest_sha256)

    @property
    def sync_group(self) -> str:
        return str(self._native_window.sync_group)

    @property
    def window_index(self) -> int:
        return int(self._native_window.window_index)

    @property
    def start_ns(self) -> int:
        return int(self._native_window.start_ns)

    @property
    def end_ns(self) -> int:
        return int(self._native_window.end_ns)

    @property
    def modalities(self) -> tuple[str, ...]:
        return tuple(str(value) for value in self._native_window.modalities)

    def window(self, modality: str) -> DataWindow:
        """Return the zero-copy modality window selected by the qualified plan."""

        native_window = self._native_window.window(str(modality))
        if native_window is None:
            raise KeyError(f"aligned window does not contain modality {modality!r}")
        return DataWindow(native_window)

    @property
    def provenance(self) -> dict[str, Any]:
        """Return the exact execution envelope and per-modality source evidence."""

        return {
            "plan_sha256": self.plan_sha256,
            "dataset_content_sha256": self.dataset_content_sha256,
            "manifest_sha256": self.manifest_sha256,
            "sync_group": self.sync_group,
            "window_index": self.window_index,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "modalities": {
                modality: self.window(modality).provenance
                for modality in self.modalities
            },
        }

    def __getattr__(self, name: str) -> Any:
        """Allow ``batch.fmri`` / ``batch.behavior`` zero-copy access."""

        normalized = name.lower()
        for modality in self.modalities:
            if modality.replace("-", "_").lower() == normalized:
                return self.window(modality).values
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

    def plan_aligned(
        self,
        *,
        sync_group: str,
        modalities: Sequence[str],
        duration_ns: int,
        stride_ns: int | None = None,
        policy: str = "exact",
    ) -> AlignmentPlan:
        """Verify content and prove an exact multimodal frame mapping.

        ``policy="exact"`` is currently the only accepted policy. Every window
        boundary must be representable on every selected modality's integer clock.
        The complete dataset must declare valid source SHA-256 values; planning
        verifies those bytes before returning. Requests that would require
        interpolation, extrapolation, tolerance matching, or implicit phase repair
        are rejected.
        """

        if policy != "exact":
            raise ValueError(
                "only policy='exact' is implemented; resampling policies must be explicit"
            )
        if not sync_group or sync_group != sync_group.strip():
            raise ValueError("sync_group must be non-empty and have no surrounding whitespace")
        selected = tuple(str(modality) for modality in modalities)
        if len(selected) < 2:
            raise ValueError("plan_aligned requires at least two modalities")
        if any(not modality or modality != modality.strip() for modality in selected):
            raise ValueError("plan_aligned modalities must be non-empty canonical identifiers")
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

    def stream_aligned(
        self,
        plan: AlignmentPlan,
        *,
        prefetch: int = 8,
    ) -> Iterator[AlignedWindow]:
        """Execute one frozen exact plan with bounded native prefetch.

        The native executor validates the supplied plan directly rather than
        replanning synchronization. Before the stream is accepted, it re-reads the
        current physical source files and verifies their whole-file SHA-256 values
        independently of the mmap verification cache. This is an execution-start
        integrity check, not a claim that external writers cannot mutate files
        afterwards.
        """

        if not isinstance(plan, AlignmentPlan):
            raise TypeError("stream_aligned requires an AlignmentPlan from plan_aligned()")
        if prefetch <= 0:
            raise ValueError("prefetch must be at least one")
        if plan.dataset_id != self.dataset_id:
            raise ValueError("alignment plan belongs to a different dataset")
        if plan.manifest_sha256 != self.manifest_sha256:
            raise ValueError("alignment plan does not match the currently opened manifest")

        stream = self._native_dataset.stream_aligned(
            plan=plan._native_plan,
            prefetch=prefetch,
        )
        for native_window in stream:
            yield AlignedWindow(native_window)

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
        """Stream deterministic single-modality windows with bounded native prefetch.

        v0 execution requires exactly one selected modality. This remains an
        intentional scientific guardrail: fMRI, behavior, EEG, and video clocks
        cannot be safely aligned merely by zipping sample indices. Use
        :meth:`plan_aligned` followed by :meth:`stream_aligned` for qualified exact
        multimodal execution.
        """

        native = _require_native()
        selected_modalities = tuple(modalities or ())
        if len(selected_modalities) != 1:
            raise ValueError(
                "neuros-runtime v0 execution requires exactly one modality per stream; "
                "use plan_aligned() to establish an exact multimodal clock contract"
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
    "AlignedWindow",
    "AlignmentPlan",
    "DataWindow",
    "Dataset",
    "NativeRuntimeUnavailable",
    "native_runtime_available",
]
