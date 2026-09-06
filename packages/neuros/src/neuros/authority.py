"""Thin authority adapters over promoted neurOS runtime provenance.

The runtime owns byte/interpretation and exact temporal execution identity. This
module only projects those already-qualified identities into optional higher-level
research contracts when the corresponding package is installed.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .dataset import AlignmentPlan, Dataset

RUNTIME_DATASET_BINDING_SCHEMA = "neuros.runtime_dataset_binding.v1"

_DATASET_CLAIM_BOUNDARY = (
    "verified local bytes and declared record interpretation do not establish upstream "
    "acquisition lineage, preprocessing ancestry, participant identity completeness, "
    "licensing closure, or model-performance evidence"
)


def _require_sha256(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string SHA-256")
    if value != value.strip() or len(value) != 64 or any(
        char not in "0123456789abcdef" for char in value
    ):
        raise ValueError(f"{name} must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_identifier(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def runtime_dataset_binding(dataset: Dataset) -> dict[str, Any]:
    """Freshly verify and bind one complete runtime dataset for research authority.

    Authority construction is intentionally stronger than reading the cached
    ``verified_content_sha256`` property. The complete dataset is rehashed at bridge
    time so a path-stable mutation after an earlier verification cannot silently be
    promoted into a new prospective research packet.
    """

    if not isinstance(dataset, Dataset):
        raise TypeError("dataset must be a neuros.dataset.Dataset")

    dataset_id = _require_identifier(dataset.dataset_id, name="runtime dataset_id")
    manifest_sha256 = _require_sha256(
        dataset.manifest_sha256, name="runtime manifest SHA-256"
    )

    verified_content = dataset.verify_content()
    if verified_content is None:
        raise ValueError(
            "runtime dataset binding requires a complete declared source identity; "
            "every manifest record must declare source_sha256"
        )
    verified_content = _require_sha256(
        verified_content, name="verified dataset content SHA-256"
    )
    if dataset.verified_content_sha256 != verified_content:
        raise RuntimeError(
            "runtime verification state differs from the digest returned by verify_content()"
        )

    declared_content = dataset.declared_content_sha256
    if declared_content is None:
        raise RuntimeError(
            "runtime returned a verified dataset digest without a complete declared identity"
        )
    declared_content = _require_sha256(
        declared_content, name="declared dataset content SHA-256"
    )
    if declared_content != verified_content:
        raise ValueError(
            "verified dataset content SHA-256 differs from the declared dataset identity"
        )

    return {
        "schema": RUNTIME_DATASET_BINDING_SCHEMA,
        "dataset_id": dataset_id,
        "manifest_sha256": manifest_sha256,
        "declared_dataset_content_sha256": declared_content,
        "verified_dataset_content_sha256": verified_content,
        "dataset_verification": "verified_whole_dataset",
        "source_verification_semantics": "verified_at_bridge",
        "lineage_completeness": "unknown",
        "claim_boundary": _DATASET_CLAIM_BOUNDARY,
    }


def alignment_authority_provenance(plan: AlignmentPlan) -> dict[str, Any]:
    """Project an exact runtime plan into research-facing execution metadata.

    The returned mapping preserves ``plan.sha256`` as temporal execution identity.
    It does not substitute that fingerprint for dataset content identity.
    """

    if not isinstance(plan, AlignmentPlan):
        raise TypeError("plan must be a neuros.dataset.AlignmentPlan")
    provenance = dict(plan.provenance)
    modalities: list[str] = []
    for entry in plan.entries:
        modality = _require_identifier(entry.get("modality"), name="alignment modality")
        modalities.append(modality)
    if len(modalities) < 2 or len(set(modalities)) != len(modalities):
        raise ValueError("alignment authority requires at least two unique modalities")
    provenance["modalities"] = modalities
    return provenance


def to_research_dataset_authority(
    dataset: Dataset,
    *,
    access: str,
    source_revision: str,
    metadata: Mapping[str, Any] | None = None,
) -> Any:
    """Project freshly verified runtime dataset identity into ``DatasetAuthority``.

    ``source_revision`` is the dataset/source release or revision supplied by the
    caller. It must not be silently replaced by the neurOS code revision.
    """

    binding = runtime_dataset_binding(dataset)
    try:
        from neuros.research import DatasetAuthority
    except ImportError as exc:  # pragma: no cover - optional distribution
        raise ImportError(
            "The research authority bridge requires the optional `neuros-research` distribution."
        ) from exc

    authority_metadata = dict(metadata or {})
    if "neuros_runtime" in authority_metadata:
        raise ValueError("metadata key 'neuros_runtime' is reserved by neurOS")
    authority_metadata["neuros_runtime"] = binding
    return DatasetAuthority(
        dataset_id=str(binding["dataset_id"]),
        source_fingerprint=str(binding["verified_dataset_content_sha256"]),
        access=access,
        source_revision=source_revision,
        metadata=authority_metadata,
    )


__all__ = [
    "RUNTIME_DATASET_BINDING_SCHEMA",
    "alignment_authority_provenance",
    "runtime_dataset_binding",
    "to_research_dataset_authority",
]
