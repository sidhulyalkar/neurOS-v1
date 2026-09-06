"""Fail-closed bridge from neurOS runtime provenance into research authority.

This module deliberately consumes plain mappings rather than runtime objects so
``neuros-research`` stays dependency-free. Dataset byte/interpretation identity
and temporal execution identity remain separate cryptographic claims.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ._canonical import require_nonempty, require_sha256, thaw_json
from .contracts import ExperimentPacket

RUNTIME_DATASET_BINDING_SCHEMA = "neuros.runtime_dataset_binding.v1"
RUNTIME_ALIGNMENT_AUTHORITY_SCHEMA = "neuros.runtime_alignment_authority.v1"
RUNTIME_ALIGNMENT_METADATA_KEY = "neuros_runtime_alignment"

_ALIGNMENT_CLAIM_BOUNDARY = (
    "exact integer-clock temporal correspondence is systems execution authority; "
    "it does not establish causal, physiological, acquisition-lineage, preprocessing, "
    "or model-performance claims"
)


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _require_string(value: Any, *, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    normalized = require_nonempty(value, name=name)
    if normalized != value:
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return normalized


def _require_digest(value: Any, *, name: str) -> str:
    return require_sha256(_require_string(value, name=name), name=name)


def _require_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return value


def _require_positive_int(value: Any, *, name: str) -> int:
    normalized = _require_int(value, name=name)
    if normalized <= 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _require_modalities(value: Any) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("alignment modalities must be a sequence")
    modalities = tuple(
        _require_string(item, name="alignment modality") for item in value
    )
    if len(modalities) < 2:
        raise ValueError("runtime alignment authority requires at least two modalities")
    if len(set(modalities)) != len(modalities):
        raise ValueError("runtime alignment modalities must be unique")
    return modalities


def _runtime_dataset_binding(packet: ExperimentPacket) -> Mapping[str, Any]:
    runtime = packet.dataset.metadata.get("neuros_runtime")
    runtime = _require_mapping(runtime, name="dataset metadata 'neuros_runtime'")
    if runtime.get("schema") != RUNTIME_DATASET_BINDING_SCHEMA:
        raise ValueError(
            "dataset authority is not bound by the supported neurOS runtime dataset schema"
        )
    if runtime.get("dataset_verification") != "verified_whole_dataset":
        raise ValueError("dataset authority does not prove whole-dataset runtime verification")
    if runtime.get("source_verification_semantics") != "verified_at_bridge":
        raise ValueError(
            "dataset authority must come from bridge-time runtime content verification"
        )
    if runtime.get("lineage_completeness") != "unknown":
        raise ValueError(
            "runtime dataset authority must retain conservative unknown lineage completeness"
        )
    _require_string(runtime.get("claim_boundary"), name="runtime claim_boundary")

    dataset_id = _require_string(runtime.get("dataset_id"), name="runtime dataset_id")
    if dataset_id != packet.dataset.dataset_id:
        raise ValueError("runtime dataset binding dataset_id differs from DatasetAuthority")

    declared_content = _require_digest(
        runtime.get("declared_dataset_content_sha256"),
        name="runtime declared_dataset_content_sha256",
    )
    verified_content = _require_digest(
        runtime.get("verified_dataset_content_sha256"),
        name="runtime verified_dataset_content_sha256",
    )
    if declared_content != verified_content:
        raise ValueError(
            "runtime declared and verified dataset content identities must match"
        )
    if verified_content != packet.dataset.source_fingerprint:
        raise ValueError(
            "runtime verified dataset content identity differs from DatasetAuthority.source_fingerprint"
        )

    _require_digest(runtime.get("manifest_sha256"), name="runtime manifest_sha256")
    return runtime


def bind_runtime_alignment(
    packet: ExperimentPacket,
    plan_provenance: Mapping[str, Any],
) -> ExperimentPacket:
    """Bind one frozen exact alignment plan into an immutable experiment packet.

    The plan SHA is execution identity, never dataset identity. The bridge therefore
    requires the plan's verified dataset-content SHA to equal the packet's existing
    ``DatasetAuthority.source_fingerprint`` and preserves the plan under a reserved
    packet metadata namespace.
    """

    if not isinstance(packet, ExperimentPacket):
        raise TypeError("packet must be an ExperimentPacket")
    plan = _require_mapping(plan_provenance, name="plan_provenance")
    runtime_dataset = _runtime_dataset_binding(packet)

    plan_sha256 = _require_digest(plan.get("plan_sha256"), name="alignment plan_sha256")
    dataset_id = _require_string(plan.get("dataset_id"), name="alignment dataset_id")
    if dataset_id != packet.dataset.dataset_id:
        raise ValueError("alignment plan belongs to a different dataset_id")

    dataset_content_sha256 = _require_digest(
        plan.get("dataset_content_sha256"),
        name="alignment dataset_content_sha256",
    )
    if dataset_content_sha256 != packet.dataset.source_fingerprint:
        raise ValueError(
            "alignment plan dataset content identity differs from DatasetAuthority.source_fingerprint"
        )

    manifest_sha256 = _require_digest(
        plan.get("manifest_sha256"), name="alignment manifest_sha256"
    )
    runtime_manifest = _require_digest(
        runtime_dataset.get("manifest_sha256"), name="runtime manifest_sha256"
    )
    if manifest_sha256 != runtime_manifest:
        raise ValueError("alignment plan manifest identity differs from runtime dataset authority")

    sync_group = _require_string(plan.get("sync_group"), name="alignment sync_group")
    policy = _require_string(plan.get("policy"), name="alignment policy")
    if policy != "exact":
        raise ValueError("research runtime bridge accepts exact alignment policy only")

    modalities = _require_modalities(plan.get("modalities"))
    start_ns = _require_int(plan.get("start_ns"), name="alignment start_ns")
    overlap_end_ns = _require_int(
        plan.get("overlap_end_ns"), name="alignment overlap_end_ns"
    )
    duration_ns = _require_positive_int(plan.get("duration_ns"), name="alignment duration_ns")
    stride_ns = _require_positive_int(plan.get("stride_ns"), name="alignment stride_ns")
    window_count = _require_positive_int(
        plan.get("window_count"), name="alignment window_count"
    )

    if overlap_end_ns <= start_ns:
        raise ValueError("alignment overlap_end_ns must be greater than start_ns")
    latest_start_ns = overlap_end_ns - duration_ns
    if start_ns > latest_start_ns:
        raise ValueError("alignment summary contains no executable window")
    expected_window_count = (latest_start_ns - start_ns) // stride_ns + 1
    if window_count != expected_window_count:
        raise ValueError(
            "alignment window_count is inconsistent with start/overlap/duration/stride summary"
        )

    packet_metadata = thaw_json(packet.metadata)
    if not isinstance(packet_metadata, dict):  # pragma: no cover - ExperimentPacket invariant
        raise TypeError("ExperimentPacket metadata did not thaw to a mapping")
    if RUNTIME_ALIGNMENT_METADATA_KEY in packet_metadata:
        raise ValueError(
            f"packet metadata key {RUNTIME_ALIGNMENT_METADATA_KEY!r} is reserved by neurOS"
        )

    packet_metadata[RUNTIME_ALIGNMENT_METADATA_KEY] = {
        "schema": RUNTIME_ALIGNMENT_AUTHORITY_SCHEMA,
        "plan_sha256": plan_sha256,
        "dataset_id": dataset_id,
        "dataset_content_sha256": dataset_content_sha256,
        "manifest_sha256": manifest_sha256,
        "sync_group": sync_group,
        "policy": policy,
        "modalities": list(modalities),
        "start_ns": start_ns,
        "overlap_end_ns": overlap_end_ns,
        "duration_ns": duration_ns,
        "stride_ns": stride_ns,
        "window_count": window_count,
        "claim_boundary": _ALIGNMENT_CLAIM_BOUNDARY,
    }
    return replace(packet, metadata=packet_metadata)


__all__ = [
    "RUNTIME_ALIGNMENT_AUTHORITY_SCHEMA",
    "RUNTIME_ALIGNMENT_METADATA_KEY",
    "RUNTIME_DATASET_BINDING_SCHEMA",
    "bind_runtime_alignment",
]
