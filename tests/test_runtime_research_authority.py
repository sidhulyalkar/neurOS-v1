from __future__ import annotations

import json
from pathlib import Path

import pytest
from neuros.authority import (
    alignment_authority_provenance,
    runtime_dataset_binding,
    to_research_dataset_authority,
)
from neuros.dataset import AlignmentPlan, Dataset

SHA_MANIFEST = "b" * 64
SHA_DATASET = "a" * 64
SHA_PLAN = "e" * 64


class _FakeNativeDataset:
    dataset_id = "fixture"
    manifest_sha256 = SHA_MANIFEST
    declared_dataset_content_sha256 = SHA_DATASET
    record_count = 2

    def __init__(self, *, complete: bool = True) -> None:
        self.complete = complete
        self.verify_calls = 0
        self.verified_dataset_content_sha256 = SHA_DATASET

    def verify_content(self) -> str | None:
        self.verify_calls += 1
        if not self.complete:
            self.verified_dataset_content_sha256 = None
            return None
        self.verified_dataset_content_sha256 = SHA_DATASET
        return SHA_DATASET


class _MutatedNativeDataset(_FakeNativeDataset):
    def verify_content(self) -> str | None:
        self.verify_calls += 1
        raise ValueError("source hash mismatch after mutation")


class _FakeNativePlan:
    dataset_id = "fixture"
    dataset_content_sha256 = SHA_DATASET
    manifest_sha256 = SHA_MANIFEST
    sync_group = "sub-01/run-01"
    start_ns = 0
    overlap_end_ns = 20_000_000_000
    duration_ns = 4_000_000_000
    stride_ns = 2_000_000_000
    window_count = 9
    sha256 = SHA_PLAN

    def __init__(self, modalities: tuple[object, ...] = ("behavior", "fmri")) -> None:
        self._modalities = modalities

    def to_json(self) -> str:
        return json.dumps(
            {
                "entries": [
                    {"modality": modality} for modality in self._modalities
                ]
            }
        )


def test_runtime_dataset_binding_reverifies_even_when_cached_state_exists() -> None:
    native = _FakeNativeDataset()
    dataset = Dataset(native, Path("/fixture"))

    binding = runtime_dataset_binding(dataset)

    assert native.verify_calls == 1
    assert binding["dataset_id"] == "fixture"
    assert binding["manifest_sha256"] == SHA_MANIFEST
    assert binding["declared_dataset_content_sha256"] == SHA_DATASET
    assert binding["verified_dataset_content_sha256"] == SHA_DATASET
    assert binding["dataset_verification"] == "verified_whole_dataset"
    assert binding["source_verification_semantics"] == "verified_at_bridge"
    assert binding["lineage_completeness"] == "unknown"


def test_runtime_dataset_binding_fails_closed_for_incomplete_content_identity() -> None:
    dataset = Dataset(_FakeNativeDataset(complete=False), Path("/fixture"))
    with pytest.raises(ValueError, match="complete declared source identity"):
        runtime_dataset_binding(dataset)


def test_runtime_dataset_binding_propagates_bridge_time_mutation_detection() -> None:
    dataset = Dataset(_MutatedNativeDataset(), Path("/fixture"))
    with pytest.raises(ValueError, match="source hash mismatch"):
        runtime_dataset_binding(dataset)


def test_research_dataset_authority_preserves_dataset_identity_and_revision() -> None:
    native = _FakeNativeDataset()
    dataset = Dataset(native, Path("/fixture"))

    authority = to_research_dataset_authority(
        dataset,
        access="authorized_restricted",
        source_revision="dataset-release-v1",
        metadata={"study": "fixture"},
    )

    assert native.verify_calls == 1
    assert authority.dataset_id == "fixture"
    assert authority.source_fingerprint == SHA_DATASET
    assert authority.source_revision == "dataset-release-v1"
    assert authority.metadata["study"] == "fixture"
    assert authority.metadata["neuros_runtime"]["manifest_sha256"] == SHA_MANIFEST
    assert authority.metadata["neuros_runtime"]["lineage_completeness"] == "unknown"


def test_research_dataset_authority_reserves_runtime_metadata_namespace() -> None:
    dataset = Dataset(_FakeNativeDataset(), Path("/fixture"))
    with pytest.raises(ValueError, match="reserved"):
        to_research_dataset_authority(
            dataset,
            access="authorized_restricted",
            source_revision="dataset-release-v1",
            metadata={"neuros_runtime": {"spoofed": True}},
        )


def test_alignment_authority_projection_keeps_plan_and_dataset_hashes_separate() -> None:
    provenance = alignment_authority_provenance(AlignmentPlan(_FakeNativePlan()))

    assert provenance["plan_sha256"] == SHA_PLAN
    assert provenance["dataset_content_sha256"] == SHA_DATASET
    assert provenance["manifest_sha256"] == SHA_MANIFEST
    assert provenance["modalities"] == ["behavior", "fmri"]


def test_alignment_authority_projection_rejects_noncanonical_modalities() -> None:
    plan = AlignmentPlan(_FakeNativePlan(("fmri", None)))
    with pytest.raises(ValueError, match="canonical string"):
        alignment_authority_provenance(plan)
