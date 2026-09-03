from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from neuros.dataset import DataWindow, Dataset
from orion.scientific import LineageCompleteness


SHA_MANIFEST = "a" * 64
SHA_SOURCE = "b" * 64
SHA_DATASET = "c" * 64


class _FakeNativeDataset:
    dataset_id = "fixture"
    manifest_sha256 = SHA_MANIFEST
    declared_dataset_content_sha256 = SHA_DATASET
    record_count = 1

    def __init__(self) -> None:
        self.verified_dataset_content_sha256 = None

    def verify_content(self) -> str:
        self.verified_dataset_content_sha256 = SHA_DATASET
        return SHA_DATASET


def _window() -> DataWindow:
    native = SimpleNamespace(
        record_id="r1",
        subject="sub-01",
        modality="fmri",
        start_frame=4,
        end_frame_exclusive=8,
        shape=[4, 3],
        sampling_hz=0.5,
        manifest_sha256=SHA_MANIFEST,
        source_size_bytes=4096,
        declared_source_sha256=SHA_SOURCE,
        verified_source_sha256=SHA_SOURCE,
        source_verification_state="verified_at_open",
        declared_dataset_content_sha256=SHA_DATASET,
        verified_dataset_content_sha256=SHA_DATASET,
        record_byte_start=128,
        record_byte_end_exclusive=512,
    )
    return DataWindow(native)


def test_data_window_provenance_exposes_exact_intervals_and_content_identity() -> None:
    provenance = _window().provenance
    assert provenance["start_frame"] == 4
    assert provenance["end_frame_exclusive"] == 8
    assert provenance["manifest_sha256"] == SHA_MANIFEST
    assert provenance["declared_source_sha256"] == SHA_SOURCE
    assert provenance["verified_source_sha256"] == SHA_SOURCE
    assert provenance["source_verification_state"] == "verified_at_open"
    assert provenance["declared_dataset_content_sha256"] == SHA_DATASET
    assert provenance["verified_dataset_content_sha256"] == SHA_DATASET
    assert provenance["record_byte_interval"] == {"start": 128, "end_exclusive": 512}
    assert provenance["window_frame_interval"] == {"start": 4, "end_exclusive": 8}


def test_orion_bridge_requires_explicit_whole_dataset_verification() -> None:
    dataset = Dataset(_FakeNativeDataset(), Path("/fixture"))
    with pytest.raises(ValueError, match="fully verified native dataset"):
        dataset.to_orion_lineage(upstream_source="fixture-source")


def test_orion_bridge_never_promotes_lineage_completeness_from_local_hashes() -> None:
    dataset = Dataset(_FakeNativeDataset(), Path("/fixture"))
    assert dataset.verify_content() == SHA_DATASET
    lineage = dataset.to_orion_lineage(
        upstream_source="fixture-source",
        sampling_assumptions={"sampling_rate_hz": 0.5},
        preprocessing_history=("external preprocessing unknown",),
    )
    assert lineage.content_sha256 == SHA_DATASET
    assert lineage.lineage_completeness is LineageCompleteness.UNKNOWN
    assert lineage.metadata["neuros_runtime"]["manifest_sha256"] == SHA_MANIFEST
    assert (
        lineage.metadata["neuros_runtime"]["verified_dataset_content_sha256"]
        == SHA_DATASET
    )


def test_orion_bridge_reserves_runtime_metadata_namespace() -> None:
    dataset = Dataset(_FakeNativeDataset(), Path("/fixture"))
    dataset.verify_content()
    with pytest.raises(ValueError, match="reserved"):
        dataset.to_orion_lineage(
            upstream_source="fixture-source",
            metadata={"neuros_runtime": {"spoofed": True}},
        )
