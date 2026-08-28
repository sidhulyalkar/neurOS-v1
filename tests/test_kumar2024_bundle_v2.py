from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from neuros.evidence.kumar2024 import (
    Kumar2024PreprocessingSpec,
    _identity_sha256,
    _preprocessing_authority,
    build_dataset_lineage,
    pilot_config,
    verify_bundle,
)
from neuros.evidence.kumar2024_materialized_study import BUNDLE_FILES_V2
from neuros.foundation_models.moabb_epochs import MOABBEpochDescriptor


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _descriptor() -> MOABBEpochDescriptor:
    return MOABBEpochDescriptor(
        channel_names=("C3", "Cz", "C4"),
        channel_types=("eeg", "eeg", "eeg"),
        sampling_rate_hz=512.0,
        n_times=2560,
        epoch_start_s=0.0,
        epoch_end_s=2559 / 512.0,
        event_id=(("left_hand", 1), ("right_hand", 2)),
        n_trials=80,
    )


def test_lineage_binds_raw_materialization_without_mislabeling_archive_content_hash():
    versions = {"moabb": "1.5.0", "mne": "1.6.0"}
    preprocessing = _preprocessing_authority(
        Kumar2024PreprocessingSpec(),
        _descriptor(),
        versions,
    )
    lineage = build_dataset_lineage(
        config=pilot_config(),
        preprocessing_authority=preprocessing,
        versions=versions,
        raw_materialization_sha256="a" * 64,
    )
    assert lineage.content_sha256 is None
    assert lineage.metadata["raw_materialization_sha256"] == "a" * 64
    assert "exact consumed" in lineage.metadata["raw_materialization_scope"]
    assert "not a canonical upstream archive digest" in lineage.metadata[
        "content_sha256_reason"
    ]


def test_bundle_v2_verifies_and_preserves_strict_file_set(tmp_path: Path):
    for index, name in enumerate(BUNDLE_FILES_V2):
        (tmp_path / name).write_text(f"fixture-v2-{index}\n", encoding="utf-8")
    files = {name: _sha(tmp_path / name) for name in BUNDLE_FILES_V2}
    bundle_sha = _identity_sha256(
        "neuros.nsq_kumar2024_bundle.v2", {"files": files}
    )
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "files": files,
                "bundle_sha256": bundle_sha,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    verified = verify_bundle(tmp_path)
    assert verified["verified"] is True
    assert verified["schema_version"] == 2
    assert verified["bundle_sha256"] == bundle_sha

    payload = json.loads((tmp_path / "artifact_hashes.json").read_text())
    payload["files"].pop("observation_roles.json")
    payload["bundle_sha256"] = _identity_sha256(
        "neuros.nsq_kumar2024_bundle.v2", {"files": payload["files"]}
    )
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps(payload, sort_keys=True), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="file set"):
        verify_bundle(tmp_path)


def test_bundle_v2_detects_tampering(tmp_path: Path):
    for index, name in enumerate(BUNDLE_FILES_V2):
        (tmp_path / name).write_text(f"fixture-v2-{index}\n", encoding="utf-8")
    files = {name: _sha(tmp_path / name) for name in BUNDLE_FILES_V2}
    bundle_sha = _identity_sha256(
        "neuros.nsq_kumar2024_bundle.v2", {"files": files}
    )
    (tmp_path / "artifact_hashes.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "files": files,
                "bundle_sha256": bundle_sha,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (tmp_path / "materialization.json").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_bundle(tmp_path)
