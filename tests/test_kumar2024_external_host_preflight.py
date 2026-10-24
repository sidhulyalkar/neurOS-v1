"""Offline adversarial contracts for read-only Kumar2024 external preflight."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from unittest.mock import patch

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/evidence/preflight_kumar2024_external_host.py"
SPEC = importlib.util.spec_from_file_location("nsq_readonly_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


def _repo(root: Path) -> None:
    runner = root / "scripts/evidence/run_kumar2024_external_qualification.sh"
    runner.parent.mkdir(parents=True)
    runner.write_text("#!/bin/bash\nexit 7\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", str(runner.relative_to(root))], check=True)
    subprocess.run(
        ["git", "-C", str(root), "-c", "user.email=test@example.com",
         "-c", "user.name=Test", "commit", "-qm", "fixture"], check=True
    )


def _archive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, wrong_role: bool = False
) -> Path:
    """Synthetic authority fixture: no raw EEG, model state, or frozen archive bytes."""
    others = [
        {"shard_spec_sha256": f"{index:064x}", "shard_id": f"other-{index}"}
        for index in range(1349)
    ]
    selected = {
        "shard_spec_sha256": preflight.EXPECTED_SHARD,
        "shard_id": preflight.EXPECTED_SHARD_ID,
        "method_id": "braindecode-eegnet" if wrong_role else "mne-csp-lda",
        "model_seed": None,
        "budgets_per_class": [0, 1, 2, 5, 10],
    }
    files = {
        "binding_manifest.json": json.dumps({
            "source_revision": preflight.EXPECTED_SOURCE,
            "counts": preflight.EXPECTED_COUNTS,
            "model_execution_performed": False,
            "final_assessment_predictions_generated": False,
            "final_assessment_metrics_generated": False,
        }, sort_keys=True).encode(),
        "execution_plan.json": json.dumps(
            {"template": {"shards": others + [selected]}}, sort_keys=True
        ).encode(),
    }
    hashes = {
        "schema_version": 1,
        "bundle_sha256": preflight.EXPECTED_BUNDLE_SHA,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    archive = tmp_path / "synthetic-binding.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in files.items():
            bundle.writestr(name, data)
        bundle.writestr("artifact_hashes.json", json.dumps(hashes, sort_keys=True))
    monkeypatch.setattr(
        preflight, "EXPECTED_ZIP_SHA", hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    return archive


def test_synthetic_archive_verifies_authority_not_efficacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = preflight.inspect_archive(_archive(tmp_path, monkeypatch))
    assert report["archive_integrity"] == "verified"
    assert report["authorized_shard"] == preflight.EXPECTED_SHARD
    assert report["counts"]["execution_shards"] == 1350
    assert "no execution" in report["claim_boundary"]


def test_archive_byte_tampering_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = _archive(tmp_path, monkeypatch)
    data = bytearray(archive.read_bytes())
    data[len(data) // 3] ^= 1
    archive.write_bytes(data)
    with pytest.raises(ValueError, match="archive sha256 mismatch"):
        preflight.inspect_archive(archive)


def test_wrong_authorized_shard_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(ValueError, match="authorized shard role"):
        preflight.inspect_archive(_archive(tmp_path, monkeypatch, wrong_role=True))


def test_missing_archive_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="archive not found"):
        preflight.inspect_archive(tmp_path / "absent.zip")


def test_host_probe_does_not_create_attempt_directory(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    attempt = tmp_path / "fresh-attempt"
    before = {
        file.relative_to(tmp_path)
        for file in tmp_path.rglob("*")
        if ".git" not in file.parts
    }
    runner_text = (repo / "scripts/evidence/run_kumar2024_external_qualification.sh").read_text()
    actual_which = shutil.which
    with patch.object(
        preflight.shutil, "which",
        side_effect=lambda name: "/fake/uv" if name == "uv" else actual_which(name),
    ):
        report = preflight.inspect_host(repo, attempt, "local-wsl")
    assert report["transport_host_eligible"]
    assert not attempt.exists()
    after = {
        file.relative_to(tmp_path)
        for file in tmp_path.rglob("*")
        if ".git" not in file.parts
    }
    assert after == before
    assert (repo / "scripts/evidence/run_kumar2024_external_qualification.sh").read_text() == runner_text
    assert any("historical source" in w for w in report["capacity_warnings"])


def test_existing_write_once_attempt_is_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    attempt = tmp_path / "existing-attempt"
    attempt.mkdir()
    report = preflight.inspect_host(repo, attempt, "local-wsl")
    assert not report["transport_host_eligible"]
    assert any("already exists" in error for error in report["hard_blockers"])


def test_dirty_runner_and_wrong_architecture_are_rejected(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    runner = repo / "scripts/evidence/run_kumar2024_external_qualification.sh"
    runner.write_text(runner.read_text() + "# uncommitted drift\n")
    with patch.object(preflight.platform, "machine", return_value="arm64"):
        report = preflight.inspect_host(repo, tmp_path / "new", "local-wsl")
    assert not report["transport_host_eligible"]
    assert any("Linux/x86_64" in error for error in report["hard_blockers"])
    assert any("modifications" in error for error in report["hard_blockers"])


def test_missing_uv_prevents_execution_readiness(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    actual_which = shutil.which
    with patch.object(
        preflight.shutil, "which",
        side_effect=lambda name: None if name == "uv" else actual_which(name),
    ):
        report = preflight.inspect_host(repo, tmp_path / "new", "local-wsl")
    assert "required command unavailable: uv" in report["hard_blockers"]
