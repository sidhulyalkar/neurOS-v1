"""Fail-closed, stdlib-only audit of the historical Actions CPU receipt."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = ROOT / "scripts" / "evidence"
sys.path.insert(0, str(SCRIPT_DIR))
AUDIT_PATH = SCRIPT_DIR / "verify_kumar2024_actions_qualification.py"
SPEC = importlib.util.spec_from_file_location("actions_kumar_auditor", AUDIT_PATH)
assert SPEC is not None and SPEC.loader is not None
auditor = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(auditor)

# Reuse only the separate independent verifier's *synthetic* fixture. The
# producer/worker and neurOS packages are never imported or invoked.
FIXTURE_PATH = ROOT / "tests" / "test_kumar2024_external_qualification_verifier.py"
SPEC2 = importlib.util.spec_from_file_location("external_verifier_fixture", FIXTURE_PATH)
assert SPEC2 is not None and SPEC2.loader is not None
fixture = importlib.util.module_from_spec(SPEC2)
SPEC2.loader.exec_module(fixture)


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _reseal(root: Path, **changes) -> None:
    receipt_path = root / "qualification_manifest.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt.pop("qualification_sha256", None)
    receipt.update(changes)
    receipt["qualification_sha256"] = auditor._identity(
        auditor.DOMAIN_QUALIFICATION, receipt
    )
    _write(receipt_path, receipt)


def _fixture(tmp_path: Path) -> Path:
    root, _, _ = fixture._synthetic_bundle(tmp_path)
    worker = root / "worker"
    manifest = json.loads((worker / "worker_manifest.json").read_text(encoding="utf-8"))
    hashes = json.loads((worker / "artifact_hashes.json").read_text(encoding="utf-8"))
    (root / "external_artifact_hashes.json").unlink()
    (root / "worker_run.json").write_text(
        '{"test_fixture_only":true,"balanced_accuracy":0.5}\n', encoding="utf-8"
    )
    _write(
        root / "qualification_manifest.json",
        {
            "schema_version": 1,
            "artifact_kind": "promoted_classical_worker_systems_qualification",
            "source_revision": auditor.SOURCE_REVISION,
            "binding_run_id": auditor.BINDING_RUN_ID,
            "binding_artifact_id": auditor.BINDING_ARTIFACT_ID,
            "binding_bundle_sha256": auditor.BINDING_BUNDLE_SHA256,
            "shard_spec_sha256": auditor.SHARD_SPEC_SHA256,
            "shard_id": auditor.SHARD_ID,
            "method_id": "mne-csp-lda",
            "worker_bundle_sha256": hashes["bundle_sha256"],
            "worker_shard_result_sha256": manifest["shard_result_sha256"],
            "numerical_result_interpretable": False,
            "global_analysis_performed": False,
            "external_floor_claim_generated": False,
            "orion_comparison_permitted": False,
            "claim_boundary": "synthetic noninterpretive Actions receipt",
        },
    )
    _reseal(root)
    return root


def test_accepts_independently_constructed_frozen_actions_receipt(tmp_path):
    root = _fixture(tmp_path)
    report = auditor.verify(root)
    assert report["verified"] is True
    assert report["source_revision"] == auditor.SOURCE_REVISION
    assert len(report["artifact_files_root_sha256"]) == 64
    assert report["numerical_result_interpretable"] is False
    assert report["publisher_authenticated"] is False
    assert report["externally_pinned"] is False


def test_captured_file_root_must_match_and_can_be_reused(tmp_path):
    root = _fixture(tmp_path)
    first = auditor.verify(root)
    second = auditor.verify(
        root,
        expected_files_root=first["artifact_files_root_sha256"],
        expected_qualification_sha=first["qualification_sha256"],
    )
    assert second["externally_pinned"] is True
    assert second["artifact_files_root_sha256"] == first["artifact_files_root_sha256"]


def test_captured_root_rejects_stdout_tamper_even_with_valid_worker(tmp_path):
    root = _fixture(tmp_path)
    old = auditor.verify(root)["artifact_files_root_sha256"]
    with (root / "worker_run.json").open("ab") as stream:
        stream.write(b"corrupt-output")
    assert auditor.verify(root)["artifact_files_root_sha256"] != old
    with pytest.raises(ValueError, match="externally pinned"):
        auditor.verify(root, expected_files_root=old)


def test_changed_outer_manifest_requires_new_producer_digest(tmp_path):
    root = _fixture(tmp_path)
    receipt = json.loads((root / "qualification_manifest.json").read_text())
    receipt["shard_id"] = "malicious-shard"
    _write(root / "qualification_manifest.json", receipt)
    with pytest.raises(ValueError, match="qualification content identity"):
        auditor.verify(root)


def test_resealed_wrong_shard_still_fails_authority(tmp_path):
    root = _fixture(tmp_path)
    _reseal(root, shard_spec_sha256="f" * 64)
    with pytest.raises(ValueError, match="qualification shard_spec_sha256"):
        auditor.verify(root)


def test_resealed_claim_escalation_is_rejected(tmp_path):
    root = _fixture(tmp_path)
    _reseal(root, orion_comparison_permitted=True)
    with pytest.raises(ValueError, match="qualification orion_comparison_permitted"):
        auditor.verify(root)


def test_tampered_worker_bytes_cannot_be_hidden_by_outer_reseal(tmp_path):
    root = _fixture(tmp_path)
    with (root / "worker" / "case_result.json").open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="worker file hashes"):
        auditor.verify(root)


def test_wrong_worker_row_budget_fails_even_after_resealing(tmp_path):
    root = _fixture(tmp_path)
    worker = root / "worker"
    shard = json.loads((worker / "shard_result.json").read_text())
    shard["rows"][0]["calibration_per_class"] = 99
    shard.pop("shard_result_sha256")
    shard["shard_result_sha256"] = auditor._identity(
        "neuros.kumar2024_promoted_shard_result.v1", shard
    )
    _write(worker / "shard_result.json", shard)
    _reseal(root, worker_shard_result_sha256=shard["shard_result_sha256"])
    # Even a fully recomputed outer receipt cannot excuse bad inner hashes.
    with pytest.raises(ValueError, match="worker file hashes"):
        auditor.verify(root)


def test_missing_or_extra_bytes_fail_closed(tmp_path):
    root = _fixture(tmp_path)
    (root / "worker_run.json").unlink()
    with pytest.raises(ValueError, match="file inventory"):
        auditor.verify(root)
    (root / "worker_run.json").write_text("restored")
    (root / "intruder.json").write_text("{}")
    with pytest.raises(ValueError, match="unexpected file"):
        auditor.verify(root)


def test_symlink_inside_receipt_is_rejected(tmp_path):
    root = _fixture(tmp_path)
    (root / "worker" / "extra.json").symlink_to(root / "worker_run.json")
    with pytest.raises(ValueError, match="symlink"):
        auditor.verify(root)


def test_no_scores_in_cli_output_or_error_messages(tmp_path):
    root = _fixture(tmp_path)
    good = subprocess.run(
        [sys.executable, str(AUDIT_PATH), str(root)],
        check=False, text=True, capture_output=True,
    )
    assert good.returncode == 0, good.stderr
    assert json.loads(good.stdout)["verified"] is True
    assert "balanced_accuracy" not in good.stdout
    _reseal(root, global_analysis_performed=True)
    bad = subprocess.run(
        [sys.executable, str(AUDIT_PATH), str(root)],
        check=False, text=True, capture_output=True,
    )
    assert bad.returncode != 0
    assert "balanced_accuracy" not in bad.stderr
    assert "verified" in bad.stderr
