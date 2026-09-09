from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
RUNNER_PATH = ROOT / "scripts" / "evidence" / "qualify_kumar2024_gpu_fleet_synthetic.py"
VERIFIER_PATH = ROOT / "scripts" / "evidence" / "verify_kumar2024_gpu_fleet_synthetic.py"


def load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = load("synthetic_fleet_runner_for_verifier", RUNNER_PATH)
verifier = load("synthetic_fleet_independent_verifier", VERIFIER_PATH)


def make_store(tmp_path: Path) -> tuple[Path, dict]:
    root = tmp_path / "packet"
    receipt = runner.run_synthetic_qualification(root)
    return root, receipt


def test_independent_verifier_replays_complete_store(tmp_path: Path):
    root, receipt = make_store(tmp_path)
    checked = verifier.verify_synthetic_store(root)
    assert checked["verified"] is True
    assert checked["complete"] is True
    assert checked["synthetic_qualification_sha256"] == receipt[
        "synthetic_qualification_sha256"
    ]
    assert checked["settlement_ledger_sha256"] == receipt["settlement_ledger_sha256"]
    assert checked["lease_count"] == 4
    assert checked["claim_count"] == 5
    assert checked["accepted_artifact_count"] == 4
    assert checked["infrastructure_failure_count"] == 1
    assert checked["scientific_outcomes_inspected"] is False
    assert checked["orion_comparison_permitted"] is False


def test_independent_verifier_rejects_tampered_claim(tmp_path: Path):
    root, _ = make_store(tmp_path)
    path = sorted(root.glob("claims/*/attempt-*.json"))[0]
    payload = json.loads(path.read_text())
    payload["provider_run_id"] = "shadow-provider-run"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="frozen protocol"):
        verifier.verify_synthetic_store(root)


def test_independent_verifier_rejects_tampered_artifact(tmp_path: Path):
    root, _ = make_store(tmp_path)
    for path in sorted(root.glob("outcomes/*/attempt-*.json")):
        payload = json.loads(path.read_text())
        if payload["event_type"] == "artifact_settlement":
            payload["learned_state_sha256"] = "0" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            break
    else:
        raise AssertionError("expected artifact settlement")
    with pytest.raises(ValueError, match="frozen protocol"):
        verifier.verify_synthetic_store(root)


def test_independent_verifier_rejects_missing_and_extra_records(tmp_path: Path):
    root, _ = make_store(tmp_path)
    outcome = sorted(root.glob("outcomes/*/attempt-*.json"))[0]
    outcome.unlink()
    with pytest.raises(ValueError, match="file topology differs"):
        verifier.verify_synthetic_store(root)

    root2, _ = make_store(tmp_path / "second")
    extra = root2 / "outcomes" / "shadow.json"
    extra.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="file topology differs"):
        verifier.verify_synthetic_store(root2)


def test_independent_verifier_rejects_receipt_tamper(tmp_path: Path):
    root, _ = make_store(tmp_path)
    path = root / "synthetic-qualification.json"
    payload = json.loads(path.read_text())
    payload["accepted_artifact_count"] = 3
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="independently replayed store"):
        verifier.verify_synthetic_store(root)


def test_independent_verifier_rejects_duplicate_json_keys(tmp_path: Path):
    root, _ = make_store(tmp_path)
    path = sorted(root.glob("attempts/*/attempt-*/provider-invocation.json"))[0]
    original = json.loads(path.read_text())
    path.write_text(
        "{"
        '"schema_version":1,'
        f'"provider":{json.dumps(original["provider"])},'
        f'"provider":{json.dumps(original["provider"])},'
        f'"provider_run_id":{json.dumps(original["provider_run_id"])},'
        '"scientific_execution_performed":false'
        "}",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verifier.verify_synthetic_store(root)


def test_independent_verifier_rejects_symlink_injection(tmp_path: Path):
    root, _ = make_store(tmp_path)
    target = root / "synthetic-qualification.json"
    link = root / "shadow-link.json"
    try:
        os.symlink(target, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink file is forbidden"):
        verifier.verify_synthetic_store(root)
