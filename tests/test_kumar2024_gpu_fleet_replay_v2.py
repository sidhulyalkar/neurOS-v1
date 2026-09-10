from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parents[1]
GENERATOR_PATH = REPO_ROOT / "scripts" / "evidence" / "qualify_kumar2024_gpu_fleet_synthetic.py"
VERIFIER_PATH = REPO_ROOT / "scripts" / "evidence" / "verify_kumar2024_gpu_fleet_synthetic.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


generator = _load("kumar2024_synthetic_generator_for_replay", GENERATOR_PATH)
verifier = _load("kumar2024_synthetic_independent_replay", VERIFIER_PATH)


def _packet(tmp_path: Path) -> Path:
    root = tmp_path / "packet"
    generator.run(REPO_ROOT, root)
    return root


def _first_json(root: Path, bucket: str) -> Path:
    return sorted((root / "store" / bucket).rglob("*.json"))[0]


def test_independent_replay_matches_promoted_known_identities(tmp_path: Path):
    result = verifier.verify_synthetic_packet(_packet(tmp_path))
    assert result["verified"] is True
    assert result["complete"] is True
    assert result["settlement_ledger_sha256"] == (
        "293184148318914f332b3c4fe9c6658ca7a0acabadc79d3e46d63239291ec3d5"
    )
    assert result["synthetic_qualification_sha256"] == (
        "1e35e04e2bf49085b15e837d32c5abc1cee52b1f26ac872353f9089598a68df1"
    )
    assert result["lease_count"] == 3
    assert result["claim_count"] == 4
    assert result["accepted_artifact_count"] == 3
    assert result["infrastructure_failure_count"] == 1
    assert result["model_execution_performed"] is False
    assert result["neural_data_accessed"] is False
    assert result["scientific_outcomes_inspected"] is False
    assert result["orion_comparison_permitted"] is False


def test_mutated_persisted_claim_rejects(tmp_path: Path):
    root = _packet(tmp_path)
    path = _first_json(root, "claims")
    payload = json.loads(path.read_text())
    payload["worker_id"] = "tampered-worker"
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="frozen synthetic protocol"):
        verifier.verify_synthetic_packet(root)


def test_resealed_learned_state_substitution_rejects(tmp_path: Path):
    root = _packet(tmp_path)
    paths = sorted((root / "store" / "outcomes").rglob("*.json"))
    path = next(
        p for p in paths if json.loads(p.read_text()).get("event_type") == "artifact_settlement"
    )
    payload = json.loads(path.read_text())
    payload["learned_state_sha256"] = "a" * 64
    body = dict(payload)
    body.pop("outcome_sha256")
    payload["outcome_sha256"] = verifier._identity(verifier.SETTLEMENT_SCHEMA, body)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="frozen synthetic protocol"):
        verifier.verify_synthetic_packet(root)


def test_resealed_environment_substitution_rejects(tmp_path: Path):
    root = _packet(tmp_path)
    paths = sorted((root / "store" / "outcomes").rglob("*.json"))
    path = next(
        p for p in paths if json.loads(p.read_text()).get("event_type") == "artifact_settlement"
    )
    payload = json.loads(path.read_text())
    payload["environment_authority_sha256"] = "b" * 64
    body = dict(payload)
    body.pop("outcome_sha256")
    payload["outcome_sha256"] = verifier._identity(verifier.SETTLEMENT_SCHEMA, body)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="frozen synthetic protocol"):
        verifier.verify_synthetic_packet(root)


def test_missing_outcome_rejects_closed_packet(tmp_path: Path):
    root = _packet(tmp_path)
    _first_json(root, "outcomes").unlink()
    with pytest.raises(ValueError, match="file topology differs"):
        verifier.verify_synthetic_packet(root)


def test_shadow_file_and_empty_directory_reject(tmp_path: Path):
    root = _packet(tmp_path)
    shadow = root / "store" / "claims" / "shadow.json"
    shadow.write_text("{}\n")
    with pytest.raises(ValueError, match="file topology differs"):
        verifier.verify_synthetic_packet(root)

    shadow.unlink()
    (root / "store" / "attempts" / "unexpected-empty-dir").mkdir()
    with pytest.raises(ValueError, match="directory topology differs"):
        verifier.verify_synthetic_packet(root)


def test_resealed_false_top_level_claim_rejects(tmp_path: Path):
    root = _packet(tmp_path)
    path = root / "synthetic_qualification_receipt.json"
    payload = json.loads(path.read_text())
    payload["scientific_outcomes_inspected"] = True
    body = dict(payload)
    body.pop("synthetic_qualification_sha256")
    payload["synthetic_qualification_sha256"] = verifier._identity(
        verifier.QUALIFICATION_SCHEMA, body
    )
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")
    with pytest.raises(ValueError, match="independent replay|scientific"):
        verifier.verify_synthetic_packet(root)


def test_duplicate_json_key_rejects(tmp_path: Path):
    root = _packet(tmp_path)
    path = _first_json(root, "claims")
    text = path.read_text()
    insert = text.rfind("}")
    text = text[:insert] + ',\n  "worker_id": "duplicate"\n' + text[insert:]
    path.write_text(text)
    with pytest.raises(ValueError, match="duplicate JSON key"):
        verifier.verify_synthetic_packet(root)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_symlink_injection_rejects(tmp_path: Path):
    root = _packet(tmp_path)
    target = tmp_path / "outside.json"
    target.write_text("{}\n")
    link = root / "store" / "claims" / "injected.json"
    os.symlink(target, link)
    with pytest.raises(ValueError, match="symlink"):
        verifier.verify_synthetic_packet(root)


def test_root_symlink_rejects(tmp_path: Path):
    if not hasattr(os, "symlink"):
        pytest.skip("symlinks unavailable")
    root = _packet(tmp_path)
    alias = tmp_path / "alias"
    os.symlink(root, alias, target_is_directory=True)
    with pytest.raises(ValueError, match="root may not be a symlink"):
        verifier.verify_synthetic_packet(alias)
