from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "evidence"
    / "qualify_kumar2024_gpu_fleet_synthetic.py"
)
SPEC = importlib.util.spec_from_file_location(
    "kumar2024_gpu_fleet_synthetic_qualification",
    SCRIPT_PATH,
)
assert SPEC is not None and SPEC.loader is not None
qualification = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qualification
SPEC.loader.exec_module(qualification)


def test_provider_free_qualification_is_complete_and_deterministic(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    first = qualification.run(repo_root, tmp_path / "first")
    second = qualification.run(repo_root, tmp_path / "second")

    assert first["synthetic_qualification_sha256"] == second[
        "synthetic_qualification_sha256"
    ]
    assert first["settlement_ledger_sha256"] == second["settlement_ledger_sha256"]
    assert first["accepted_artifact_count"] == 3
    assert first["infrastructure_failure_count"] == 1
    assert first["attempt_count"] == 4
    assert first["model_execution_performed"] is False
    assert first["neural_data_accessed"] is False
    assert first["scientific_outcomes_inspected"] is False
    assert first["orion_comparison_permitted"] is False

    first_bytes = (
        tmp_path / "first" / "synthetic_qualification_receipt.json"
    ).read_bytes()
    second_bytes = (
        tmp_path / "second" / "synthetic_qualification_receipt.json"
    ).read_bytes()
    assert first_bytes == second_bytes


def test_verifier_rejects_resealed_false_scientific_claim(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    output = tmp_path / "qualification"
    qualification.run(repo_root, output)

    receipt_path = output / "synthetic_qualification_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["scientific_outcomes_inspected"] = True
    payload = dict(receipt)
    payload.pop("synthetic_qualification_sha256")
    receipt["synthetic_qualification_sha256"] = qualification._identity(
        qualification.QUALIFICATION_SCHEMA,
        payload,
    )
    receipt_path.write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")

    with pytest.raises(ValueError, match="scientific_outcomes_inspected=false"):
        qualification.verify(output)


def test_runner_refuses_output_reuse(tmp_path: Path):
    repo_root = Path(__file__).parents[1]
    output = tmp_path / "qualification"
    qualification.run(repo_root, output)
    with pytest.raises(FileExistsError, match="refusing to reuse"):
        qualification.run(repo_root, output)
