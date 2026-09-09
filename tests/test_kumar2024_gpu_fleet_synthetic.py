from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).parents[1]
    / "scripts"
    / "evidence"
    / "qualify_kumar2024_gpu_fleet_synthetic.py"
)
SPEC = importlib.util.spec_from_file_location("synthetic_fleet_qualification", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
qual = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = qual
SPEC.loader.exec_module(qual)


def test_synthetic_fleet_completes_with_one_infrastructure_retry(tmp_path: Path):
    receipt = qual.run_synthetic_qualification(tmp_path / "run")
    assert receipt["complete"] is True
    assert receipt["lease_count"] == 4
    assert receipt["claim_count"] == 5
    assert receipt["accepted_artifact_count"] == 4
    assert receipt["infrastructure_failure_count"] == 1
    assert receipt["pending_lease_count"] == 0
    assert receipt["scientific_execution_performed"] is False
    assert receipt["scientific_outcomes_inspected"] is False
    assert receipt["numerical_result_interpretable"] is False
    assert receipt["orion_comparison_permitted"] is False


def test_synthetic_receipt_is_path_and_iteration_order_independent(tmp_path: Path):
    first = qual.run_synthetic_qualification(tmp_path / "first")
    second = qual.run_synthetic_qualification(tmp_path / "second")
    assert first == second


def test_synthetic_store_has_write_once_attempt_topology_and_no_scores(tmp_path: Path):
    root = tmp_path / "run"
    qual.run_synthetic_qualification(root)

    claims = sorted(root.glob("claims/*/attempt-*.json"))
    attempts = sorted(root.glob("attempts/*/attempt-*"))
    outcomes = sorted(root.glob("outcomes/*/attempt-*.json"))
    assert len(claims) == 5
    assert len(attempts) == 5
    assert len(outcomes) == 5

    for path in [*claims, *outcomes, root / "synthetic-qualification.json"]:
        payload = json.loads(path.read_text())
        text = json.dumps(payload, sort_keys=True).lower()
        for forbidden in (
            "balanced_accuracy",
            '"accuracy"',
            '"loss"',
            "predictions",
            "probabilities",
            "scientific_score",
        ):
            assert forbidden not in text
