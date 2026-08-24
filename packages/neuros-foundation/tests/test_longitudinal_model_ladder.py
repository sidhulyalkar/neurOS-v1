from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from neuros.foundation_models import GroupedEvaluationData

pytest.importorskip("torch")
pytest.importorskip("neuros_sourceweigher")
pytest.importorskip("mne")


def _load_runner():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "evidence" / "run_moabb_model_ladder.py"
    spec = importlib.util.spec_from_file_location("neuros_model_ladder_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture() -> GroupedEvaluationData:
    rng = np.random.default_rng(1717)
    X = []
    y = []
    metadata = []
    for session_index, session in enumerate(("0", "1", "2")):
        for label_index, label in enumerate(("left_hand", "right_hand")):
            for trial in range(8):
                signal = rng.normal(scale=0.35, size=(4, 128)).astype(np.float32)
                direction = -1.0 if label_index == 0 else 1.0
                signal[label_index, 18:94] += direction * (1.2 + 0.05 * session_index)
                signal[3] += 0.04 * session_index
                X.append(signal)
                y.append(label)
                metadata.append(
                    {
                        "subject": "1",
                        "session": session,
                        "run": f"r-{trial // 4}",
                    }
                )
    return GroupedEvaluationData.from_moabb_result(
        (np.asarray(X), np.asarray(y), metadata),
        dataset_id="moabb-kumar2024",
    )


def test_complete_model_ladder_writes_one_authoritative_bundle(tmp_path, monkeypatch):
    runner = _load_runner()
    data = _fixture()

    fake_spec = SimpleNamespace(
        key="kumar2024",
        class_name="Kumar2024",
        source_id="moabb-kumar2024",
        case_metadata=lambda subject: {
            "subject": int(subject),
            "original_protocol": "GR" if int(subject) <= 9 else "PAR",
        },
    )
    monkeypatch.setattr(
        runner,
        "build_moabb_longitudinal_dataset",
        lambda *args, **kwargs: (fake_spec, object(), object()),
    )
    monkeypatch.setattr(runner, "collect_moabb", lambda *args, **kwargs: data)
    monkeypatch.setattr(
        runner,
        "validate_observed_sessions",
        lambda _spec, observed: tuple(observed),
    )

    output = tmp_path / "ladder"
    methods = (
        "csp-lda,eegnet,eeg-conformer,frozen-eegnet,frozen-eeg-conformer,"
        "sourceweigher-eegnet,sourceweigher-eeg-conformer"
    )
    code = runner.main(
        [
            "--dataset",
            "kumar2024",
            "--subjects",
            "1",
            "--held-out-sessions",
            "2",
            "--methods",
            methods,
            "--model-seeds",
            "101",
            "--budgets",
            "0,1,2",
            "--history-policy",
            "prior",
            "--epochs",
            "1",
            "--batch-size",
            "16",
            "--device",
            "cpu",
            "--csp-components",
            "2",
            "--output",
            str(output),
        ]
    )
    assert code == 0

    manifest = json.loads((output / "study_manifest.json").read_text())
    authority = json.loads((output / "split_authority.json").read_text())
    method_runs = json.loads((output / "method_runs.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    report = (output / "report.md").read_text()
    with (output / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    expected_methods = {
        "csp-lda",
        "eegnet",
        "eeg-conformer",
        "frozen-eegnet",
        "frozen-eeg-conformer",
        "sourceweigher-eegnet",
        "sourceweigher-eeg-conformer",
    }
    assert set(manifest["methods"]) == expected_methods
    assert manifest["history_policy"] == "prior"
    assert manifest["subjects"] == [1]
    assert len(authority["cases"]) == 1
    case = authority["cases"][0]
    assert case["source_group_values"] == ["0", "1"]
    assert case["held_out_values"] == ["2"]
    assert len(case["processed_data_sha256"]) == 64

    assert len(method_runs["runs"]) == 7
    assert {item["requested_method_id"] for item in method_runs["runs"]} == expected_methods
    assert all(item["status"] == "ok" for item in method_runs["runs"])

    assert {row["method_id"] for row in rows} == expected_methods
    assert {row["original_protocol"] for row in rows} == {"GR"}
    assert {row["held_out_session"] for row in rows} == {"2"}
    assert {row["authority_fingerprint"] for row in rows} == {
        case["authority_fingerprint"]
    }
    assert {row["partition_fingerprint"] for row in rows} == {
        case["partition_fingerprint"]
    }
    assert {row["calibration_split_fingerprint"] for row in rows} == {
        case["calibration_split_fingerprint"]
    }

    unavailable = [
        row for row in rows if row["status"] == "unavailable_no_target_observations"
    ]
    assert {row["method_id"] for row in unavailable} == {
        "sourceweigher-eegnet",
        "sourceweigher-eeg-conformer",
    }
    assert {row["calibration_per_class"] for row in unavailable} == {"0"}

    ok_rows = [row for row in rows if row["status"] == "ok"]
    assert all(0.0 <= float(row["balanced_accuracy"]) <= 1.0 for row in ok_rows)
    assert summary["failed_rows"] == 0
    assert summary["unexpected_unavailable_rows"] == 0
    assert summary["promotion_ready_descriptive"] is True
    assert {row["original_protocol"] for row in summary["cohort_budget_summary"]} == {"GR"}
    assert {row["held_out_session"] for row in summary["target_session_budget_summary"]} == {"2"}

    full_auc_methods = {row["method_id"] for row in summary["complete_frontier_auc"]}
    assert "sourceweigher-eegnet" not in full_auc_methods
    assert "sourceweigher-eeg-conformer" not in full_auc_methods
    positive_auc_methods = {
        row["method_id"] for row in summary["positive_budget_adaptation_auc"]
    }
    assert expected_methods <= positive_auc_methods

    paired = {row["method_id"]: row for row in summary["paired_case_set_audit"]}
    assert set(paired) == expected_methods
    assert all(row["paired_across_supported_budgets"] for row in paired.values())
    assert paired["sourceweigher-eegnet"]["supported_budgets"] == [1, 2]

    assert "Descriptive promotion gate: **PASS**" in report
    assert "Positive-budget adaptation AUC" in report
    assert "do not receive a fabricated full-frontier AUC" in report

    assert set(hashes["sha256"]) == {
        "study_manifest.json",
        "split_authority.json",
        "method_runs.json",
        "results.csv",
        "summary.json",
        "report.md",
    }
    for digest in hashes["sha256"].values():
        assert len(digest) == 64
