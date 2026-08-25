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
pytest.importorskip("braindecode")


def _load_runner():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "evidence" / "run_moabb_braindecode_pair.py"
    spec = importlib.util.spec_from_file_location("neuros_braindecode_pair_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture() -> GroupedEvaluationData:
    rng = np.random.default_rng(9090)
    X: list[np.ndarray] = []
    y: list[str] = []
    metadata: list[dict[str, str]] = []
    for session_index, session in enumerate(("0", "1", "2")):
        for label_index, label in enumerate(("left_hand", "right_hand")):
            for trial in range(8):
                signal = rng.normal(scale=0.3, size=(4, 128)).astype(np.float32)
                signal[label_index, 16:104] += (-1.0 if label_index == 0 else 1.0) * (
                    1.1 + 0.03 * session_index
                )
                X.append(signal)
                y.append(label)
                metadata.append(
                    {
                        "subject": "1",
                        "session": session,
                        "run": f"run-{trial // 4}",
                    }
                )
    return GroupedEvaluationData.from_moabb_result(
        (np.asarray(X), np.asarray(y), metadata),
        dataset_id="moabb-kumar2024",
    )


def test_paired_runner_writes_self_consistent_artifact_bundle(tmp_path, monkeypatch):
    runner = _load_runner()
    data = _fixture()
    fake_spec = SimpleNamespace(
        key="kumar2024",
        class_name="Kumar2024",
        source_id="moabb-kumar2024",
        case_metadata=lambda subject: {
            "subject": int(subject),
            "original_protocol": "GR",
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

    output = tmp_path / "paired"
    code = runner.main(
        [
            "--dataset",
            "kumar2024",
            "--subjects",
            "1",
            "--held-out-sessions",
            "2",
            "--model-seeds",
            "101",
            "--budgets",
            "0,1",
            "--history-policy",
            "prior",
            "--resample",
            "128",
            "--epochs",
            "1",
            "--batch-size",
            "8",
            "--device",
            "cpu",
            "--output",
            str(output),
        ]
    )
    assert code == 0

    manifest = json.loads((output / "study_manifest.json").read_text())
    authority = json.loads((output / "split_authority.json").read_text())
    native = json.loads((output / "native_runs.json").read_text())
    external = json.loads((output / "external_runs.json").read_text())
    paired = json.loads((output / "paired_runs.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    report = (output / "report.md").read_text()
    with (output / "paired_results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert manifest["study"] == "native_vs_braindecode_eegnet_longitudinal_pair"
    assert manifest["dataset_id"] == "moabb-kumar2024"
    assert manifest["history_policy"] == "prior"
    assert manifest["resample_hz"] == 128.0
    assert manifest["budgets_per_class"] == [0, 1]
    assert manifest["model_seeds"] == [101]
    assert manifest["native_method_id"] == "eegnet"
    assert manifest["external_method_id"] == "braindecode-eegnet"

    assert len(authority["cases"]) == 1
    frozen = authority["cases"][0]
    assert frozen["source_group_values"] == ["0", "1"]
    assert frozen["held_out_values"] == ["2"]
    assert len(frozen["processed_data_sha256"]) == 64

    assert len(native["runs"]) == 1
    assert len(external["runs"]) == 1
    assert len(paired["runs"]) == 1
    assert external["runs"][0]["method_spec"]["sample_rate_hz"] == 128.0
    assert external["runs"][0]["upstream_version"].startswith("1.7.")

    assert len(rows) == 2
    assert {row["calibration_per_class"] for row in rows} == {"0", "1"}
    assert {row["subject"] for row in rows} == {"1"}
    assert {row["original_protocol"] for row in rows} == {"GR"}
    assert {row["held_out_session"] for row in rows} == {"2"}
    assert {row["authority_fingerprint"] for row in rows} == {
        frozen["authority_fingerprint"]
    }
    assert all(len(row["native_model_state_sha256"]) == 64 for row in rows)
    assert all(len(row["external_model_state_sha256"]) == 64 for row in rows)
    assert all(row["external_representation_evidence_available"] == "False" for row in rows)
    assert all(row["external_mechanistic_evidence_available"] == "False" for row in rows)

    assert summary["descriptive_only"] is True
    assert summary["paired_case_set_constant_across_budgets"] is True
    assert len(summary["case_set_fingerprints"]) == 1
    assert [item["calibration_per_class"] for item in summary["budget_summary"]] == [0, 1]
    assert all(item["n_cases"] == 1 for item in summary["budget_summary"])
    assert all(item["n_seed_pairs"] == 1 for item in summary["budget_summary"])

    assert "same serialized source/calibration/evaluation authority" in report
    assert "identical architecture" in report
    assert "participant-independent inferential" not in report

    expected_artifacts = {
        "study_manifest.json",
        "split_authority.json",
        "native_runs.json",
        "external_runs.json",
        "paired_runs.json",
        "paired_results.csv",
        "summary.json",
        "report.md",
    }
    assert set(hashes["sha256"]) == expected_artifacts
    assert all(len(digest) == 64 for digest in hashes["sha256"].values())
