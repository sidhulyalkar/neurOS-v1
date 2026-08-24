from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from neuros.foundation_models import GroupedEvaluationData


def _load_runner():
    root = Path(__file__).resolve().parents[3]
    path = root / "scripts" / "evidence" / "run_moabb_longitudinal.py"
    spec = importlib.util.spec_from_file_location("neuros_longitudinal_runner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synthetic_longitudinal_eeg() -> GroupedEvaluationData:
    rng = np.random.default_rng(77)
    X = []
    y = []
    metadata = []
    for session_index, session in enumerate(("0", "1", "2")):
        for label_index, label in enumerate(("left_hand", "right_hand")):
            for trial in range(10):
                signal = rng.normal(scale=0.8, size=(4, 128))
                # Give the transparent baseline a stable but session-shifted signal.
                signal[0, 24:72] += (1.0 if label_index else -1.0) + 0.08 * session_index
                X.append(signal)
                y.append(label)
                metadata.append(
                    {
                        "subject": "1",
                        "session": session,
                        "run": f"run-{trial // 5}",
                    }
                )
    return GroupedEvaluationData.from_moabb_result(
        (np.asarray(X, dtype=np.float64), np.asarray(y), metadata),
        dataset_id="moabb-kumar2024",
    )


def _patch_data_sources(runner, monkeypatch, bundle):
    monkeypatch.setattr(
        runner,
        "_dataset_and_paradigm",
        lambda *args, **kwargs: (object(), object()),
    )
    monkeypatch.setattr(runner, "collect_moabb", lambda *args, **kwargs: bundle)


def test_runner_writes_consistent_prior_only_evidence_bundle(tmp_path, monkeypatch):
    runner = _load_runner()
    bundle = _synthetic_longitudinal_eeg()
    _patch_data_sources(runner, monkeypatch, bundle)

    output = tmp_path / "evidence"
    code = runner.main(
        [
            "--dataset",
            "kumar2024",
            "--subjects",
            "1",
            "--budgets",
            "0,1",
            "--history-policy",
            "prior",
            "--csp-components",
            "2",
            "--output",
            str(output),
        ]
    )
    assert code == 0

    manifest = json.loads((output / "study_manifest.json").read_text())
    summary = json.loads((output / "summary.json").read_text())
    hashes = json.loads((output / "artifact_hashes.json").read_text())
    report = (output / "report.md").read_text()
    with (output / "results.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert manifest["history_policy"] == "prior"
    assert manifest["allow_incomplete_budgets"] is False
    assert [item["held_out_session"] for item in manifest["splits"]] == ["1", "2"]
    assert manifest["splits"][0]["source_sessions"] == ["0"]
    assert manifest["splits"][1]["source_sessions"] == ["0", "1"]
    assert manifest["splits"][0]["observed_session_order"] == ["0", "1", "2"]

    assert len(rows) == 4  # two eligible held-out sessions x two budgets
    assert {row["history_policy"] for row in rows} == {"prior"}
    assert {row["calibration_per_class"] for row in rows} == {"0", "1"}
    assert len(
        {
            row["calibration_split_fingerprint"]
            for row in rows
            if row["held_out_session"] == "1"
        }
    ) == 1

    assert summary["paired_case_sets_identical"] is True
    assert [point["calibration_per_class"] for point in summary["curve"]] == [0, 1]
    assert len({point["case_set_fingerprint"] for point in summary["curve"]}) == 1
    assert "future sessions excluded" in report
    assert "paired across identical subject-session cases" in report
    assert set(hashes["sha256"]) == {
        "results.csv",
        "summary.json",
        "study_manifest.json",
        "report.md",
    }

    for name in hashes["sha256"]:
        assert len(hashes["sha256"][name]) == 64


def test_runner_fails_closed_if_requested_frontier_would_be_unpaired(tmp_path, monkeypatch):
    runner = _load_runner()
    bundle = _synthetic_longitudinal_eeg()
    _patch_data_sources(runner, monkeypatch, bundle)

    # With 10 examples per class and a 0.5 frozen evaluation fraction, the
    # balanced calibration maximum is 5/class. A promoted 6/class point must
    # fail instead of dropping this subject/session from the right side of the curve.
    with pytest.raises(RuntimeError, match="not paired"):
        runner.main(
            [
                "--dataset",
                "kumar2024",
                "--subjects",
                "1",
                "--budgets",
                "0,6",
                "--history-policy",
                "prior",
                "--csp-components",
                "2",
                "--output",
                str(tmp_path / "strict-failure"),
            ]
        )
