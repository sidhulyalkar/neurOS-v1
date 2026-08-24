from __future__ import annotations

import json

import numpy as np
import pytest

from neuros.foundation_models import (
    GroupedEvaluationData,
    collect_moabb,
    find_evidence_sources,
    get_evidence_source,
    hold_out_groups,
)


def _moabb_like_result():
    X = np.arange(12 * 3 * 8, dtype=np.float64).reshape(12, 3, 8)
    y = np.asarray(["left", "right"] * 6)
    metadata = []
    for subject in ("1", "2"):
        for session in ("0", "1", "2"):
            for trial in range(2):
                metadata.append(
                    {
                        "subject": subject,
                        "session": session,
                        "run": f"run-{trial}",
                    }
                )
    return X, y, metadata


def test_curated_catalog_answers_distinct_real_world_questions():
    wang = get_evidence_source("moabb-wang2026")
    assert wang.subjects == 39
    assert wang.sessions == 5
    assert "online_bci" in wang.roles

    invasive = find_evidence_sources(modality="intracortical")
    assert {source.id for source in invasive} == {"falcon-h1", "falcon-h2"}

    longitudinal = find_evidence_sources(role="longitudinal_bci")
    assert {source.id for source in longitudinal} == {
        "moabb-wang2026",
        "moabb-kumar2024",
        "moabb-ma2020",
    }


def test_from_moabb_result_preserves_subject_session_and_recording_identity():
    data = GroupedEvaluationData.from_moabb_result(
        _moabb_like_result(),
        dataset_id="fixture",
    )
    assert data.X.shape == (12, 3, 8)
    assert set(data.groups) == {"subject", "session", "run", "recording"}
    assert data.groups["recording"][0] == "1/0"
    assert data.groups["recording"][-1] == "2/2"


def test_session_holdout_is_disjoint_and_manifest_is_deterministic():
    data = GroupedEvaluationData.from_moabb_result(
        _moabb_like_result(),
        dataset_id="fixture",
    )
    partition = hold_out_groups(data, split_unit="session", held_out_values=["2"])
    assert len(partition.train_indices) == 8
    assert len(partition.test_indices) == 4
    assert set(data.groups["session"][partition.train_indices]) == {"0", "1"}
    assert set(data.groups["session"][partition.test_indices]) == {"2"}

    protocol = partition.protocol(
        name="fixture-cross-session",
        transfer_regime="few_shot",
        preprocessing="8-30 Hz fit/transform parameters learned from train only",
    )
    manifest = partition.manifest(protocol=protocol)

    assert manifest["evidence_tier"] == "real_dataset"
    assert manifest["train_group_values"] == ["0", "1"]
    assert manifest["test_group_values"] == ["2"]
    assert manifest["partition_fingerprint"] == partition.fingerprint
    assert manifest["protocol"]["fingerprint"] == protocol.fingerprint
    json.dumps(manifest, sort_keys=True)

    repeat = hold_out_groups(data, split_unit="session", held_out_values=[2])
    assert repeat.fingerprint == partition.fingerprint


def test_unknown_or_total_holdout_fails_closed():
    data = GroupedEvaluationData.from_moabb_result(
        _moabb_like_result(),
        dataset_id="fixture",
    )
    with pytest.raises(ValueError, match="unknown held-out"):
        hold_out_groups(data, split_unit="session", held_out_values=["9"])
    with pytest.raises(ValueError, match="entire dataset"):
        hold_out_groups(data, split_unit="subject", held_out_values=["1", "2"])
    with pytest.raises(ValueError, match="deployment-unit"):
        hold_out_groups(data, split_unit="sample", held_out_values=[0])


def test_moabb_metadata_without_deployment_identity_is_rejected():
    X, y, _ = _moabb_like_result()
    metadata = [{"condition": "left"} for _ in range(len(X))]
    with pytest.raises(ValueError, match="subject/session/run"):
        GroupedEvaluationData.from_moabb_result(
            (X, y, metadata),
            dataset_id="bad",
        )


class _FakeDataset:
    code = "FakeMOABB"


class _FakeParadigm:
    def __init__(self):
        self.calls = []

    def get_data(self, *, dataset, subjects, **kwargs):
        self.calls.append((dataset, subjects, kwargs))
        return _moabb_like_result()


def test_collect_moabb_is_a_thin_explicit_adapter():
    dataset = _FakeDataset()
    paradigm = _FakeParadigm()
    data = collect_moabb(
        dataset,
        paradigm,
        subjects=[1, 2],
        return_epochs=False,
    )
    assert data.dataset_id == "FakeMOABB"
    assert paradigm.calls == [(dataset, [1, 2], {"return_epochs": False})]


def test_moabb_result_shape_and_metadata_contracts_fail_closed():
    X, y, metadata = _moabb_like_result()
    with pytest.raises(TypeError, match="3-tuple"):
        GroupedEvaluationData.from_moabb_result((X, y), dataset_id="bad")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="metadata row count"):
        GroupedEvaluationData.from_moabb_result(
            (X, y, metadata[:-1]),
            dataset_id="bad",
        )
