from __future__ import annotations

import json

import numpy as np
import pytest

from neuros.foundation_models import (
    GroupedEvaluationData,
    LongitudinalCaseAuthority,
    TaskDecoderMethodSpec,
    chronological_partition,
    make_nested_calibration_split,
    run_task_decoder_case,
)

pytest.importorskip("torch")


def _fixture() -> GroupedEvaluationData:
    rng = np.random.default_rng(321)
    X = []
    y = []
    metadata = []
    for session_index, session in enumerate(("0", "1", "2")):
        for label_index, label in enumerate(("left", "right")):
            for trial in range(8):
                x = rng.normal(scale=0.5, size=(4, 128)).astype(np.float32)
                # Deliberately simple discriminative temporal pattern.
                sign = -1.0 if label_index == 0 else 1.0
                x[label_index, 24:88] += sign * (1.2 + 0.05 * session_index)
                X.append(x)
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
        dataset_id="fixture",
    )


def _authority(data: GroupedEvaluationData) -> LongitudinalCaseAuthority:
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="2",
        order=("0", "1", "2"),
    )
    split = make_nested_calibration_split(partition, evaluation_fraction=0.5, seed=2026)
    return LongitudinalCaseAuthority.from_split(
        split,
        case_id="subject-1/session-2",
        history_policy="prior",
        observed_group_order=("0", "1", "2"),
        case_metadata={"subject": 1, "original_protocol": "GR"},
    )


def test_eegnet_runs_two_budgets_against_same_authority():
    data = _fixture()
    authority = _authority(data)
    spec = TaskDecoderMethodSpec(
        method_id="eegnet",
        model_seed=101,
        model_kwargs={
            "n_epochs": 1,
            "batch_size": 16,
            "device": "cpu",
            "temporal_filters": 4,
            "depth_multiplier": 1,
            "separable_filters": 8,
            "temporal_kernel": 15,
            "separable_kernel": 7,
        },
    )
    result = run_task_decoder_case(
        data,
        authority,
        spec=spec,
        budgets_per_class=(0, 1),
    )

    assert result.authority_fingerprint == authority.authority_fingerprint
    assert result.parameter_count > 0
    assert result.resolved_model_config["n_epochs"] == 1
    assert result.resolved_model_config["device_spec"] == "cpu"
    assert len(result.rows) == 2
    assert {row["calibration_per_class"] for row in result.rows} == {0, 1}
    assert {row["authority_fingerprint"] for row in result.rows} == {
        authority.authority_fingerprint
    }
    assert {row["calibration_split_fingerprint"] for row in result.rows} == {
        authority.calibration_split_fingerprint
    }
    for row in result.rows:
        assert 0.0 <= row["accuracy"] <= 1.0
        assert 0.0 <= row["balanced_accuracy"] <= 1.0
        assert row["roc_auc"] is None or 0.0 <= row["roc_auc"] <= 1.0
        assert 0.0 <= row["brier_score"] <= 2.0
        assert 0.0 <= row["expected_calibration_error"] <= 1.0
        assert row["evaluation_representation"]["finite"] is True
        assert row["source_representation"]["finite"] is True

    json.dumps(result.to_dict(), sort_keys=True)


def test_compact_conformer_runs_under_same_contract():
    data = _fixture()
    authority = _authority(data)
    spec = TaskDecoderMethodSpec(
        method_id="eeg-conformer",
        model_seed=503,
        model_kwargs={
            "n_epochs": 1,
            "batch_size": 16,
            "device": "cpu",
            "embedding_dim": 8,
            "temporal_kernel": 9,
            "pool_length": 8,
            "pool_stride": 4,
            "n_heads": 2,
            "n_layers": 1,
            "feedforward_multiplier": 2,
            "dropout": 0.1,
        },
    )
    result = run_task_decoder_case(
        data,
        authority,
        spec=spec,
        budgets_per_class=(0,),
    )

    assert result.parameter_count > 0
    assert result.resolved_model_config["embedding_dim"] == 8
    assert result.resolved_model_config["n_layers"] == 1
    assert result.rows[0]["evaluation_representation"]["n_features"] == 8


def test_method_spec_forbids_overriding_evidence_controlled_fields():
    for key in ("n_channels", "n_classes", "random_state"):
        with pytest.raises(ValueError, match="controlled"):
            TaskDecoderMethodSpec(
                method_id="eegnet",
                model_seed=1,
                model_kwargs={key: 123},
            )


def test_method_runner_rejects_budget_beyond_authority():
    data = _fixture()
    authority = _authority(data)
    split = authority.restore(data)
    spec = TaskDecoderMethodSpec(
        method_id="eegnet",
        model_seed=1,
        model_kwargs={"n_epochs": 1, "device": "cpu"},
    )
    with pytest.raises(ValueError, match="exceeds authority maximum"):
        run_task_decoder_case(
            data,
            authority,
            spec=spec,
            budgets_per_class=(split.max_budget_per_class + 1,),
        )


def test_method_runner_validates_processed_data_before_training():
    data = _fixture()
    authority = _authority(data)
    changed_x = data.X.copy()
    changed_x[0, 0, 0] += 0.5
    changed = GroupedEvaluationData(
        dataset_id=data.dataset_id,
        X=changed_x,
        y=data.y,
        groups=data.groups,
        metadata=data.metadata,
    )
    spec = TaskDecoderMethodSpec(
        method_id="eegnet",
        model_seed=1,
        model_kwargs={"n_epochs": 1, "device": "cpu"},
    )
    with pytest.raises(ValueError, match="SHA-256"):
        run_task_decoder_case(
            changed,
            authority,
            spec=spec,
            budgets_per_class=(0,),
        )
