from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.longitudinal_external import (
    ExternalTaskDecoderMethodSpec,
    pair_task_performance,
    run_external_task_decoder_case,
)
from neuros.foundation_models.longitudinal_methods import (
    TaskDecoderMethodSpec,
    run_task_decoder_case,
)
from neuros.foundation_models.real_world import GroupedEvaluationData

pytest.importorskip("torch")
pytest.importorskip("braindecode")


def _fixture() -> GroupedEvaluationData:
    rng = np.random.default_rng(4242)
    X: list[np.ndarray] = []
    y: list[str] = []
    metadata: list[dict[str, str]] = []
    for session_index, session in enumerate(("0", "1", "2")):
        for label_index, label in enumerate(("left_hand", "right_hand")):
            for trial in range(8):
                signal = rng.normal(scale=0.25, size=(4, 128)).astype(np.float32)
                direction = -1.0 if label_index == 0 else 1.0
                signal[label_index, 20:100] += direction * (1.0 + 0.04 * session_index)
                signal[3] += 0.03 * session_index
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
        dataset_id="synthetic-longitudinal-eeg",
    )


def _authority(data: GroupedEvaluationData) -> LongitudinalCaseAuthority:
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="2",
        order=("0", "1", "2"),
    )
    split = make_nested_calibration_split(
        partition,
        evaluation_fraction=0.5,
        seed=91,
    )
    return LongitudinalCaseAuthority.from_split(
        split,
        case_id="subject-1/session-2",
        history_policy="prior",
        observed_group_order=("0", "1", "2"),
        case_metadata={"subject": 1},
    )


def test_external_spec_refuses_evidence_authority_overrides():
    with pytest.raises(ValueError, match="controlled by the evidence runner"):
        ExternalTaskDecoderMethodSpec(
            "braindecode-eegnet",
            model_seed=7,
            model_kwargs={"n_times": 999},
        )

    with pytest.raises(ValueError, match="finite JSON-serializable"):
        ExternalTaskDecoderMethodSpec(
            "braindecode-eegnet",
            model_seed=7,
            model_kwargs={"learning_rate": float("nan")},
        )

    with pytest.raises(ValueError, match="finite and positive"):
        ExternalTaskDecoderMethodSpec(
            "braindecode-eegnet",
            model_seed=7,
            sample_rate_hz=float("nan"),
        )


def test_braindecode_eegnet_is_paired_with_native_eegnet_under_one_authority():
    data = _fixture()
    authority = _authority(data)
    common = {
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "n_epochs": 1,
        "batch_size": 8,
        "device": "cpu",
    }
    native = run_task_decoder_case(
        data,
        authority,
        spec=TaskDecoderMethodSpec(
            "eegnet",
            model_seed=101,
            model_kwargs=common,
        ),
        budgets_per_class=(0, 1),
    )
    external = run_external_task_decoder_case(
        data,
        authority,
        spec=ExternalTaskDecoderMethodSpec(
            "braindecode-eegnet",
            model_seed=101,
            sample_rate_hz=128.0,
            model_kwargs=common,
        ),
        budgets_per_class=(0, 1),
    )

    assert native.authority_fingerprint == external.authority_fingerprint
    assert external.upstream_version is not None
    assert external.upstream_version.startswith("1.7.")
    assert external.parameter_count > 0
    assert external.resolved_model_config["sample_rate_hz"] == 128.0
    assert external.method_spec.sample_rate_hz == 128.0
    assert len(external.method_run_fingerprint) == 16
    assert len(external.analysis_manifest_fingerprint) == 16

    for row in external.rows:
        assert len(str(row["model_state_sha256"])) == 64
        assert row["authority_fingerprint"] == authority.authority_fingerprint
        assert row["processed_data_sha256"] == authority.processed_data_sha256
        assert row["partition_fingerprint"] == authority.partition_fingerprint
        assert row["calibration_split_fingerprint"] == authority.calibration_split_fingerprint
        assert row["sample_rate_hz"] == 128.0
        assert row["representation_evidence_available"] is False
        assert row["mechanistic_evidence_available"] is False
        assert 0.0 <= float(row["balanced_accuracy"]) <= 1.0
        assert 0.0 <= float(row["accuracy"]) <= 1.0

    paired = pair_task_performance(native, external)
    assert paired.authority_fingerprint == authority.authority_fingerprint
    assert len(paired.rows) == 2
    assert len(paired.pair_fingerprint) == 16

    for row in paired.rows:
        assert row["native_method_id"] == "eegnet"
        assert row["external_method_id"] == "braindecode-eegnet"
        assert row["sample_rate_hz"] == 128.0
        assert row["external_representation_evidence_available"] is False
        assert row["external_mechanistic_evidence_available"] is False
        expected = float(row["external_balanced_accuracy"]) - float(
            row["native_balanced_accuracy"]
        )
        assert np.isclose(row["delta_external_minus_native_balanced_accuracy"], expected)
        assert row["native_model_state_sha256"] != row["external_model_state_sha256"]


def test_pairing_refuses_results_from_different_authorities():
    data = _fixture()
    authority = _authority(data)
    common = {"n_epochs": 1, "batch_size": 8, "device": "cpu"}
    native = run_task_decoder_case(
        data,
        authority,
        spec=TaskDecoderMethodSpec("eegnet", model_seed=5, model_kwargs=common),
        budgets_per_class=(0,),
    )
    external = run_external_task_decoder_case(
        data,
        authority,
        spec=ExternalTaskDecoderMethodSpec(
            "braindecode-eegnet",
            model_seed=5,
            sample_rate_hz=128.0,
            model_kwargs=common,
        ),
        budgets_per_class=(0,),
    )

    corrupted = replace(external, authority_fingerprint="not-the-same-authority")
    with pytest.raises(ValueError, match="one authority fingerprint"):
        pair_task_performance(native, corrupted)
