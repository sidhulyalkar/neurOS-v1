from __future__ import annotations

import json

import numpy as np
import pytest

from neuros.foundation_models import (
    GroupedEvaluationData,
    LongitudinalCaseAuthority,
    chronological_partition,
    make_nested_calibration_split,
)
from neuros.foundation_models.longitudinal_transfer import (
    FrozenTransferMethodSpec,
    run_frozen_transfer_case,
)

pytest.importorskip("torch")
pytest.importorskip("neuros_sourceweigher")


def _fixture() -> GroupedEvaluationData:
    rng = np.random.default_rng(909)
    X = []
    y = []
    metadata = []
    for session_index, session in enumerate(("0", "1", "2")):
        for label_index, label in enumerate(("left", "right")):
            for trial in range(8):
                x = rng.normal(scale=0.45, size=(4, 128)).astype(np.float32)
                sign = -1.0 if label_index == 0 else 1.0
                x[label_index, 20:92] += sign * (1.0 + 0.07 * session_index)
                # Session-specific nuisance shift gives SourceWeigher something
                # nontrivial to summarize in representation space.
                x[3] += 0.03 * session_index
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
    )


def _encoder_kwargs():
    return {
        "n_epochs": 1,
        "batch_size": 16,
        "device": "cpu",
        "temporal_filters": 4,
        "depth_multiplier": 1,
        "separable_filters": 8,
        "temporal_kernel": 15,
        "separable_kernel": 7,
    }


def test_frozen_logistic_reuses_one_source_trained_encoder_across_budgets():
    data = _fixture()
    authority = _authority(data)
    spec = FrozenTransferMethodSpec(
        method_id="frozen-logistic",
        encoder_id="eegnet",
        encoder_seed=101,
        encoder_kwargs=_encoder_kwargs(),
        readout_c=1.0,
    )
    result = run_frozen_transfer_case(
        data,
        authority,
        spec=spec,
        budgets_per_class=(0, 1, 2),
    )

    assert result.authority_fingerprint == authority.authority_fingerprint
    assert result.encoder_parameter_count > 0
    assert result.encoder_fit_s >= 0.0
    assert len(result.rows) == 3
    assert {row["status"] for row in result.rows} == {"ok"}
    assert [row["calibration_per_class"] for row in result.rows] == [0, 1, 2]
    assert all(row["sourceweigher"] is None for row in result.rows)
    for row in result.rows:
        assert row["authority_fingerprint"] == authority.authority_fingerprint
        assert 0.0 <= row["balanced_accuracy"] <= 1.0
        assert 0.0 <= row["brier_score"] <= 2.0
    json.dumps(result.to_dict(), sort_keys=True)


def test_sourceweigher_uses_only_declared_target_calibration_embeddings(monkeypatch):
    import neuros.foundation_models.longitudinal_transfer as transfer

    data = _fixture()
    authority = _authority(data)
    observed_target_sizes = []
    real = transfer._sourceweigher_result

    def wrapped(*, source_embeddings, target_embeddings):
        observed_target_sizes.append(len(target_embeddings))
        return real(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )

    monkeypatch.setattr(transfer, "_sourceweigher_result", wrapped)
    spec = FrozenTransferMethodSpec(
        method_id="sourceweigher-mean",
        encoder_id="eegnet",
        encoder_seed=101,
        encoder_kwargs=_encoder_kwargs(),
    )
    result = run_frozen_transfer_case(
        data,
        authority,
        spec=spec,
        budgets_per_class=(0, 1, 2),
    )

    assert observed_target_sizes == [2, 4]  # 1/class then 2/class; never evaluation set
    zero, one, two = result.rows
    assert zero["status"] == "unavailable_no_target_observations"
    assert "evaluation examples are forbidden" in zero["failure_reason"]

    for row in (one, two):
        assert row["status"] == "ok"
        payload = row["sourceweigher"]
        assert payload is not None
        assert payload["source_ids"] == ["0", "1"]
        assert np.isclose(sum(payload["weights"]), 1.0)
        diagnostics = payload["diagnostics"]
        assert diagnostics["effective_sample_size"] >= 1.0
        assert diagnostics["entropy"] >= 0.0
        assert diagnostics["max_weight"] <= 1.0 + 1e-9
        assert diagnostics["iterations"] >= 1


def test_sourceweigher_rejects_zero_only_run_without_using_eval_data():
    data = _fixture()
    authority = _authority(data)
    spec = FrozenTransferMethodSpec(
        method_id="sourceweigher-mean",
        encoder_id="eegnet",
        encoder_seed=101,
        encoder_kwargs=_encoder_kwargs(),
    )
    result = run_frozen_transfer_case(
        data,
        authority,
        spec=spec,
        budgets_per_class=(0,),
    )
    assert result.rows[0]["status"] == "unavailable_no_target_observations"


def test_frozen_transfer_validates_authority_before_encoder_training():
    data = _fixture()
    authority = _authority(data)
    changed_x = data.X.copy()
    changed_x[0, 0, 0] += 0.2
    changed = GroupedEvaluationData(
        dataset_id=data.dataset_id,
        X=changed_x,
        y=data.y,
        groups=data.groups,
        metadata=data.metadata,
    )
    spec = FrozenTransferMethodSpec(
        method_id="frozen-logistic",
        encoder_id="eegnet",
        encoder_seed=101,
        encoder_kwargs=_encoder_kwargs(),
    )
    with pytest.raises(ValueError, match="SHA-256"):
        run_frozen_transfer_case(
            changed,
            authority,
            spec=spec,
            budgets_per_class=(0,),
        )
