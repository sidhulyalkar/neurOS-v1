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
    prepare_frozen_encoder_case,
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


def _prepared(data, authority):
    return prepare_frozen_encoder_case(
        data,
        authority,
        encoder_id="eegnet",
        encoder_seed=101,
        encoder_kwargs=_encoder_kwargs(),
    )


def test_frozen_logistic_reuses_one_source_trained_encoder_across_budgets():
    data = _fixture()
    authority = _authority(data)
    prepared = _prepared(data, authority)
    spec = FrozenTransferMethodSpec(
        method_id="frozen-eegnet",
        strategy="frozen-logistic",
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
        prepared=prepared,
    )

    assert result.authority_fingerprint == authority.authority_fingerprint
    assert result.encoder_state_manifest["encoder_parameter_count"] > 0
    assert result.encoder_state_manifest["encoder_fit_s"] >= 0.0
    assert result.encoder_state_manifest["encoder_state_fingerprint"] == prepared.fingerprint
    assert result.encoder_state_manifest["representation_sha256"] == prepared.representation_sha256
    assert len(result.rows) == 3
    assert {row["status"] for row in result.rows} == {"ok"}
    assert {row["method_id"] for row in result.rows} == {"frozen-eegnet"}
    assert {row["transfer_strategy"] for row in result.rows} == {"frozen-logistic"}
    assert [row["calibration_per_class"] for row in result.rows] == [0, 1, 2]
    assert all(row["sourceweigher"] is None for row in result.rows)
    assert {row["encoder_state_fingerprint"] for row in result.rows} == {prepared.fingerprint}
    assert {row["representation_sha256"] for row in result.rows} == {prepared.representation_sha256}
    for row in result.rows:
        assert row["authority_fingerprint"] == authority.authority_fingerprint
        assert 0.0 <= row["balanced_accuracy"] <= 1.0
        assert 0.0 <= row["brier_score"] <= 2.0
    json.dumps(result.to_dict(), sort_keys=True)


def test_sourceweigher_and_unweighted_transfer_share_exact_prepared_encoder(monkeypatch):
    import neuros.foundation_models.longitudinal_transfer as transfer

    data = _fixture()
    authority = _authority(data)
    prepared = _prepared(data, authority)
    observed_target_sizes = []
    real = transfer._sourceweigher_result

    def wrapped(*, source_embeddings, target_embeddings):
        observed_target_sizes.append(len(target_embeddings))
        return real(
            source_embeddings=source_embeddings,
            target_embeddings=target_embeddings,
        )

    monkeypatch.setattr(transfer, "_sourceweigher_result", wrapped)
    frozen = run_frozen_transfer_case(
        data,
        authority,
        spec=FrozenTransferMethodSpec(
            method_id="frozen-eegnet",
            strategy="frozen-logistic",
            encoder_id="eegnet",
            encoder_seed=101,
            encoder_kwargs=_encoder_kwargs(),
        ),
        budgets_per_class=(0, 1, 2),
        prepared=prepared,
    )
    weighted = run_frozen_transfer_case(
        data,
        authority,
        spec=FrozenTransferMethodSpec(
            method_id="sourceweigher-eegnet",
            strategy="sourceweigher-mean",
            encoder_id="eegnet",
            encoder_seed=101,
            encoder_kwargs=_encoder_kwargs(),
        ),
        budgets_per_class=(0, 1, 2),
        prepared=prepared,
    )

    assert observed_target_sizes == [2, 4]
    assert frozen.encoder_state_manifest["encoder_state_fingerprint"] == weighted.encoder_state_manifest["encoder_state_fingerprint"]
    assert frozen.encoder_state_manifest["representation_sha256"] == weighted.encoder_state_manifest["representation_sha256"]
    assert frozen.encoder_state_manifest["representation_sha256"] == prepared.representation_sha256

    zero, one, two = weighted.rows
    assert zero["status"] == "unavailable_no_target_observations"
    assert zero["method_id"] == "sourceweigher-eegnet"
    assert zero["transfer_strategy"] == "sourceweigher-mean"
    assert zero["partition_fingerprint"] == authority.partition_fingerprint
    assert zero["calibration_split_fingerprint"] == authority.calibration_split_fingerprint
    assert zero["encoder_state_fingerprint"] == prepared.fingerprint
    assert "evaluation examples are forbidden" in zero["failure_reason"]

    for row in (one, two):
        assert row["status"] == "ok"
        assert row["method_id"] == "sourceweigher-eegnet"
        assert row["representation_sha256"] == prepared.representation_sha256
        payload = row["sourceweigher"]
        assert payload is not None
        assert payload["source_ids"] == ["0", "1"]
        assert np.isclose(sum(payload["weights"]), 1.0)
        diagnostics = payload["diagnostics"]
        assert diagnostics["effective_sample_size"] >= 1.0
        assert diagnostics["entropy"] >= 0.0
        assert diagnostics["max_weight"] <= 1.0 + 1e-9
        assert diagnostics["iterations"] >= 1


def test_distinct_encoders_can_have_distinct_ladder_method_ids():
    first = FrozenTransferMethodSpec(
        method_id="frozen-eegnet",
        strategy="frozen-logistic",
        encoder_id="eegnet",
        encoder_seed=101,
    )
    second = FrozenTransferMethodSpec(
        method_id="frozen-eeg-conformer",
        strategy="frozen-logistic",
        encoder_id="eeg-conformer",
        encoder_seed=101,
    )
    assert first.method_id != second.method_id
    assert first.strategy == second.strategy
    assert first.fingerprint != second.fingerprint


def test_sourceweigher_zero_only_run_is_explicitly_unavailable():
    data = _fixture()
    authority = _authority(data)
    prepared = _prepared(data, authority)
    spec = FrozenTransferMethodSpec(
        method_id="sourceweigher-eegnet",
        strategy="sourceweigher-mean",
        encoder_id="eegnet",
        encoder_seed=101,
        encoder_kwargs=_encoder_kwargs(),
    )
    result = run_frozen_transfer_case(
        data,
        authority,
        spec=spec,
        budgets_per_class=(0,),
        prepared=prepared,
    )
    row = result.rows[0]
    assert row["status"] == "unavailable_no_target_observations"
    assert row["authority_fingerprint"] == authority.authority_fingerprint
    assert row["partition_fingerprint"] == authority.partition_fingerprint
    assert row["calibration_split_fingerprint"] == authority.calibration_split_fingerprint


def test_prepared_encoder_validates_authority_before_transfer():
    data = _fixture()
    authority = _authority(data)
    prepared = _prepared(data, authority)
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
        method_id="frozen-eegnet",
        strategy="frozen-logistic",
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
            prepared=prepared,
        )