from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.qualification import QualificationProtocolSpec
from neuros.foundation_models.qualification_baselines import (
    MNECSPLDAFactory,
    RiemannianTangentLogRegFactory,
    UpstreamBraindecodeFactory,
)
from neuros.foundation_models.qualification_runner import (
    DEFAULT_CLASSIFICATION_SCORECARD,
    QualificationExecutionContext,
    run_external_qualification_case,
)
from neuros.foundation_models.real_world import GroupedEvaluationData

LINEAGE_SHA = "e" * 64


def _eeg_data() -> GroupedEvaluationData:
    rng = np.random.default_rng(19)
    n_sessions = 3
    trials_per_session = 16
    n_samples = n_sessions * trials_per_session
    n_channels = 4
    n_times = 128
    time = np.linspace(0.0, 1.0, n_times, endpoint=False)
    X = rng.normal(scale=0.2, size=(n_samples, n_channels, n_times)).astype(np.float32)
    y = np.asarray(["left", "right"] * (n_samples // 2), dtype=str)
    for index, label in enumerate(y):
        if label == "left":
            X[index, 0] += np.sin(2.0 * np.pi * 10.0 * time).astype(np.float32)
            X[index, 1] += 0.35 * np.sin(2.0 * np.pi * 10.0 * time).astype(np.float32)
        else:
            X[index, 2] += np.sin(2.0 * np.pi * 12.0 * time).astype(np.float32)
            X[index, 3] += 0.35 * np.sin(2.0 * np.pi * 12.0 * time).astype(np.float32)
    session = np.repeat(np.asarray(["s1", "s2", "s3"], dtype=str), trials_per_session)
    subject = np.asarray(["p1"] * n_samples, dtype=str)
    trial = np.asarray([f"t{index:03d}" for index in range(n_samples)], dtype=str)
    return GroupedEvaluationData(
        dataset_id="fixture-upstream-mi",
        X=X,
        y=y,
        groups={"subject": subject, "session": session, "trial": trial},
    )


def _authority(data: GroupedEvaluationData) -> LongitudinalCaseAuthority:
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="s3",
        order=("s1", "s2", "s3"),
    )
    split = make_nested_calibration_split(partition, evaluation_fraction=0.5, seed=5)
    return LongitudinalCaseAuthority.from_split(
        split,
        case_id="p1:s3",
        history_policy="prior",
        observed_group_order=("s1", "s2", "s3"),
    )


def _protocol(
    data: GroupedEvaluationData,
    authority: LongitudinalCaseAuthority,
    *,
    budgets: tuple[int, ...],
) -> QualificationProtocolSpec:
    return QualificationProtocolSpec(
        protocol_id="fixture-upstream-nsq-v1",
        dataset_id=data.dataset_id,
        dataset_lineage_sha256=LINEAGE_SHA,
        task_id="left-vs-right-mi",
        independent_unit="participant",
        grouping_hierarchy=("participant", "session", "trial"),
        calibration_budgets_per_class=budgets,
        metric_scorecard_sha256=DEFAULT_CLASSIFICATION_SCORECARD.sha256,
        protocol_status="frozen",
    )


def _context() -> QualificationExecutionContext:
    return QualificationExecutionContext(
        observed_dataset_lineage_sha256=LINEAGE_SHA,
    )


def test_mne_csp_lda_participates_as_external_probability_method():
    pytest.importorskip("mne")
    pytest.importorskip("sklearn")
    data = _eeg_data()
    authority = _authority(data)
    factory = MNECSPLDAFactory(n_components=2)
    result = run_external_qualification_case(
        data,
        authority,
        _protocol(data, authority, budgets=(0, 1)),
        factory,
        execution_context=_context(),
    )

    assert factory.method_spec.method_id == "mne-csp-lda"
    assert "mne.decoding.CSP" in factory.method_spec.implementation
    assert all(row.status == "success" for row in result.rows)
    assert all(row.probability_available for row in result.rows)
    assert all(row.score is not None for row in result.rows)
    assert all(row.score.availability["brier_score"] == "available" for row in result.rows)
    # Scientific comparison is allowed without pretending joblib/pickle is a
    # qualified promoted-state identity.
    assert all(row.learned_state_addressable is False for row in result.rows)
    assert all(row.external_learned_state_sha256 is None for row in result.rows)
    assert all(row.qualification_model_state_sha256 is not None for row in result.rows)


def test_pyriemann_rg_lr_participates_without_transductive_test_update():
    pytest.importorskip("pyriemann")
    pytest.importorskip("sklearn")
    data = _eeg_data()
    authority = _authority(data)
    factory = RiemannianTangentLogRegFactory()
    spec = factory.method_spec

    assert spec.method_id == "pyriemann-rg-lr"
    assert spec.metadata["covariance_estimator"] == "scm"
    assert spec.metadata["tangent_metric"] == "riemann"
    assert spec.metadata["tangent_space_update"] is False
    assert spec.metadata["transductive_evaluation_batch_update"] is False

    result = run_external_qualification_case(
        data,
        authority,
        _protocol(data, authority, budgets=(0, 1)),
        factory,
        execution_context=_context(),
    )
    assert all(row.status == "success" for row in result.rows)
    assert all(row.probability_available for row in result.rows)
    assert all(row.score is not None for row in result.rows)
    assert all(row.learned_state_addressable is False for row in result.rows)
    assert all(row.external_learned_state_sha256 is None for row in result.rows)
    assert all(row.qualification_model_state_sha256 is not None for row in result.rows)



def test_braindecode_factory_is_direct_upstream_not_neuros_model_wrapper():
    pytest.importorskip("braindecode")
    pytest.importorskip("torch")
    factory = UpstreamBraindecodeFactory(
        model_name="EEGNet",
        sample_rate_hz=128.0,
        optimizer_name="Adam",
        learning_rate=0.000625,
        n_epochs=2,
        batch_size=8,
        random_state=3,
        validation_fraction=0.2,
        validation_seed=7,
        early_stopping_patience=1,
        restore_best=True,
    )
    spec = factory.method_spec
    assert spec.implementation.startswith("braindecode.models.EEGNet+")
    assert spec.metadata["neuros_model_wrapper_used"] is False
    assert spec.metadata["optimizer"] == "torch.optim.Adam"
    assert spec.metadata["train_split"] == {
        "implementation": "skorch.dataset.ValidSplit",
        "fraction": 0.2,
        "stratified": False,
        "seed": 7,
    }
    assert spec.metadata["state_selection"]["restore_best"] is True
    assert spec.metadata["final_assessment_used_for_state_selection"] is False
    assert spec.probability_semantics == "uncalibrated_softmax"


def test_upstream_braindecode_executes_through_same_nsq_referee():
    pytest.importorskip("braindecode")
    pytest.importorskip("torch")
    data = _eeg_data()
    authority = _authority(data)
    factory = UpstreamBraindecodeFactory(
        model_name="EEGNet",
        sample_rate_hz=128.0,
        optimizer_name="Adam",
        learning_rate=0.000625,
        n_epochs=2,
        batch_size=8,
        random_state=3,
        validation_fraction=0.2,
        validation_seed=7,
        early_stopping_patience=1,
        restore_best=True,
    )
    result = run_external_qualification_case(
        data,
        authority,
        _protocol(data, authority, budgets=(0,)),
        factory,
        execution_context=_context(),
    )

    assert len(result.rows) == 1
    row = result.rows[0]
    assert row.status == "success"
    assert row.probability_available is True
    assert row.learned_state_addressable is True
    assert row.external_state_identity_kind == "tensor_sha256"
    assert row.external_learned_state_sha256 is not None
    assert row.qualification_model_state_sha256 is not None
    assert row.external_learned_state_sha256 != row.qualification_model_state_sha256
    assert row.qualification_model_state is not None
    metadata = row.qualification_model_state.learned_state.metadata
    assert metadata["validation_policy"] == "skorch.ValidSplit"
    assert metadata["final_assessment_used_for_state_selection"] is False
    assert len(metadata["validation_relative_indices_sha256"]) == 64
    assert row.score is not None
    assert row.score.availability["brier_score"] == "available"


def test_upstream_braindecode_binds_validation_membership_and_selected_tensor_state():
    pytest.importorskip("braindecode")
    pytest.importorskip("torch")
    data = _eeg_data()
    factory = UpstreamBraindecodeFactory(
        model_name="EEGNet",
        sample_rate_hz=128.0,
        optimizer_name="Adam",
        learning_rate=0.000625,
        n_epochs=2,
        batch_size=8,
        random_state=11,
        validation_fraction=0.2,
        validation_seed=13,
        early_stopping_patience=1,
        restore_best=True,
    )
    decoder = factory.create()
    decoder.fit(data.X[:32], data.y[:32])
    learned = decoder.learned_state()
    metadata = learned.metadata

    assert learned.state_identity_kind == "tensor_sha256"
    assert learned.state_sha256 is not None
    assert metadata["model_seed"] == 11
    assert metadata["validation_seed"] == 13
    assert metadata["validation_policy"] == "skorch.ValidSplit"
    assert metadata["validation_stratified"] is False
    assert metadata["restore_best"] is True
    assert metadata["final_assessment_used_for_state_selection"] is False
    assert metadata["validation_samples"] == len(metadata["validation_relative_indices"])
    assert len(set(metadata["validation_relative_indices"])) == metadata["validation_samples"]
    assert metadata["best_observed_epoch"] <= metadata["stopped_epoch"]
    assert len(metadata["validation_relative_indices_sha256"]) == 64
    assert len(metadata["train_relative_indices_sha256"]) == 64



def test_missing_upstream_braindecode_architecture_is_preserved_as_unavailable():
    pytest.importorskip("braindecode")
    pytest.importorskip("torch")
    data = _eeg_data()
    authority = _authority(data)
    factory = UpstreamBraindecodeFactory(
        model_name="DefinitelyMissingNSQArchitecture",
        sample_rate_hz=128.0,
        n_epochs=1,
        batch_size=8,
        random_state=3,
    )
    result = run_external_qualification_case(
        data,
        authority,
        _protocol(data, authority, budgets=(0, 1)),
        factory,
        execution_context=_context(),
    )

    assert len(result.rows) == 2
    assert all(row.status == "unavailable" for row in result.rows)
    assert all(row.failure_kind == "ImportError" for row in result.rows)
    assert all("does not expose model" in (row.failure_reason or "") for row in result.rows)
    assert all(row.score is None for row in result.rows)
