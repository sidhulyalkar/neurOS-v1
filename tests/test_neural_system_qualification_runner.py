from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.qualification import (
    ExternalDecoderMethodSpec,
    ExternalLearnedState,
)
from neuros.foundation_models.qualification_runner import (
    DEFAULT_CLASSIFICATION_SCORECARD,
    ClassificationScorecardV1,
    QualificationExecutionContext,
    run_external_qualification_case,
)
from neuros.foundation_models.real_world import GroupedEvaluationData

LINEAGE_SHA = "a" * 64
PREPROCESSING_SHA = "b" * 64
CALIBRATION_SHA = "c" * 64
STATE_SHA = "d" * 64


def _data() -> GroupedEvaluationData:
    # Every sample encodes its row index, allowing tests to audit exactly which
    # observations crossed the external-model boundary.
    n_samples = 36
    X = np.arange(n_samples, dtype=np.float32)[:, None, None]
    X = np.repeat(X, 8, axis=2)
    y = np.asarray(["left", "right"] * (n_samples // 2), dtype=str)
    session = np.repeat(np.asarray(["s1", "s2", "s3"], dtype=str), 12)
    subject = np.asarray(["p1"] * n_samples, dtype=str)
    trial = np.asarray([f"t{index:02d}" for index in range(n_samples)], dtype=str)
    return GroupedEvaluationData(
        dataset_id="fixture-longitudinal-mi",
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
    split = make_nested_calibration_split(
        partition,
        evaluation_fraction=0.5,
        seed=11,
    )
    return LongitudinalCaseAuthority.from_split(
        split,
        case_id="p1:s3",
        history_policy="prior",
        observed_group_order=("s1", "s2", "s3"),
        case_metadata={"participant": "p1"},
    )


def _protocol(data: GroupedEvaluationData, authority: LongitudinalCaseAuthority):
    from neuros.foundation_models.qualification import QualificationProtocolSpec

    split = authority.restore(data)
    return QualificationProtocolSpec(
        protocol_id="fixture-nsq-v1",
        dataset_id=data.dataset_id,
        dataset_lineage_sha256=LINEAGE_SHA,
        task_id="left-vs-right-motor-imagery",
        independent_unit="participant",
        grouping_hierarchy=("participant", "session", "trial"),
        calibration_budgets_per_class=tuple(range(split.max_budget_per_class + 1)),
        metric_scorecard_sha256=DEFAULT_CLASSIFICATION_SCORECARD.sha256,
        protocol_status="frozen",
        metadata={"fixture": True},
    )


def _context(*, unlabeled_target_examples: int = 0) -> QualificationExecutionContext:
    return QualificationExecutionContext(
        observed_dataset_lineage_sha256=LINEAGE_SHA,
        preprocessing_authority_sha256s=(PREPROCESSING_SHA,),
        calibration_authority_sha256s=(CALIBRATION_SHA,),
        unlabeled_target_examples=unlabeled_target_examples,
        target_example_duration_s=1.25 if unlabeled_target_examples else None,
    )


class LabelOnlyDecoder:
    def __init__(self) -> None:
        self.fit_indices: tuple[int, ...] = ()
        self.labels: tuple[str, ...] = ()

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.fit_indices = tuple(int(value) for value in X[:, 0, 0])
        self.labels = tuple(sorted(np.unique(y.astype(str)).tolist()))

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.labels
        return np.asarray([self.labels[0]] * len(X), dtype=str)

    def learned_state(self) -> ExternalLearnedState:
        return ExternalLearnedState()


class LabelOnlyFactory:
    def __init__(self, *, adaptation: str = "none") -> None:
        self.created: list[LabelOnlyDecoder] = []
        self._spec = ExternalDecoderMethodSpec(
            method_id=f"fixture-label-only-{adaptation}",
            implementation="tests.LabelOnlyDecoder",
            implementation_version="1",
            input_axes=("sample", "channel", "time"),
            probability_semantics="unavailable",
            target_adaptation_mode=adaptation,  # type: ignore[arg-type]
        )

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        return self._spec

    def create(self) -> LabelOnlyDecoder:
        decoder = LabelOnlyDecoder()
        self.created.append(decoder)
        return decoder


class AdaptiveDecoder(LabelOnlyDecoder):
    def __init__(self) -> None:
        super().__init__()
        self.adaptation_indices: tuple[int, ...] = ()

    def adapt_unlabeled(self, X: np.ndarray) -> None:
        self.adaptation_indices = tuple(int(value) for value in X[:, 0, 0])


class AdaptiveFactory:
    def __init__(self) -> None:
        self.created: list[AdaptiveDecoder] = []
        self._spec = ExternalDecoderMethodSpec(
            method_id="fixture-unlabeled-adapter",
            implementation="tests.AdaptiveDecoder",
            implementation_version="1",
            input_axes=("sample", "channel", "time"),
            probability_semantics="unavailable",
            target_adaptation_mode="unlabeled",
        )

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        return self._spec

    def create(self) -> AdaptiveDecoder:
        decoder = AdaptiveDecoder()
        self.created.append(decoder)
        return decoder


class ProbabilityDecoder(LabelOnlyDecoder):
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        assert len(self.labels) == 2
        return np.full((len(X), 2), 0.5, dtype=np.float32)

    def probability_class_labels(self):
        return self.labels

    def learned_state(self) -> ExternalLearnedState:
        return ExternalLearnedState(
            state_identity_kind="tensor_sha256",
            state_sha256=STATE_SHA,
        )


class ProbabilityFactory:
    def __init__(self, *, reverse_probability_order: bool = False) -> None:
        self.created: list[ProbabilityDecoder] = []
        self.reverse_probability_order = reverse_probability_order
        self._spec = ExternalDecoderMethodSpec(
            method_id="fixture-probability",
            implementation="tests.ProbabilityDecoder",
            implementation_version="1",
            input_axes=("sample", "channel", "time"),
            probability_semantics="uncalibrated_probability",
        )

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        return self._spec

    def create(self) -> ProbabilityDecoder:
        decoder = ProbabilityDecoder()
        if self.reverse_probability_order:
            original = decoder.probability_class_labels

            def reversed_labels():
                return tuple(reversed(original()))

            decoder.probability_class_labels = reversed_labels  # type: ignore[method-assign]
        self.created.append(decoder)
        return decoder


class FailingDecoder(LabelOnlyDecoder):
    def __init__(self, failure: Exception | None) -> None:
        super().__init__()
        self.failure = failure

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.failure is not None:
            raise self.failure
        super().fit(X, y)


class FailingFactory:
    def __init__(self, failures: list[Exception | None]) -> None:
        self.failures = list(failures)
        self.created = 0
        self._spec = ExternalDecoderMethodSpec(
            method_id="fixture-failure-preservation",
            implementation="tests.FailingDecoder",
            implementation_version="1",
            input_axes=("sample", "channel", "time"),
            probability_semantics="unavailable",
        )

    @property
    def method_spec(self) -> ExternalDecoderMethodSpec:
        return self._spec

    def create(self) -> FailingDecoder:
        failure = self.failures[self.created]
        self.created += 1
        return FailingDecoder(failure)


def test_default_scorecard_has_full_stable_identity_and_explicit_semantics():
    first = ClassificationScorecardV1()
    second = ClassificationScorecardV1()
    assert len(first.sha256) == 64
    assert first.sha256 == second.sha256
    assert first.metric_names == (
        "balanced_accuracy",
        "accuracy",
        "roc_auc",
        "brier_score",
        "expected_calibration_error",
    )
    assert first.to_dict()["metric_semantics"]["roc_auc"]["positive_class"] == (
        "second_canonical_source_label"
    )


def test_runner_creates_fresh_external_state_for_every_declared_budget():
    data = _data()
    authority = _authority(data)
    protocol = _protocol(data, authority)
    factory = LabelOnlyFactory()

    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        factory,
        execution_context=_context(),
    )

    assert len(factory.created) == len(protocol.calibration_budgets_per_class)
    assert len({id(decoder) for decoder in factory.created}) == len(factory.created)
    assert tuple(row.calibration_per_class for row in result.rows) == (
        protocol.calibration_budgets_per_class
    )
    assert all(row.status == "success" for row in result.rows)
    assert len(result.sha256) == 64


def test_final_assessment_rows_never_cross_the_fit_boundary():
    data = _data()
    authority = _authority(data)
    split = authority.restore(data)
    evaluation = set(split.evaluation_indices.tolist())
    protocol = _protocol(data, authority)
    factory = LabelOnlyFactory()

    run_external_qualification_case(
        data,
        authority,
        protocol,
        factory,
        execution_context=_context(),
    )

    for decoder in factory.created:
        assert not evaluation.intersection(decoder.fit_indices)


def test_unlabeled_target_rows_use_only_non_evaluation_authorized_target_pool():
    data = _data()
    authority = _authority(data)
    split = authority.restore(data)
    evaluation = set(split.evaluation_indices.tolist())
    source = set(split.source_train_indices.tolist())
    protocol = _protocol(data, authority)
    factory = AdaptiveFactory()

    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        factory,
        execution_context=_context(unlabeled_target_examples=1),
    )

    assert all(row.status == "success" for row in result.rows)
    assert all(row.unlabeled_target_examples == 1 for row in result.rows)
    assert all(row.unlabeled_target_seconds == pytest.approx(1.25) for row in result.rows)
    for decoder in factory.created:
        assert len(decoder.adaptation_indices) == 1
        assert not evaluation.intersection(decoder.adaptation_indices)
        assert not source.intersection(decoder.adaptation_indices)
        assert not evaluation.intersection(decoder.fit_indices)


def test_label_only_method_keeps_probability_metrics_explicitly_unavailable():
    data = _data()
    authority = _authority(data)
    protocol = _protocol(data, authority)
    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        LabelOnlyFactory(),
        execution_context=_context(),
    )

    row = result.rows[0]
    assert row.status == "success"
    assert row.probability_available is False
    assert row.score is not None
    assert row.score.availability["balanced_accuracy"] == "available"
    assert row.score.availability["accuracy"] == "available"
    assert row.score.metrics["brier_score"] is None
    assert row.score.availability["brier_score"] == "unavailable_probability_output"
    assert row.score.metrics["expected_calibration_error"] is None


def test_probability_method_requires_exact_fitted_class_order():
    data = _data()
    authority = _authority(data)
    protocol = _protocol(data, authority)
    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        ProbabilityFactory(reverse_probability_order=True),
        execution_context=_context(),
    )

    assert len(result.rows) == len(protocol.calibration_budgets_per_class)
    assert all(row.status == "failed" for row in result.rows)
    assert all(row.failure_type == "ValueError" for row in result.rows)
    assert all("probability class order" in row.failure_reason for row in result.rows)


def test_probability_method_scores_only_after_validated_class_order():
    data = _data()
    authority = _authority(data)
    protocol = _protocol(data, authority)
    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        ProbabilityFactory(),
        execution_context=_context(),
    )

    row = result.rows[0]
    assert row.status == "success"
    assert row.probability_available is True
    assert row.score is not None
    assert row.score.availability["brier_score"] == "available"
    assert row.score.availability["expected_calibration_error"] == "available"
    assert row.score.availability["roc_auc"] == "available"
    assert row.learned_state_addressable is True


def test_external_failures_remain_in_the_frontier_instead_of_disappearing():
    data = _data()
    authority = _authority(data)
    protocol = _protocol(data, authority)
    failures: list[Exception | None] = [
        None,
        ImportError("optional backend absent"),
        MemoryError("synthetic oom"),
        RuntimeError("optimizer exploded"),
    ]
    assert len(failures) == len(protocol.calibration_budgets_per_class)

    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        FailingFactory(failures),
        execution_context=_context(),
    )

    assert [row.status for row in result.rows] == [
        "success",
        "unavailable",
        "oom",
        "failed",
    ]
    assert len(result.rows) == len(protocol.calibration_budgets_per_class)
    assert result.rows[1].failure_reason == "optional backend absent"
    assert result.rows[2].failure_type == "MemoryError"


def test_observed_dataset_lineage_mismatch_fails_before_external_model_creation():
    data = _data()
    authority = _authority(data)
    protocol = _protocol(data, authority)
    factory = LabelOnlyFactory()
    wrong = QualificationExecutionContext(
        observed_dataset_lineage_sha256="f" * 64,
    )

    with pytest.raises(ValueError, match="observed dataset lineage"):
        run_external_qualification_case(
            data,
            authority,
            protocol,
            factory,
            execution_context=wrong,
        )
    assert factory.created == []


def test_metric_authority_mismatch_fails_before_external_model_creation():
    data = _data()
    authority = _authority(data)
    protocol = replace(_protocol(data, authority), metric_scorecard_sha256="f" * 64)
    factory = LabelOnlyFactory()

    with pytest.raises(ValueError, match="metric scorecard"):
        run_external_qualification_case(
            data,
            authority,
            protocol,
            factory,
            execution_context=_context(),
        )
    assert factory.created == []


def test_processed_array_tampering_is_rejected_by_existing_case_authority():
    data = _data()
    authority = _authority(data)
    protocol = _protocol(data, authority)
    factory = LabelOnlyFactory()
    tampered_X = data.X.copy()
    tampered_X[0, 0, 0] += 0.125
    tampered = GroupedEvaluationData(
        dataset_id=data.dataset_id,
        X=tampered_X,
        y=data.y,
        groups=data.groups,
    )

    with pytest.raises(ValueError, match="processed neural data SHA-256"):
        run_external_qualification_case(
            tampered,
            authority,
            protocol,
            factory,
            execution_context=_context(),
        )
    assert factory.created == []


def test_protocol_budget_beyond_case_authority_fails_before_fit():
    data = _data()
    authority = _authority(data)
    protocol = replace(
        _protocol(data, authority),
        calibration_budgets_per_class=(0, 99),
    )
    factory = LabelOnlyFactory()

    with pytest.raises(ValueError, match="exceeds frozen authority maximum"):
        run_external_qualification_case(
            data,
            authority,
            protocol,
            factory,
            execution_context=_context(),
        )
    assert factory.created == []


def test_runner_refuses_draft_protocol_even_when_every_other_hash_matches():
    data = _data()
    authority = _authority(data)
    protocol = replace(_protocol(data, authority), protocol_status="draft")
    with pytest.raises(ValueError, match="requires protocol_status='frozen'"):
        run_external_qualification_case(
            data,
            authority,
            protocol,
            LabelOnlyFactory(),
            execution_context=_context(),
        )
