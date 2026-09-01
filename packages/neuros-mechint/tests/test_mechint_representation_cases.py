from __future__ import annotations

import numpy as np
import pytest

from neuros_mechint.representations import (
    FitRegime,
    PCARepresentation,
    SequenceBatch,
)
from neuros_mechint.representations.cases import (
    CasePreservingRepresentationBenchmark,
    CasePreservingRepresentationResult,
    CaseStatus,
    RepresentationCaseOutcome,
    RepresentationNonconvergenceError,
)


def _data() -> tuple[SequenceBatch, SequenceBatch, SequenceBatch]:
    rng = np.random.default_rng(22)
    train = SequenceBatch(
        sequences=(rng.normal(size=(12, 4)),),
        sequence_ids=("train",),
    )
    evaluation = SequenceBatch(
        sequences=(
            rng.normal(size=(8, 4)),
            rng.normal(size=(9, 4)),
            rng.normal(size=(10, 4)),
        ),
        sequence_ids=("a", "b", "c"),
    )
    reference = SequenceBatch(
        sequences=tuple(
            np.column_stack(
                [
                    np.linspace(0.0, 1.0, sequence.shape[0]),
                    np.sin(np.linspace(0.0, np.pi, sequence.shape[0])),
                ]
            )
            for sequence in evaluation.sequences
        ),
        sequence_ids=evaluation.sequence_ids,
    )
    return train, evaluation, reference


def test_inductive_method_fits_once_and_splits_all_cases() -> None:
    train, evaluation, reference = _data()

    class CountingPCA(PCARepresentation):
        def __init__(self) -> None:
            super().__init__(2, method_id="counting_pca")
            self.calls = 0

        def embed(self, train, evaluation):
            self.calls += 1
            return super().embed(train, evaluation)

    method = CountingPCA()
    result = CasePreservingRepresentationBenchmark([method], neighborhood_k=3).run(
        train,
        evaluation,
        reference=reference,
    )

    assert method.calls == 1
    assert len(result.cases) == 3
    assert all(case.status is CaseStatus.OK for case in result.cases)
    assert tuple(case.sequence_id for case in result.cases) == evaluation.sequence_ids
    summary = result.summary_for_method("counting_pca")
    assert summary.total_cases == 3
    assert summary.ok_cases == 3
    assert summary.failure_rate == 0.0
    assert summary.metadata["declared_total_cases"] == 3
    assert "reference_pairwise_distance_rank" in summary.metrics


def test_sequence_local_method_preserves_successful_siblings_when_one_case_fails() -> None:
    train, evaluation, _ = _data()
    seen: list[str] = []

    class SelectivelyBroken:
        method_id = "transductive"
        fit_regime = FitRegime.TRANSDUCTIVE_TARGET_OBSERVED

        def embed(self, train, evaluation):
            assert len(evaluation.sequences) == 1
            sequence_id = evaluation.sequence_ids[0]
            seen.append(sequence_id)
            if sequence_id == "b":
                raise RuntimeError("acf failure for b")
            source = evaluation.sequences[0]
            from neuros_mechint.representations import RepresentationEmbedding

            return RepresentationEmbedding(
                method_id=self.method_id,
                sequences=(source[:, :2],),
                sequence_ids=(sequence_id,),
                fit_regime=self.fit_regime,
                metadata={"target_specific_fit_observations": source.shape[0]},
            )

    result = CasePreservingRepresentationBenchmark(
        [SelectivelyBroken()], neighborhood_k=3
    ).run(train, evaluation)

    assert seen == ["a", "b", "c"]
    by_case = result.by_case()
    assert by_case[("transductive", "a")].status is CaseStatus.OK
    assert by_case[("transductive", "b")].status is CaseStatus.FAILED
    assert by_case[("transductive", "c")].status is CaseStatus.OK
    assert by_case[("transductive", "b")].metrics == {}
    assert by_case[("transductive", "b")].error_message == "acf failure for b"

    summary = result.summary_for_method("transductive")
    assert summary.total_cases == 3
    assert summary.ok_cases == 2
    assert summary.failed_cases == 1
    assert summary.failure_rate == pytest.approx(1 / 3)
    assert summary.metadata["successful_metric_cases"] == 2


def test_complete_cartesian_product_is_enforced() -> None:
    case = RepresentationCaseOutcome(
        method_id="pca",
        sequence_id="a",
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
        status=CaseStatus.OK,
        embedding=np.ones((4, 2)),
        metrics={"score": 1.0},
    )
    with pytest.raises(ValueError, match="Cartesian product"):
        CasePreservingRepresentationResult(
            train_sequence_ids=("train",),
            evaluation_sequence_ids=("a", "b"),
            method_ids=("pca",),
            cases=(case,),
        )


def test_duplicate_case_identity_is_rejected() -> None:
    case = RepresentationCaseOutcome(
        method_id="pca",
        sequence_id="a",
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
        status=CaseStatus.OK,
        embedding=np.ones((4, 2)),
        metrics={"score": 1.0},
    )
    with pytest.raises(ValueError, match="duplicate representation case"):
        CasePreservingRepresentationResult(
            train_sequence_ids=("train",),
            evaluation_sequence_ids=("a",),
            method_ids=("pca",),
            cases=(case, case),
        )


def test_failed_case_cannot_smuggle_scientific_metrics() -> None:
    with pytest.raises(ValueError, match="cannot carry scientific metric"):
        RepresentationCaseOutcome(
            method_id="tphate",
            sequence_id="a",
            fit_regime=FitRegime.TRANSDUCTIVE_TARGET_OBSERVED,
            status=CaseStatus.FAILED,
            metrics={"pairwise_distance_rank": 0.9},
            error_type="RuntimeError",
            error_message="failed",
        )


def test_nonconvergence_is_distinct_from_generic_failure() -> None:
    train, evaluation, _ = _data()

    class Nonconverged:
        method_id = "nonconverged"
        fit_regime = FitRegime.TRANSDUCTIVE_TARGET_OBSERVED

        def embed(self, train, evaluation):
            if evaluation.sequence_ids[0] == "c":
                raise RepresentationNonconvergenceError("optimizer exhausted")
            from neuros_mechint.representations import RepresentationEmbedding

            source = evaluation.sequences[0]
            return RepresentationEmbedding(
                method_id=self.method_id,
                sequences=(source[:, :2],),
                sequence_ids=evaluation.sequence_ids,
                fit_regime=self.fit_regime,
            )

    result = CasePreservingRepresentationBenchmark([Nonconverged()]).run(
        train, evaluation
    )
    summary = result.summary_for_method("nonconverged")
    assert summary.ok_cases == 2
    assert summary.nonconverged_cases == 1
    assert summary.failed_cases == 0
    assert result.by_case()[("nonconverged", "c")].status is CaseStatus.NONCONVERGED


def test_unavailable_sequence_local_cases_preserve_full_denominator() -> None:
    train, evaluation, _ = _data()

    class Unavailable:
        method_id = "external"
        fit_regime = FitRegime.EXTERNAL_PRETRAINED

        def embed(self, train, evaluation):
            from neuros_mechint.representations import (
                RepresentationEmbedding,
                RepresentationUnavailableError,
            )

            sequence_id = evaluation.sequence_ids[0]
            if sequence_id == "b":
                raise RepresentationUnavailableError("missing embedding b")
            source = evaluation.sequences[0]
            return RepresentationEmbedding(
                method_id=self.method_id,
                sequences=(source[:, :2],),
                sequence_ids=evaluation.sequence_ids,
                fit_regime=self.fit_regime,
            )

    result = CasePreservingRepresentationBenchmark([Unavailable()]).run(
        train, evaluation
    )
    summary = result.summary_for_method("external")
    assert summary.total_cases == 3
    assert summary.ok_cases == 2
    assert summary.unavailable_cases == 1
    assert summary.failure_rate == pytest.approx(1 / 3)


def test_multiple_methods_produce_exact_method_x_sequence_grid() -> None:
    train, evaluation, _ = _data()

    class IdentityExternal:
        method_id = "external"
        fit_regime = FitRegime.EXTERNAL_PRETRAINED

        def embed(self, train, evaluation):
            from neuros_mechint.representations import RepresentationEmbedding

            source = evaluation.sequences[0]
            return RepresentationEmbedding(
                method_id=self.method_id,
                sequences=(source[:, :2],),
                sequence_ids=evaluation.sequence_ids,
                fit_regime=self.fit_regime,
            )

    result = CasePreservingRepresentationBenchmark(
        [PCARepresentation(2), IdentityExternal()],
        neighborhood_k=3,
    ).run(train, evaluation)

    assert len(result.cases) == 6
    assert set(result.by_case()) == {
        (method_id, sequence_id)
        for method_id in ("pca", "external")
        for sequence_id in evaluation.sequence_ids
    }
    assert not hasattr(result, "winner")
    assert result.metadata["ranking_policy"] == "none"
