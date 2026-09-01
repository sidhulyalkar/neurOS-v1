from __future__ import annotations

import numpy as np
import pytest

from neuros_mechint.representations import (
    EvaluationScope,
    FitRegime,
    PCARepresentation,
    RepresentationEmbedding,
    RepresentationUnavailableError,
)
from neuros_mechint.representations.cases import CaseStatus
from neuros_mechint.representations.sweep import (
    ControlledNoiseSweepResult,
    SweepCaseRecord,
    run_controlled_noise_sweep,
)
from neuros_mechint.representations.synthetic import make_controlled_temporal_manifold


def test_controlled_generator_is_exactly_reproducible() -> None:
    first = make_controlled_temporal_manifold(noise_std=0.35, seed=4)
    second = make_controlled_temporal_manifold(noise_std=0.35, seed=4)
    np.testing.assert_array_equal(
        first.train.sequences[0], second.train.sequences[0]
    )
    np.testing.assert_array_equal(
        first.evaluation.sequences[0], second.evaluation.sequences[0]
    )
    np.testing.assert_array_equal(
        first.reference.sequences[0], second.reference.sequences[0]
    )
    assert dict(first.metadata) == dict(second.metadata)


def test_noise_sweep_changes_amplitude_not_reference_geometry() -> None:
    clean = make_controlled_temporal_manifold(noise_std=0.0, seed=9)
    noisy = make_controlled_temporal_manifold(noise_std=0.75, seed=9)
    np.testing.assert_array_equal(
        clean.reference.sequences[0], noisy.reference.sequences[0]
    )
    assert not np.array_equal(
        clean.evaluation.sequences[0], noisy.evaluation.sequences[0]
    )
    assert (
        clean.metadata["coupled_noise_policy"]
        == "fixed_seed_reuses_mixing_and_standardized_noise_across_noise_levels"
    )


def test_sweep_preserves_exact_noise_seed_method_grid() -> None:
    result = run_controlled_noise_sweep(
        lambda: (PCARepresentation(3),),
        noise_levels=(0.0, 0.35, 0.75),
        seeds=(1, 2, 3),
        neighborhood_k=5,
    )
    assert len(result.records) == 9
    assert result.noise_levels == (0.0, 0.35, 0.75)
    assert result.seeds == (1, 2, 3)
    assert result.method_ids == ("pca",)
    assert all(record.status is CaseStatus.OK for record in result.records)
    assert not hasattr(result, "winner")
    assert result.metadata["ranking_policy"] == "none"


def test_sweep_summary_reports_seed_uncertainty_and_denominator() -> None:
    result = run_controlled_noise_sweep(
        lambda: (PCARepresentation(3),),
        noise_levels=(0.2,),
        seeds=(1, 2, 3, 4),
    )
    summary = result.summary("pca", 0.2)
    assert summary.total_cases == 4
    assert summary.ok_cases == 4
    assert summary.non_ok_rate == 0.0
    assert summary.failed_rate == 0.0
    assert summary.metric_n["reference_pairwise_distance_rank"] == 4
    assert summary.metadata["declared_seed_count"] == 4
    assert summary.metric_mean["reference_pairwise_distance_rank"] is not None
    assert summary.metric_std["reference_pairwise_distance_rank"] is not None
    assert summary.metric_sem["reference_pairwise_distance_rank"] is not None


def test_unavailable_method_stays_in_every_sweep_point() -> None:
    class UnavailableMethod:
        method_id = "unavailable"
        fit_regime = FitRegime.TRANSDUCTIVE_TARGET_OBSERVED
        evaluation_scope = EvaluationScope.SEQUENCE_LOCAL

        def embed(self, train, evaluation):
            raise RepresentationUnavailableError("optional dependency absent")

    result = run_controlled_noise_sweep(
        lambda: (PCARepresentation(2), UnavailableMethod()),
        noise_levels=(0.0, 0.5),
        seeds=(1, 2),
    )
    unavailable = [
        record for record in result.records if record.method_id == "unavailable"
    ]
    assert len(unavailable) == 4
    assert all(
        record.status is CaseStatus.UNAVAILABLE for record in unavailable
    )
    for noise in result.noise_levels:
        summary = result.summary("unavailable", noise)
        assert summary.total_cases == 2
        assert summary.ok_cases == 0
        assert summary.unavailable_cases == 2
        assert summary.non_ok_rate == 1.0
        assert summary.failed_rate == 0.0
        assert summary.unavailable_rate == 1.0


def test_method_factory_identity_cannot_drift_across_points() -> None:
    calls = 0

    class Identity:
        fit_regime = FitRegime.EXTERNAL_PRETRAINED
        evaluation_scope = EvaluationScope.SEQUENCE_LOCAL

        def __init__(self, method_id):
            self.method_id = method_id

        def embed(self, train, evaluation):
            source = evaluation.sequences[0]
            return RepresentationEmbedding(
                method_id=self.method_id,
                sequences=(source[:, :2],),
                sequence_ids=evaluation.sequence_ids,
                fit_regime=self.fit_regime,
            )

    def factory():
        nonlocal calls
        calls += 1
        return (Identity("first" if calls == 1 else "drifted"),)

    with pytest.raises(ValueError, match="same ordered method IDs"):
        run_controlled_noise_sweep(
            factory,
            noise_levels=(0.0, 0.1),
            seeds=(1,),
        )


def test_sweep_result_rejects_missing_grid_record() -> None:
    record = SweepCaseRecord(
        noise_std=0.0,
        seed=1,
        method_id="pca",
        sequence_id="eval",
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
        evaluation_scope=EvaluationScope.BATCH_TRANSFORM,
        status=CaseStatus.OK,
        metrics={"score": 1.0},
    )
    with pytest.raises(ValueError, match="exact declared noise"):
        ControlledNoiseSweepResult(
            noise_levels=(0.0, 0.5),
            seeds=(1,),
            method_ids=("pca",),
            evaluation_sequence_ids=("eval",),
            records=(record,),
        )



def test_sweep_metric_values_reject_bool_and_text_coercion() -> None:
    for value in (True, "0.5"):
        with pytest.raises(TypeError, match="finite real"):
            SweepCaseRecord(
                noise_std=0.0,
                seed=1,
                method_id="pca",
                sequence_id="eval",
                fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
                evaluation_scope=EvaluationScope.BATCH_TRANSFORM,
                status=CaseStatus.OK,
                metrics={"score": value},
            )


def test_method_factory_evaluation_scope_cannot_drift_across_points() -> None:
    calls = 0

    class ScopedIdentity:
        method_id = "scoped"
        fit_regime = FitRegime.EXTERNAL_PRETRAINED

        def __init__(self, scope):
            self.evaluation_scope = scope

        def embed(self, train, evaluation):
            source = evaluation.sequences[0]
            return RepresentationEmbedding(
                method_id=self.method_id,
                sequences=(source[:, :2],),
                sequence_ids=evaluation.sequence_ids,
                fit_regime=self.fit_regime,
            )

    def factory():
        nonlocal calls
        calls += 1
        scope = (
            EvaluationScope.SEQUENCE_LOCAL
            if calls == 1
            else EvaluationScope.BATCH_TRANSFORM
        )
        return (ScopedIdentity(scope),)

    with pytest.raises(ValueError, match="evaluation scopes"):
        run_controlled_noise_sweep(
            factory,
            noise_levels=(0.0, 0.1),
            seeds=(1,),
        )


def test_sweep_summary_rejects_metric_schema_drift() -> None:
    records = (
        SweepCaseRecord(
            noise_std=0.0,
            seed=1,
            method_id="x",
            sequence_id="eval",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
            status=CaseStatus.OK,
            metrics={"a": 1.0},
        ),
        SweepCaseRecord(
            noise_std=0.0,
            seed=2,
            method_id="x",
            sequence_id="eval",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
            status=CaseStatus.OK,
            metrics={"b": 1.0},
        ),
    )
    result = ControlledNoiseSweepResult(
        noise_levels=(0.0,),
        seeds=(1, 2),
        method_ids=("x",),
        evaluation_sequence_ids=("eval",),
        records=records,
    )
    with pytest.raises(ValueError, match="identical metric schema"):
        result.summary("x", 0.0)
