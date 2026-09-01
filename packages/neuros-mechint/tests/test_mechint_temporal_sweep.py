import numpy as np
import pytest

from neuros_mechint.representations.cases import CaseStatus
from neuros_mechint.representations.contracts import (
    EvaluationScope,
    FitRegime,
    RepresentationEmbedding,
    RepresentationUnavailableError,
)
from neuros_mechint.representations.corruptions import TemporalCorruption
from neuros_mechint.representations.pca import PCARepresentation
from neuros_mechint.representations.temporal_ablation import (
    TemporalOrderInterventionRepresentation,
)
from neuros_mechint.representations.temporal_sweep import (
    ControlledTemporalAblationResult,
    run_controlled_temporal_ablation,
)


class _CumulativeSequenceLocal:
    method_id = "cumulative"
    fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE
    evaluation_scope = EvaluationScope.SEQUENCE_LOCAL

    def embed(self, train, evaluation):
        return RepresentationEmbedding(
            method_id=self.method_id,
            sequences=tuple(
                np.cumsum(sequence[:, :3], axis=0)
                for sequence in evaluation.sequences
            ),
            sequence_ids=evaluation.sequence_ids,
            fit_regime=self.fit_regime,
        )


class _UnavailableSequenceLocal(_CumulativeSequenceLocal):
    method_id = "unavailable_fake"

    def embed(self, train, evaluation):
        raise RepresentationUnavailableError("fixture dependency unavailable")


def _method_factory():
    base = _CumulativeSequenceLocal()
    return (
        PCARepresentation(n_components=3),
        base,
        TemporalOrderInterventionRepresentation(
            _CumulativeSequenceLocal(),
            seed=101,
            block_size=4,
        ),
    )


def _small_result():
    return run_controlled_temporal_ablation(
        _method_factory,
        corruptions=(TemporalCorruption.IID_GAUSSIAN, TemporalCorruption.AR1),
        corruption_scales=(0.0, 0.5),
        seeds=(0, 1),
        neighborhood_k=5,
    )


def test_temporal_sweep_preserves_exact_factorial_case_grid():
    result = _small_result()
    assert result.metadata["schema"] == (
        "neuros.representation.controlled_temporal_ablation.v1"
    )
    assert result.metadata["ranking_policy"] == "none"
    assert result.method_ids == (
        "pca",
        "cumulative",
        "cumulative__temporal_order_destroyed",
    )
    assert result.evaluation_sequence_ids == ("eval",)
    assert len(result.records) == 24
    assert len(result.summaries()) == 12
    assert all(record.status is CaseStatus.OK for record in result.records)


def test_temporal_sweep_exposes_seed_denominators_per_metric():
    result = _small_result()
    for summary in result.summaries():
        assert summary.total_cases == 2
        assert summary.ok_cases == 2
        assert summary.non_ok_rate == 0.0
        assert summary.failed_rate == 0.0
        assert summary.unavailable_rate == 0.0
        assert summary.nonconverged_rate == 0.0
        assert summary.metric_n["reference_pairwise_distance_rank"] == 2


def test_temporal_order_negative_control_is_detectable_in_pilot_metrics():
    result = _small_result()
    baseline = result.summary(
        "cumulative",
        TemporalCorruption.AR1,
        0.5,
    )
    destroyed = result.summary(
        "cumulative__temporal_order_destroyed",
        TemporalCorruption.AR1,
        0.5,
    )
    baseline_metric = baseline.metric_mean["reference_pairwise_distance_rank"]
    destroyed_metric = destroyed.metric_mean["reference_pairwise_distance_rank"]
    assert baseline_metric is not None
    assert destroyed_metric is not None
    assert not np.isclose(baseline_metric, destroyed_metric)


def test_temporal_sweep_is_deterministic_for_fixed_declared_grid():
    first = _small_result()
    second = _small_result()
    first_rows = [
        (
            row.corruption,
            row.corruption_scale,
            row.seed,
            row.method_id,
            row.sequence_id,
            row.status,
            dict(row.metrics),
        )
        for row in first.records
    ]
    second_rows = [
        (
            row.corruption,
            row.corruption_scale,
            row.seed,
            row.method_id,
            row.sequence_id,
            row.status,
            dict(row.metrics),
        )
        for row in second.records
    ]
    assert first_rows == second_rows


def test_temporal_sweep_preserves_unavailable_rows_and_rates():
    result = run_controlled_temporal_ablation(
        lambda: (_UnavailableSequenceLocal(),),
        corruptions=(TemporalCorruption.SLOW_DRIFT,),
        corruption_scales=(0.5,),
        seeds=(0, 1, 2),
    )
    assert len(result.records) == 3
    assert all(
        record.status is CaseStatus.UNAVAILABLE for record in result.records
    )
    assert all(record.metrics == {} for record in result.records)
    summary = result.summaries()[0]
    assert summary.total_cases == 3
    assert summary.ok_cases == 0
    assert summary.unavailable_cases == 3
    assert summary.unavailable_rate == 1.0
    assert summary.failed_rate == 0.0
    assert summary.metric_n == {}


def test_temporal_sweep_rejects_incomplete_cartesian_evidence():
    result = _small_result()
    with pytest.raises(ValueError, match="exact declared"):
        ControlledTemporalAblationResult(
            corruptions=result.corruptions,
            corruption_scales=result.corruption_scales,
            seeds=result.seeds,
            method_ids=result.method_ids,
            evaluation_sequence_ids=result.evaluation_sequence_ids,
            records=result.records[:-1],
            metadata=result.metadata,
        )


def test_temporal_sweep_rejects_duplicate_grid_axes():
    with pytest.raises(ValueError, match="corruptions"):
        run_controlled_temporal_ablation(
            _method_factory,
            corruptions=("iid_gaussian", "iid_gaussian"),
            corruption_scales=(0.0,),
            seeds=(0,),
        )
    with pytest.raises(ValueError, match="corruption_scales"):
        run_controlled_temporal_ablation(
            _method_factory,
            corruptions=("iid_gaussian",),
            corruption_scales=(0.0, 0.0),
            seeds=(0,),
        )


def test_temporal_sweep_rejects_method_authority_drift_across_points():
    calls = {"count": 0}

    def drifting_factory():
        calls["count"] += 1
        method = _CumulativeSequenceLocal()
        if calls["count"] > 1:
            method.method_id = "changed"
        return (method,)

    with pytest.raises(ValueError, match="same ordered method IDs"):
        run_controlled_temporal_ablation(
            drifting_factory,
            corruptions=("iid_gaussian",),
            corruption_scales=(0.0, 0.5),
            seeds=(0,),
        )
