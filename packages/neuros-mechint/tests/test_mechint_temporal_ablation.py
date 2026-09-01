import numpy as np
import pytest

from neuros_mechint.representations.contracts import (
    EvaluationScope,
    FitRegime,
    RepresentationEmbedding,
    SequenceBatch,
)
from neuros_mechint.representations.corruptions import (
    TemporalCorruption,
    make_controlled_corruption_manifold,
)
from neuros_mechint.representations.temporal_ablation import (
    TemporalOrderInterventionRepresentation,
)


def _batches():
    train = SequenceBatch(
        sequences=(np.arange(36, dtype=float).reshape(12, 3) / 10.0,),
        sequence_ids=("train",),
    )
    evaluation = SequenceBatch(
        sequences=(np.arange(30, dtype=float).reshape(10, 3) / 7.0,),
        sequence_ids=("eval",),
    )
    return train, evaluation


class _IdentitySequenceLocal:
    method_id = "identity_local"
    fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE
    evaluation_scope = EvaluationScope.SEQUENCE_LOCAL

    def embed(self, train, evaluation):
        return RepresentationEmbedding(
            method_id=self.method_id,
            sequences=tuple(sequence[:, :2] for sequence in evaluation.sequences),
            sequence_ids=evaluation.sequence_ids,
            fit_regime=self.fit_regime,
            metadata={"fake": "identity"},
        )


class _CumulativeSequenceLocal(_IdentitySequenceLocal):
    method_id = "cumulative_local"

    def embed(self, train, evaluation):
        return RepresentationEmbedding(
            method_id=self.method_id,
            sequences=tuple(np.cumsum(sequence[:, :2], axis=0) for sequence in evaluation.sequences),
            sequence_ids=evaluation.sequence_ids,
            fit_regime=self.fit_regime,
        )


class _BatchMethod(_IdentitySequenceLocal):
    method_id = "batch_method"
    evaluation_scope = EvaluationScope.BATCH_TRANSFORM


def test_temporal_intervention_restores_exact_row_identity_for_order_invariant_method():
    train, evaluation = _batches()
    wrapped = TemporalOrderInterventionRepresentation(
        _IdentitySequenceLocal(), seed=11, block_size=2
    )
    output = wrapped.embed(train, evaluation)
    np.testing.assert_array_equal(output.sequences[0], evaluation.sequences[0][:, :2])
    assert output.sequence_ids == evaluation.sequence_ids
    assert output.metadata["row_identity_policy"] == "inverse_permute_embedding_before_scoring"
    assert output.metadata["intervention"] == "temporal_order_destroyed"


def test_temporal_intervention_changes_order_sensitive_representation():
    train, evaluation = _batches()
    base_method = _CumulativeSequenceLocal()
    baseline = base_method.embed(train, evaluation)
    intervened = TemporalOrderInterventionRepresentation(
        base_method, seed=5, block_size=1
    ).embed(train, evaluation)
    assert not np.allclose(intervened.sequences[0], baseline.sequences[0])


def test_temporal_intervention_is_deterministic_and_seed_bound():
    train, evaluation = _batches()
    first = TemporalOrderInterventionRepresentation(
        _IdentitySequenceLocal(), seed=3, block_size=2
    ).embed(train, evaluation)
    again = TemporalOrderInterventionRepresentation(
        _IdentitySequenceLocal(), seed=3, block_size=2
    ).embed(train, evaluation)
    other = TemporalOrderInterventionRepresentation(
        _IdentitySequenceLocal(), seed=4, block_size=2
    ).embed(train, evaluation)
    first_digest = first.metadata["permutation_sha256_by_sequence"]["eval"]
    assert first_digest == again.metadata["permutation_sha256_by_sequence"]["eval"]
    assert first_digest != other.metadata["permutation_sha256_by_sequence"]["eval"]


def test_temporal_intervention_rejects_batch_scoped_methods():
    with pytest.raises(ValueError, match="sequence-local"):
        TemporalOrderInterventionRepresentation(_BatchMethod())


def test_temporal_intervention_rejects_noop_block_geometry():
    train, evaluation = _batches()
    wrapped = TemporalOrderInterventionRepresentation(
        _IdentitySequenceLocal(), block_size=evaluation.sequences[0].shape[0]
    )
    with pytest.raises(ValueError, match="at least two temporal blocks"):
        wrapped.embed(train, evaluation)


@pytest.mark.parametrize("kind", list(TemporalCorruption))
def test_controlled_corruptions_are_exactly_deterministic(kind):
    first = make_controlled_corruption_manifold(
        corruption=kind, corruption_scale=0.4, seed=17
    )
    second = make_controlled_corruption_manifold(
        corruption=kind, corruption_scale=0.4, seed=17
    )
    np.testing.assert_array_equal(first.train.sequences[0], second.train.sequences[0])
    np.testing.assert_array_equal(
        first.evaluation.sequences[0], second.evaluation.sequences[0]
    )
    np.testing.assert_array_equal(
        first.reference.sequences[0], second.reference.sequences[0]
    )
    assert first.metadata["corruption_kind"] == kind.value


def test_zero_scale_has_identical_clean_observations_across_corruption_kinds():
    fixtures = [
        make_controlled_corruption_manifold(
            corruption=kind, corruption_scale=0.0, seed=19
        )
        for kind in TemporalCorruption
    ]
    for fixture in fixtures[1:]:
        np.testing.assert_array_equal(
            fixture.train.sequences[0], fixtures[0].train.sequences[0]
        )
        np.testing.assert_array_equal(
            fixture.evaluation.sequences[0], fixtures[0].evaluation.sequences[0]
        )
        np.testing.assert_array_equal(
            fixture.reference.sequences[0], fixtures[0].reference.sequences[0]
        )


@pytest.mark.parametrize("kind", list(TemporalCorruption))
def test_corruption_scale_changes_only_amplitude_for_fixed_seed_and_kind(kind):
    clean = make_controlled_corruption_manifold(
        corruption=kind, corruption_scale=0.0, seed=23
    )
    low = make_controlled_corruption_manifold(
        corruption=kind, corruption_scale=0.2, seed=23
    )
    high = make_controlled_corruption_manifold(
        corruption=kind, corruption_scale=0.8, seed=23
    )
    low_residual = low.evaluation.sequences[0] - clean.evaluation.sequences[0]
    high_residual = high.evaluation.sequences[0] - clean.evaluation.sequences[0]
    np.testing.assert_allclose(high_residual, 4.0 * low_residual, atol=1e-12, rtol=1e-12)


def _mean_lag_one_correlation(values):
    correlations = []
    for feature in range(values.shape[1]):
        correlations.append(np.corrcoef(values[:-1, feature], values[1:, feature])[0, 1])
    return float(np.nanmean(correlations))


def test_ar1_fixture_has_strong_temporal_autocorrelation_relative_to_iid():
    clean = make_controlled_corruption_manifold(
        corruption=TemporalCorruption.IID_GAUSSIAN, corruption_scale=0.0, seed=29
    )
    iid = make_controlled_corruption_manifold(
        corruption=TemporalCorruption.IID_GAUSSIAN, corruption_scale=1.0, seed=29
    )
    ar1 = make_controlled_corruption_manifold(
        corruption=TemporalCorruption.AR1,
        corruption_scale=1.0,
        seed=29,
        ar_coefficient=0.9,
    )
    iid_residual = iid.evaluation.sequences[0] - clean.evaluation.sequences[0]
    ar1_residual = ar1.evaluation.sequences[0] - clean.evaluation.sequences[0]
    assert _mean_lag_one_correlation(ar1_residual) > 0.7
    assert _mean_lag_one_correlation(ar1_residual) > _mean_lag_one_correlation(iid_residual) + 0.5


def test_sparse_spike_fixture_is_actually_sparse():
    clean = make_controlled_corruption_manifold(
        corruption=TemporalCorruption.SPARSE_SPIKES, corruption_scale=0.0, seed=31
    )
    spiky = make_controlled_corruption_manifold(
        corruption=TemporalCorruption.SPARSE_SPIKES,
        corruption_scale=1.0,
        seed=31,
        spike_probability=0.03,
    )
    residual = spiky.evaluation.sequences[0] - clean.evaluation.sequences[0]
    fraction = float(np.count_nonzero(residual) / residual.size)
    assert 0.005 < fraction < 0.08


def test_slow_drift_fixture_is_smoother_than_iid_corruption():
    clean = make_controlled_corruption_manifold(
        corruption=TemporalCorruption.IID_GAUSSIAN, corruption_scale=0.0, seed=37
    )
    iid = make_controlled_corruption_manifold(
        corruption=TemporalCorruption.IID_GAUSSIAN, corruption_scale=1.0, seed=37
    )
    drift = make_controlled_corruption_manifold(
        corruption=TemporalCorruption.SLOW_DRIFT,
        corruption_scale=1.0,
        seed=37,
        drift_cycles=1.5,
    )
    iid_residual = iid.evaluation.sequences[0] - clean.evaluation.sequences[0]
    drift_residual = drift.evaluation.sequences[0] - clean.evaluation.sequences[0]
    iid_step = float(np.mean(np.linalg.norm(np.diff(iid_residual, axis=0), axis=1)))
    drift_step = float(np.mean(np.linalg.norm(np.diff(drift_residual, axis=0), axis=1)))
    assert drift_step < iid_step * 0.25


def test_corruption_contract_rejects_invalid_scales_and_parameters():
    with pytest.raises(TypeError):
        make_controlled_corruption_manifold(
            corruption="iid_gaussian", corruption_scale=True, seed=1
        )
    with pytest.raises(ValueError):
        make_controlled_corruption_manifold(
            corruption="ar1", corruption_scale=1.0, seed=1, ar_coefficient=1.0
        )
    with pytest.raises(ValueError):
        make_controlled_corruption_manifold(
            corruption="sparse_spikes",
            corruption_scale=1.0,
            seed=1,
            spike_probability=0.0,
        )
    with pytest.raises(ValueError):
        make_controlled_corruption_manifold(
            corruption="slow_drift", corruption_scale=1.0, seed=1, drift_cycles=0.0
        )
