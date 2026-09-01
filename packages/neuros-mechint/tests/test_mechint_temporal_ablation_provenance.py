import numpy as np
import pytest

from neuros_mechint.representations.contracts import (
    EvaluationScope,
    FitRegime,
    RepresentationEmbedding,
    SequenceBatch,
)
from neuros_mechint.representations.corruptions import (
    make_controlled_corruption_manifold,
)
from neuros_mechint.representations.temporal_ablation import (
    TemporalOrderInterventionRepresentation,
)


def _batches():
    train = SequenceBatch(
        sequences=(np.arange(36, dtype=float).reshape(12, 3),),
        sequence_ids=("train",),
    )
    evaluation = SequenceBatch(
        sequences=(np.arange(30, dtype=float).reshape(10, 3),),
        sequence_ids=("eval",),
    )
    return train, evaluation


class _BaseSequenceLocal:
    method_id = "declared"
    fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE
    evaluation_scope = EvaluationScope.SEQUENCE_LOCAL

    def _embedding(self, evaluation, *, method_id=None, fit_regime=None):
        return RepresentationEmbedding(
            method_id=method_id or self.method_id,
            sequences=tuple(sequence[:, :2] for sequence in evaluation.sequences),
            sequence_ids=evaluation.sequence_ids,
            fit_regime=fit_regime or self.fit_regime,
        )


class _LyingMethodId(_BaseSequenceLocal):
    def embed(self, train, evaluation):
        return self._embedding(evaluation, method_id="different")


class _LyingFitRegime(_BaseSequenceLocal):
    def embed(self, train, evaluation):
        return self._embedding(
            evaluation,
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
        )


def test_temporal_intervention_rejects_wrapped_method_id_drift():
    train, evaluation = _batches()
    wrapped = TemporalOrderInterventionRepresentation(_LyingMethodId())
    with pytest.raises(ValueError, match="method_id"):
        wrapped.embed(train, evaluation)


def test_temporal_intervention_rejects_wrapped_fit_regime_drift():
    train, evaluation = _batches()
    wrapped = TemporalOrderInterventionRepresentation(_LyingFitRegime())
    with pytest.raises(ValueError, match="fit_regime"):
        wrapped.embed(train, evaluation)


@pytest.mark.parametrize(
    ("kwargs", "error"),
    [
        ({"seed": -1}, ValueError),
        ({"ar_coefficient": np.nan}, ValueError),
        ({"spike_probability": np.inf}, ValueError),
        ({"drift_cycles": np.nan}, ValueError),
    ],
)
def test_corruption_metadata_parameters_fail_closed_even_when_not_active(
    kwargs,
    error,
):
    call_kwargs = {"seed": 1, **kwargs}
    with pytest.raises(error):
        make_controlled_corruption_manifold(
            corruption="iid_gaussian",
            corruption_scale=0.0,
            **call_kwargs,
        )
