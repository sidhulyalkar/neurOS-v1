from __future__ import annotations

import numpy as np
import torch

from neuros_mechint.representations import (
    AutoencoderRepresentation,
    FitRegime,
    LagPredictiveAutoencoderRepresentation,
    SequenceBatch,
    build_lagged_training_pairs,
)


def _batch() -> SequenceBatch:
    return SequenceBatch(
        sequences=(
            np.column_stack(
                (
                    np.arange(6, dtype=float),
                    100.0 + np.arange(6, dtype=float),
                )
            ),
            np.column_stack(
                (
                    1000.0 + np.arange(5, dtype=float),
                    2000.0 + np.arange(5, dtype=float),
                )
            ),
        ),
        sequence_ids=("run-a", "run-b"),
    )


def test_lagged_pairs_are_exact_and_never_cross_sequence_boundaries() -> None:
    batch = _batch()
    inputs, targets, pair_ids = build_lagged_training_pairs(batch, lag=1)

    assert inputs.shape == targets.shape == (9, 2)
    assert pair_ids == ("run-a",) * 5 + ("run-b",) * 4
    np.testing.assert_array_equal(inputs[:5], batch.sequences[0][:-1])
    np.testing.assert_array_equal(targets[:5], batch.sequences[0][1:])
    np.testing.assert_array_equal(inputs[5:], batch.sequences[1][:-1])
    np.testing.assert_array_equal(targets[5:], batch.sequences[1][1:])


def test_shuffled_successor_null_preserves_each_sequence_target_pool() -> None:
    batch = _batch()
    ordered_inputs, ordered_targets, ordered_ids = build_lagged_training_pairs(
        batch,
        lag=1,
        shuffle_targets=False,
        seed=17,
    )
    shuffled_inputs, shuffled_targets, shuffled_ids = build_lagged_training_pairs(
        batch,
        lag=1,
        shuffle_targets=True,
        seed=17,
    )

    np.testing.assert_array_equal(shuffled_inputs, ordered_inputs)
    assert shuffled_ids == ordered_ids
    assert not np.array_equal(shuffled_targets[:5], ordered_targets[:5])
    assert not np.array_equal(shuffled_targets[5:], ordered_targets[5:])

    for start, stop in ((0, 5), (5, 9)):
        expected = sorted(map(tuple, ordered_targets[start:stop].tolist()))
        observed = sorted(map(tuple, shuffled_targets[start:stop].tolist()))
        assert observed == expected


def test_predictive_representation_is_deterministic_and_train_only() -> None:
    train = _batch()
    evaluation = SequenceBatch(
        sequences=(
            np.column_stack(
                (
                    np.linspace(0.5, 4.5, 5),
                    np.linspace(100.5, 104.5, 5),
                )
            ),
        ),
        sequence_ids=("eval",),
    )

    left = LagPredictiveAutoencoderRepresentation(
        2,
        hidden_dim=8,
        epochs=8,
        batch_size=4,
        seed=23,
        method_id="predictive",
    )
    right = LagPredictiveAutoencoderRepresentation(
        2,
        hidden_dim=8,
        epochs=8,
        batch_size=4,
        seed=23,
        method_id="predictive",
    )
    left_embedding = left.embed(train, evaluation)
    right_embedding = right.embed(train, evaluation)

    assert left_embedding.fit_regime is FitRegime.TRAIN_ONLY_INDUCTIVE
    assert left.training_pair_count_ == 9
    assert left.training_loss_ == right.training_loss_
    np.testing.assert_array_equal(left.mean_, right.mean_)
    np.testing.assert_array_equal(left.scale_, right.scale_)
    np.testing.assert_array_equal(
        left_embedding.sequences[0],
        right_embedding.sequences[0],
    )
    assert left_embedding.metadata["sequence_boundary_policy"] == "never_cross"
    assert left_embedding.metadata["target_mode"] == "within_sequence_successor"
    assert left_embedding.metadata["target_specific_fit_observations"] == 0

    assert left.model_ is not None
    assert right.model_ is not None
    for name, tensor in left.model_.state_dict().items():
        assert torch.equal(tensor, right.model_.state_dict()[name])


def test_evaluation_values_cannot_change_fitted_predictive_parameters() -> None:
    train = _batch()
    evaluation_a = SequenceBatch(
        sequences=(np.arange(10, dtype=float).reshape(5, 2),),
        sequence_ids=("eval",),
    )
    evaluation_b = SequenceBatch(
        sequences=((10000.0 + np.arange(10, dtype=float)).reshape(5, 2),),
        sequence_ids=("eval",),
    )

    model_a = LagPredictiveAutoencoderRepresentation(
        2,
        hidden_dim=8,
        epochs=6,
        batch_size=4,
        seed=31,
    )
    model_b = LagPredictiveAutoencoderRepresentation(
        2,
        hidden_dim=8,
        epochs=6,
        batch_size=4,
        seed=31,
    )
    model_a.embed(train, evaluation_a)
    model_b.embed(train, evaluation_b)

    assert model_a.training_loss_ == model_b.training_loss_
    np.testing.assert_array_equal(model_a.mean_, model_b.mean_)
    np.testing.assert_array_equal(model_a.scale_, model_b.scale_)
    assert model_a.model_ is not None
    assert model_b.model_ is not None
    for name, tensor in model_a.model_.state_dict().items():
        assert torch.equal(tensor, model_b.model_.state_dict()[name])


def test_shuffled_predictive_representation_declares_null_semantics() -> None:
    train = _batch()
    evaluation = SequenceBatch(
        sequences=(np.arange(10, dtype=float).reshape(5, 2),),
        sequence_ids=("eval",),
    )
    method = LagPredictiveAutoencoderRepresentation(
        2,
        hidden_dim=8,
        epochs=2,
        batch_size=4,
        seed=7,
        shuffle_targets=True,
        method_id="predictive_shuffled",
    )
    embedding = method.embed(train, evaluation)

    assert embedding.metadata["target_mode"] == "within_sequence_shuffled_successor"
    assert embedding.metadata["sequence_boundary_policy"] == "never_cross"


def test_predictive_and_reconstruction_controls_share_parameter_shapes() -> None:
    train = _batch()
    evaluation = SequenceBatch(
        sequences=(np.arange(10, dtype=float).reshape(5, 2),),
        sequence_ids=("eval",),
    )
    reconstruction = AutoencoderRepresentation(
        2,
        hidden_dim=8,
        epochs=1,
        batch_size=4,
        seed=5,
        method_id="reconstruction",
    )
    predictive = LagPredictiveAutoencoderRepresentation(
        2,
        hidden_dim=8,
        epochs=1,
        batch_size=4,
        seed=5,
        method_id="predictive",
    )
    reconstruction.embed(train, evaluation)
    predictive.embed(train, evaluation)

    assert reconstruction.model_ is not None
    assert predictive.model_ is not None
    reconstruction_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in reconstruction.model_.state_dict().items()
    }
    predictive_shapes = {
        name: tuple(tensor.shape)
        for name, tensor in predictive.model_.state_dict().items()
    }
    assert predictive_shapes == reconstruction_shapes
