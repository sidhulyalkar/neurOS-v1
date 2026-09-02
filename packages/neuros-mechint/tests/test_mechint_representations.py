from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neuros_mechint.representations import (
    AutoencoderRepresentation,
    FitRegime,
    MethodStatus,
    PCARepresentation,
    PrecomputedTemporalSSLRepresentation,
    RepresentationBenchmark,
    SequenceBatch,
    TPHATEEmbeddingError,
    TPHATERepresentation,
    TPHATEUnavailableError,
    aggregate_geometry_metrics,
)
import neuros_mechint.representations.tphate as tphate_module


def _batches() -> tuple[SequenceBatch, SequenceBatch]:
    rng = np.random.default_rng(12)
    train = SequenceBatch(
        sequences=(
            rng.normal(size=(12, 4)),
            rng.normal(loc=0.5, size=(10, 4)),
        ),
        sequence_ids=("train-a", "train-b"),
        metadata={"nested": {"roles": ["train"]}},
    )
    evaluation = SequenceBatch(
        sequences=(
            rng.normal(size=(9, 4)),
            rng.normal(loc=-0.5, size=(8, 4)),
        ),
        sequence_ids=("eval-a", "eval-b"),
    )
    return train, evaluation


def test_sequence_batch_detaches_arrays_and_nested_metadata() -> None:
    array = np.arange(20, dtype=float).reshape(5, 4)
    nested = ["train"]
    batch = SequenceBatch(
        sequences=(array,),
        sequence_ids=("s1",),
        metadata={"nested": {"roles": nested}},
    )
    array[:] = -99
    nested.append("mutated")

    np.testing.assert_array_equal(
        batch.sequences[0],
        np.arange(20, dtype=float).reshape(5, 4),
    )
    assert batch.metadata["nested"]["roles"] == ("train",)
    with pytest.raises(ValueError):
        batch.sequences[0][0, 0] = 1


@pytest.mark.parametrize(
    ("sequences", "ids", "error"),
    [
        ((), (), ValueError),
        ((np.ones((2, 3)),), ("short",), ValueError),
        ((np.ones((3, 3)), np.ones((3, 4))), ("a", "b"), ValueError),
        ((np.ones((3, 3)), np.ones((3, 3))), ("same", "same"), ValueError),
        ((np.array([[1, 2], [3, 4], [5, np.nan]]),), ("nan",), ValueError),
        ((np.ones((3, 3), dtype=bool),), ("bool",), TypeError),
        ((np.array([["a"], ["b"], ["c"]], dtype=object),), ("obj",), TypeError),
    ],
)
def test_sequence_batch_rejects_invalid_authority(
    sequences: tuple[np.ndarray, ...],
    ids: tuple[str, ...],
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        SequenceBatch(sequences=sequences, sequence_ids=ids)


def test_pca_is_train_only_and_preserves_sequence_boundaries() -> None:
    train, evaluation = _batches()
    method = PCARepresentation(n_components=2)
    embedding = method.embed(train, evaluation)

    assert embedding.fit_regime is FitRegime.TRAIN_ONLY_INDUCTIVE
    assert embedding.sequence_ids == evaluation.sequence_ids
    assert [x.shape for x in embedding.sequences] == [(9, 2), (8, 2)]
    np.testing.assert_allclose(method.mean_, np.mean(train.concatenate(), axis=0))
    assert embedding.metadata["target_specific_fit_observations"] == 0

    extreme = SequenceBatch(
        sequences=(np.full((9, 4), 1e9), np.full((8, 4), -1e9)),
        sequence_ids=evaluation.sequence_ids,
    )
    second = PCARepresentation(n_components=2)
    second.embed(train, extreme)
    np.testing.assert_allclose(second.mean_, method.mean_)
    np.testing.assert_allclose(
        np.abs(second.components_),
        np.abs(method.components_),
        atol=1e-12,
    )


def test_autoencoder_is_train_only_deterministic_and_boundary_preserving() -> None:
    train, evaluation = _batches()
    first = AutoencoderRepresentation(
        n_components=2,
        hidden_dim=8,
        epochs=4,
        batch_size=7,
        learning_rate=5e-3,
        seed=17,
    )
    second = AutoencoderRepresentation(
        n_components=2,
        hidden_dim=8,
        epochs=4,
        batch_size=7,
        learning_rate=5e-3,
        seed=17,
    )
    z1 = first.embed(train, evaluation)
    z2 = second.embed(train, evaluation)

    assert z1.fit_regime is FitRegime.TRAIN_ONLY_INDUCTIVE
    assert [x.shape for x in z1.sequences] == [(9, 2), (8, 2)]
    np.testing.assert_allclose(first.mean_, np.mean(train.concatenate(), axis=0), atol=1e-6)
    assert first.training_loss_ is not None and np.isfinite(first.training_loss_)
    for a, b in zip(z1.sequences, z2.sequences, strict=True):
        np.testing.assert_allclose(a, b, atol=1e-6, rtol=1e-6)


def test_tphate_fits_each_evaluation_sequence_independently(monkeypatch: pytest.MonkeyPatch) -> None:
    train, evaluation = _batches()
    instances: list[object] = []

    class FakeTPHATE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.inputs: list[np.ndarray] = []
            instances.append(self)

        def fit_transform(self, x):
            x = np.asarray(x)
            self.inputs.append(np.array(x, copy=True))
            return x[:, : self.kwargs["n_components"]]

    fake_module = SimpleNamespace(TPHATE=FakeTPHATE, __version__="test-1.2.1")
    monkeypatch.setattr(tphate_module, "import_module", lambda _: fake_module)

    embedding = TPHATERepresentation(n_components=2).embed(train, evaluation)

    assert embedding.fit_regime is FitRegime.TRANSDUCTIVE_TARGET_OBSERVED
    assert len(instances) == len(evaluation.sequences)
    for instance, sequence in zip(instances, evaluation.sequences, strict=True):
        assert len(instance.inputs) == 1
        np.testing.assert_array_equal(instance.inputs[0], sequence)
        assert instance.kwargs["n_landmark"] is None
        assert instance.kwargs["random_state"] == 0
    assert embedding.metadata["target_specific_fit_observations"] == evaluation.sample_count
    assert embedding.metadata["coordinate_frame"] == "per_sequence_unaligned_mds"
    assert embedding.metadata["sequence_boundary_policy"] == "fresh_estimator_per_sequence"


def test_tphate_records_requested_and_effective_n_pca(monkeypatch: pytest.MonkeyPatch) -> None:
    train, evaluation = _batches()
    instances: list[object] = []

    class FakeTPHATE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instances.append(self)

        def fit_transform(self, x):
            x = np.asarray(x)
            return x[:, : self.kwargs["n_components"]]

    monkeypatch.setattr(
        tphate_module,
        "import_module",
        lambda _: SimpleNamespace(TPHATE=FakeTPHATE, __version__="test"),
    )
    embedding = TPHATERepresentation(n_components=2, n_pca=2).embed(train, evaluation)

    assert [instance.kwargs["n_pca"] for instance in instances] == [2, 2]
    assert embedding.metadata["requested_n_pca"] == 2
    assert dict(embedding.metadata["effective_n_pca_by_sequence"]) == {
        "eval-a": 2,
        "eval-b": 2,
    }


def test_tphate_records_when_n_pca_is_disabled_per_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, evaluation = _batches()
    instances: list[object] = []

    class FakeTPHATE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            instances.append(self)

        def fit_transform(self, x):
            x = np.asarray(x)
            return x[:, : self.kwargs["n_components"]]

    monkeypatch.setattr(
        tphate_module,
        "import_module",
        lambda _: SimpleNamespace(TPHATE=FakeTPHATE, __version__="test"),
    )
    embedding = TPHATERepresentation(n_components=2, n_pca=4).embed(train, evaluation)

    assert [instance.kwargs["n_pca"] for instance in instances] == [None, None]
    assert embedding.metadata["requested_n_pca"] == 4
    assert dict(embedding.metadata["effective_n_pca_by_sequence"]) == {
        "eval-a": None,
        "eval-b": None,
    }
    assert (
        embedding.metadata["n_pca_policy"]
        == "disable_when_not_strictly_below_sequence_shape"
    )


def test_tphate_missing_dependency_fails_with_license_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, evaluation = _batches()

    def missing(_: str):
        raise ModuleNotFoundError("no tphate")

    monkeypatch.setattr(tphate_module, "import_module", missing)
    with pytest.raises(TPHATEUnavailableError, match="Non-Commercial License"):
        TPHATERepresentation().embed(train, evaluation)


def test_tphate_translates_upstream_autocorrelation_dropoff_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, evaluation = _batches()

    class NoDropoffTPHATE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, x):
            raise IndexError("index 0 is out of bounds")

    monkeypatch.setattr(
        tphate_module,
        "import_module",
        lambda _: SimpleNamespace(TPHATE=NoDropoffTPHATE, __version__="test"),
    )
    with pytest.raises(TPHATEEmbeddingError, match="no negative crossing"):
        TPHATERepresentation().embed(train, evaluation)


def test_external_temporal_ssl_binds_exact_sequence_identity_and_lineage() -> None:
    train, evaluation = _batches()
    original = {
        sequence_id: np.column_stack(
            [np.arange(sequence.shape[0]), np.arange(sequence.shape[0]) ** 2]
        ).astype(float)
        for sequence_id, sequence in zip(
            evaluation.sequence_ids,
            evaluation.sequences,
            strict=True,
        )
    }
    method = PrecomputedTemporalSSLRepresentation(
        original,
        model_id="eegpt",
        model_version="checkpoint-sha256:test",
        pretraining_datasets=("TUEG",),
        pretraining_lineage_status="possible_overlap",
        metadata={"provider": {"name": "external"}},
    )
    original["eval-a"][:] = -100
    embedding = method.embed(train, evaluation)

    assert embedding.fit_regime is FitRegime.EXTERNAL_PRETRAINED
    assert embedding.metadata["model_id"] == "eegpt"
    assert embedding.metadata["pretraining_lineage_status"] == "possible_overlap"
    assert embedding.metadata["target_specific_fit_observations"] == 0
    assert not np.all(embedding.sequences[0] == -100)


def test_external_temporal_ssl_rejects_timepoint_identity_mismatch() -> None:
    train, evaluation = _batches()
    method = PrecomputedTemporalSSLRepresentation(
        {"eval-a": np.ones((8, 2)), "eval-b": np.ones((8, 2))},
        model_id="ssl",
        model_version="1",
    )
    with pytest.raises(ValueError, match="timepoints"):
        method.embed(train, evaluation)


def test_geometry_metrics_are_invariant_to_rigid_coordinate_transforms() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=(30, 6))
    embedding = rng.normal(size=(30, 3))
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    transformed = embedding @ q + np.array([10.0, -4.0, 2.0])

    first = aggregate_geometry_metrics((source,), (embedding,), k=4)
    second = aggregate_geometry_metrics((source,), (transformed,), k=4)
    assert first.keys() == second.keys()
    for key in first:
        if first[key] is None:
            assert second[key] is None
        else:
            assert second[key] == pytest.approx(first[key], abs=1e-12)


def test_benchmark_preserves_unavailable_methods_and_has_no_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, evaluation = _batches()

    def missing(_: str):
        raise ModuleNotFoundError("no tphate")

    monkeypatch.setattr(tphate_module, "import_module", missing)
    result = RepresentationBenchmark(
        [PCARepresentation(2), TPHATERepresentation(2)],
        neighborhood_k=3,
    ).run(train, evaluation)

    outcomes = result.by_method()
    assert outcomes["pca"].status is MethodStatus.OK
    assert outcomes["tphate"].status is MethodStatus.UNAVAILABLE
    assert "Non-Commercial License" in outcomes["tphate"].error_message
    assert result.metadata["ranking_policy"] == "none"
    assert not hasattr(result, "winner")


def test_benchmark_preserves_method_failure_without_dropping_other_results() -> None:
    train, evaluation = _batches()

    class Broken:
        method_id = "broken"
        fit_regime = FitRegime.EXTERNAL_PRETRAINED

        def embed(self, train, evaluation):
            raise RuntimeError("deliberate")

    result = RepresentationBenchmark(
        [PCARepresentation(2), Broken()],
        neighborhood_k=3,
    ).run(train, evaluation)

    outcomes = result.by_method()
    assert outcomes["pca"].status is MethodStatus.OK
    assert outcomes["broken"].status is MethodStatus.FAILED
    assert outcomes["broken"].error_type == "RuntimeError"
    assert outcomes["broken"].error_message == "deliberate"


def test_benchmark_can_score_known_reference_geometry() -> None:
    train, evaluation = _batches()
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

    result = RepresentationBenchmark(
        [PCARepresentation(2)],
        neighborhood_k=3,
    ).run(train, evaluation, reference=reference)

    outcome = result.by_method()["pca"]
    assert outcome.status is MethodStatus.OK
    assert "reference_local_knn_preservation" in outcome.metrics
    assert "reference_pairwise_distance_rank" in outcome.metrics
    assert result.metadata["reference_geometry"] == "provided"


def test_reference_geometry_must_preserve_evaluation_identity() -> None:
    train, evaluation = _batches()
    reference = SequenceBatch(
        sequences=(np.ones((9, 2)), np.ones((8, 2))),
        sequence_ids=("eval-a", "wrong-id"),
    )
    with pytest.raises(ValueError, match="reference sequence identity"):
        RepresentationBenchmark([PCARepresentation(2)]).run(
            train,
            evaluation,
            reference=reference,
        )
