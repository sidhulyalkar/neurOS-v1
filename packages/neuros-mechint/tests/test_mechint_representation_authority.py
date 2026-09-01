from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from neuros_mechint.representations import (
    PrecomputedTemporalSSLRepresentation,
    SequenceBatch,
    TPHATERepresentation,
    pairwise_distance_rank_preservation,
    temporal_continuity_ratio,
)
import neuros_mechint.representations.tphate as tphate_module


def test_sequence_batch_rejects_complex_values_before_method_dispatch() -> None:
    values = np.ones((5, 3), dtype=np.complex128) * (1 + 2j)
    with pytest.raises(TypeError, match="real numeric"):
        SequenceBatch(sequences=(values,), sequence_ids=("complex",))


def test_nested_metadata_keys_are_not_coerced_to_strings() -> None:
    with pytest.raises(ValueError, match="metadata keys"):
        SequenceBatch(
            sequences=(np.ones((5, 3)),),
            sequence_ids=("s1",),
            metadata={"nested": {7: "ambiguous"}},
        )


def test_external_ssl_rejects_complex_latent_coordinates() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        PrecomputedTemporalSSLRepresentation(
            {"eval": np.ones((5, 2), dtype=np.complex128)},
            model_id="ssl",
            model_version="1",
        )


def test_public_geometry_metrics_reject_complex_inputs_without_projection() -> None:
    source = np.arange(15, dtype=float).reshape(5, 3)
    complex_embedding = np.ones((5, 2), dtype=np.complex128) * (1 + 1j)
    with pytest.raises(TypeError, match="real numeric"):
        pairwise_distance_rank_preservation(source, complex_embedding)
    with pytest.raises(TypeError, match="real numeric"):
        temporal_continuity_ratio(complex_embedding)


def test_tphate_records_requested_and_effective_pca_per_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_n_pca: list[int | None] = []

    class FakeTPHATE:
        def __init__(self, **kwargs):
            seen_n_pca.append(kwargs["n_pca"])
            self.n_components = kwargs["n_components"]

        def fit_transform(self, x):
            return np.asarray(x)[:, : self.n_components]

    monkeypatch.setattr(
        tphate_module,
        "import_module",
        lambda _: SimpleNamespace(TPHATE=FakeTPHATE, __version__="test"),
    )
    train = SequenceBatch(
        sequences=(np.arange(60, dtype=float).reshape(12, 5),),
        sequence_ids=("train",),
    )
    evaluation = SequenceBatch(
        sequences=(
            np.arange(40, dtype=float).reshape(8, 5),
            np.arange(72, dtype=float).reshape(12, 6)[:, :5],
        ),
        sequence_ids=("short", "long"),
    )
    embedding = TPHATERepresentation(n_components=2, n_pca=4).embed(train, evaluation)

    assert seen_n_pca == [4, 4]
    assert embedding.metadata["requested_n_pca"] == 4
    assert embedding.metadata["effective_n_pca_by_sequence"] == {
        "short": 4,
        "long": 4,
    }


def test_tphate_disables_invalid_pca_and_records_the_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen_n_pca: list[int | None] = []

    class FakeTPHATE:
        def __init__(self, **kwargs):
            seen_n_pca.append(kwargs["n_pca"])
            self.n_components = kwargs["n_components"]

        def fit_transform(self, x):
            return np.asarray(x)[:, : self.n_components]

    monkeypatch.setattr(
        tphate_module,
        "import_module",
        lambda _: SimpleNamespace(TPHATE=FakeTPHATE, __version__="test"),
    )
    train = SequenceBatch(
        sequences=(np.arange(60, dtype=float).reshape(12, 5),),
        sequence_ids=("train",),
    )
    evaluation = SequenceBatch(
        sequences=(np.arange(40, dtype=float).reshape(8, 5),),
        sequence_ids=("short",),
    )
    embedding = TPHATERepresentation(n_components=2, n_pca=5).embed(train, evaluation)

    assert seen_n_pca == [None]
    assert embedding.metadata["requested_n_pca"] == 5
    assert embedding.metadata["effective_n_pca_by_sequence"] == {"short": None}
    assert (
        embedding.metadata["n_pca_policy"]
        == "disable_when_not_strictly_below_sequence_shape"
    )
