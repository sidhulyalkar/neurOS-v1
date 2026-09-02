from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import neuros_mechint.representations.tphate as tphate_module
from neuros_mechint.representations import (
    FitRegime,
    MethodStatus,
    PCARepresentation,
    PrecomputedTemporalSSLRepresentation,
    SequenceBatch,
    TPHATERepresentation,
)
from neuros_mechint.representations.sequence_authority import (
    SequenceMethodOutcome,
    SequenceRepresentationBenchmarkResult,
    run_sequencewise_representation_benchmark,
)


def _batches() -> tuple[SequenceBatch, SequenceBatch, SequenceBatch]:
    rng = np.random.default_rng(31)
    train = SequenceBatch(
        sequences=(rng.normal(size=(16, 4)),),
        sequence_ids=("train",),
    )
    evaluation = SequenceBatch(
        sequences=(rng.normal(size=(11, 4)), rng.normal(size=(9, 4))),
        sequence_ids=("eval-a", "eval-b"),
    )
    reference = SequenceBatch(
        sequences=(rng.normal(size=(11, 3)), rng.normal(size=(9, 3))),
        sequence_ids=evaluation.sequence_ids,
    )
    return train, evaluation, reference


def test_native_tphate_sequence_failure_preserves_successful_sibling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, evaluation, reference = _batches()
    fit_calls: list[str] = []

    class PartiallyFailingTPHATE:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit_transform(self, x):
            fit_calls.append("call")
            if len(fit_calls) == 2:
                raise IndexError("no ACF crossing")
            array = np.asarray(x)
            return array[:, : self.kwargs["n_components"]]

    monkeypatch.setattr(
        tphate_module,
        "import_module",
        lambda _: SimpleNamespace(TPHATE=PartiallyFailingTPHATE, __version__="test"),
    )

    result = run_sequencewise_representation_benchmark(
        [PCARepresentation(2), TPHATERepresentation(2)],
        train,
        evaluation,
        reference=reference,
        neighborhood_k=3,
    )
    by_pair = result.by_pair()
    assert len(result.outcomes) == 4
    assert by_pair[("pca", "eval-a")].status is MethodStatus.OK
    assert by_pair[("pca", "eval-b")].status is MethodStatus.OK
    assert by_pair[("tphate", "eval-a")].status is MethodStatus.OK
    assert by_pair[("tphate", "eval-b")].status is MethodStatus.FAILED
    assert by_pair[("tphate", "eval-b")].error_type == "TPHATEEmbeddingError"
    assert by_pair[("tphate", "eval-b")].metrics == {}
    assert "reference_pairwise_distance_rank" in by_pair[("tphate", "eval-a")].metrics
    assert (
        by_pair[("tphate", "eval-a")].metadata["execution_scope"]
        == "native_per_sequence"
    )
    assert result.metadata["ranking_policy"] == "none"


def test_external_sequence_binding_preserves_present_sibling_when_one_is_missing() -> None:
    train, evaluation, _ = _batches()
    embeddings = {
        "eval-a": np.column_stack(
            [np.arange(11, dtype=float), np.arange(11, dtype=float) ** 2]
        )
    }
    method = PrecomputedTemporalSSLRepresentation(
        embeddings,
        model_id="fixture-ssl",
        model_version="sha256:fixture",
        pretraining_lineage_status="disjoint_verified",
    )
    result = run_sequencewise_representation_benchmark([method], train, evaluation)
    by_pair = result.by_pair()
    assert by_pair[("temporal_ssl", "eval-a")].status is MethodStatus.OK
    assert by_pair[("temporal_ssl", "eval-b")].status is MethodStatus.FAILED
    assert by_pair[("temporal_ssl", "eval-b")].error_type == "KeyError"
    assert by_pair[("temporal_ssl", "eval-b")].metrics == {}


def test_missing_tphate_dependency_is_preserved_for_every_declared_sequence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    train, evaluation, _ = _batches()

    def missing(_: str):
        raise ModuleNotFoundError("no tphate")

    monkeypatch.setattr(tphate_module, "import_module", missing)
    result = run_sequencewise_representation_benchmark(
        [TPHATERepresentation(2)],
        train,
        evaluation,
    )
    outcomes = result.by_pair()
    assert outcomes[("tphate", "eval-a")].status is MethodStatus.UNAVAILABLE
    assert outcomes[("tphate", "eval-b")].status is MethodStatus.UNAVAILABLE
    assert all(row.metrics == {} for row in outcomes.values())


def test_batch_only_method_failure_is_conservatively_bound_to_all_sequences() -> None:
    train, evaluation, _ = _batches()

    class BrokenBatchMethod:
        method_id = "broken"
        fit_regime = FitRegime.TRAIN_ONLY_INDUCTIVE

        def embed(self, train, evaluation):
            raise RuntimeError("deliberate batch failure")

    result = run_sequencewise_representation_benchmark(
        [BrokenBatchMethod()],
        train,
        evaluation,
    )
    outcomes = result.by_pair()
    assert set(outcomes) == {("broken", "eval-a"), ("broken", "eval-b")}
    for row in outcomes.values():
        assert row.status is MethodStatus.FAILED
        assert row.error_type == "RuntimeError"
        assert row.metadata["runtime_attribution"] == "shared_not_sequence_additive"
        assert row.metrics == {}


def test_sequence_result_rejects_missing_method_sequence_pair() -> None:
    train, evaluation, _ = _batches()
    complete = run_sequencewise_representation_benchmark(
        [PCARepresentation(2)],
        train,
        evaluation,
    )
    with pytest.raises(ValueError, match="Cartesian complete"):
        SequenceRepresentationBenchmarkResult(
            method_ids=complete.method_ids,
            train_sequence_ids=complete.train_sequence_ids,
            evaluation_sequence_ids=complete.evaluation_sequence_ids,
            outcomes=complete.outcomes[:-1],
        )


def test_failed_sequence_outcome_cannot_carry_scientific_metrics() -> None:
    with pytest.raises(ValueError, match="cannot carry scientific metrics"):
        SequenceMethodOutcome(
            method_id="broken",
            sequence_id="eval-a",
            fit_regime=FitRegime.EXTERNAL_PRETRAINED,
            status=MethodStatus.FAILED,
            metrics={"reference_pairwise_distance_rank": 0.99},
            error_type="RuntimeError",
            error_message="failed",
        )


def test_embed_sequence_interfaces_require_exactly_one_sequence() -> None:
    train, evaluation, _ = _batches()
    with pytest.raises(ValueError, match="exactly one evaluation sequence"):
        TPHATERepresentation(2).embed_sequence(train, evaluation)

    embeddings = {
        sequence_id: np.ones((sequence.shape[0], 2))
        for sequence_id, sequence in zip(
            evaluation.sequence_ids,
            evaluation.sequences,
            strict=True,
        )
    }
    external = PrecomputedTemporalSSLRepresentation(
        embeddings,
        model_id="fixture",
        model_version="1",
    )
    with pytest.raises(ValueError, match="exactly one evaluation sequence"):
        external.embed_sequence(train, evaluation)
