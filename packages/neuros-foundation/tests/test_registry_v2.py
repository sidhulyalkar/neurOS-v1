"""Tests for the v2.1 registry/probe architecture."""

from __future__ import annotations

import numpy as np
import pytest

from neuros.foundation_models import (
    DEFAULT_REGISTRY,
    AdapterUnavailableError,
    EvaluationProtocol,
    FoundationEmbeddingDecoder,
    benchmark_embeddings,
    effective_rank,
    invariance_score,
    linear_cka,
    linear_probe,
    representation_report,
)


def test_catalog_spans_modern_modalities_and_industry() -> None:
    ids = {card.id for card in DEFAULT_REGISTRY.cards()}
    assert {"zuna-1.1", "reve-base", "neuript", "ndt3", "brainlm", "neuros-neurofmx"} <= ids
    assert DEFAULT_REGISTRY.get_card("emotiv-axon").access.value == "closed"
    assert DEFAULT_REGISTRY.get_card("manas-1").integration.value == "catalog"


def test_registry_filters_eeg_and_recent_years() -> None:
    cards = DEFAULT_REGISTRY.filter(modality="eeg", min_year=2025)
    ids = {card.id for card in cards}
    assert "zuna-1.1" in ids
    assert "reve-base" in ids
    assert "ndt3" not in ids


def test_catalog_only_model_fails_closed() -> None:
    status = DEFAULT_REGISTRY.availability("emotiv-axon")
    assert not status.available
    with pytest.raises(AdapterUnavailableError):
        DEFAULT_REGISTRY.adapter("emotiv-axon")


def test_model_comparison_exposes_runnable_separately() -> None:
    rows = DEFAULT_REGISTRY.compare(
        ["zuna-1.1", "reve-base"],
        fields=("id", "input_geometry", "access"),
    )
    assert rows[0]["id"] == "zuna-1.1"
    assert "runnable" in rows[0]
    assert rows[1]["id"] == "reve-base"


def test_representation_probes_are_deterministic() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(64, 12))
    rotated = x @ np.linalg.qr(rng.normal(size=(12, 12)))[0]

    report = representation_report(x)
    assert report["finite"] is True
    assert 1 <= effective_rank(x) <= 12
    assert linear_cka(x, x) == pytest.approx(1.0)
    assert linear_cka(x, rotated) == pytest.approx(1.0)
    assert invariance_score(x, x) == pytest.approx(1.0)


def test_linear_probe_finds_simple_signal() -> None:
    rng = np.random.default_rng(2)
    x_train = rng.normal(size=(300, 16))
    x_test = rng.normal(size=(120, 16))
    y_train = (x_train[:, 0] - 0.25 * x_train[:, 1] > 0).astype(int)
    y_test = (x_test[:, 0] - 0.25 * x_test[:, 1] > 0).astype(int)

    result = linear_probe(x_train, y_train, x_test, y_test, task="classification")
    assert result["score"] >= 0.9


def test_benchmark_records_protocol_and_similarity() -> None:
    rng = np.random.default_rng(3)
    train = rng.normal(size=(100, 8))
    test = rng.normal(size=(50, 8))
    y_train = (train[:, 0] > 0).astype(int)
    y_test = (test[:, 0] > 0).astype(int)
    protocol = EvaluationProtocol(name="cross-subject-test")

    report = benchmark_embeddings(
        train_embeddings={"a": train, "b": train[:, ::-1]},
        test_embeddings={"a": test, "b": test[:, ::-1]},
        train_targets=y_train,
        test_targets=y_test,
        protocol=protocol,
    )
    payload = report.to_dict()
    assert payload["protocol"]["fingerprint"] == protocol.fingerprint
    assert len(payload["model_results"]) == 2
    assert payload["pairwise_similarity"][0]["linear_cka"] == pytest.approx(1.0)


def test_sample_level_protocol_requires_explicit_acknowledgement() -> None:
    with pytest.raises(ValueError, match="sample-level splits"):
        EvaluationProtocol(name="unsafe", split_unit="sample")

    protocol = EvaluationProtocol(
        name="intentional-sample-split",
        split_unit="sample",
        notes=("Sample split is intentional for this within-recording diagnostic.",),
    )
    assert protocol.split_unit == "sample"


def test_foundation_embedding_decoder_implements_neuros_model_contract() -> None:
    rng = np.random.default_rng(4)
    x = rng.normal(size=(80, 4))
    y = (x[:, 0] > 0).astype(int)

    model = FoundationEmbeddingDecoder(
        encoder=lambda values: np.concatenate([values, values**2], axis=1),
        model_id="synthetic",
    )
    model.train(x, y)
    prediction = model.predict(x[:5])
    assert prediction.shape == (5,)
    output = model.infer(x[:1])
    assert output.prediction in {0, 1}
