from __future__ import annotations

import pytest

from .demo import run_demo, synthetic_data
from .method import ExternalFactory


def test_learned_state_is_deterministic() -> None:
    data = synthetic_data()
    first = ExternalFactory().create()
    second = ExternalFactory().create()

    first.fit(data.X, data.y)
    second.fit(data.X, data.y)

    assert first.learned_state().state_sha256 == second.learned_state().state_sha256


def test_learned_state_changes_with_different_data() -> None:
    data = synthetic_data()
    first = ExternalFactory().create()
    second = ExternalFactory().create()

    first.fit(data.X, data.y)
    second.fit(data.X[:30], data.y[:30])

    assert first.learned_state().state_sha256 != second.learned_state().state_sha256


def test_learned_state_requires_fitted_estimator() -> None:
    with pytest.raises(RuntimeError, match="requires fit"):
        ExternalFactory().create().learned_state()


def test_example_runs_through_public_nsq_import_surface() -> None:
    factory = ExternalFactory()
    assert factory.create() is not factory.create()

    payload = run_demo()

    assert payload["artifact_kind"] == "synthetic_external_method_nsq_referee_demo"
    assert len(payload["case_result_sha256"]) == 64
    assert all(row["status"] == "success" for row in payload["rows"])
    assert all(row["learned_state_addressable"] for row in payload["rows"])
    assert len({row["evaluation_indices_sha256"] for row in payload["rows"]}) == 1
    assert payload["claim_boundary"]["production_referee_exercised"] is True
    assert payload["claim_boundary"]["numerical_result_interpretable"] is False
    assert payload["claim_boundary"]["real_neural_efficacy_claim_permitted"] is False
    assert payload["claim_boundary"]["kumar2024_comparison_changed"] is False
