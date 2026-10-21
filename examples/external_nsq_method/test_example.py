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


def test_multiple_calibration_budgets_execute_with_distinct_run_contracts() -> None:
    payload = run_demo()
    rows = payload["rows"]

    # Multiple calibration budgets were executed, each exactly once, in order.
    budgets = [row["calibration_per_class"] for row in rows]
    assert len(budgets) >= 2
    assert budgets == sorted(set(budgets))
    assert budgets[0] == 0

    # Each budget produced its own run-contract identity.
    run_contracts = [row["run_contract_sha256"] for row in rows]
    assert len(set(run_contracts)) == len(rows)

    # Larger budgets actually consumed more labeled target examples.
    labeled = [row["labeled_target_examples"] for row in rows]
    assert labeled == sorted(labeled)
    assert labeled[0] == 0 and labeled[-1] > 0

    # The untouched final-evaluation partition is identical across budgets.
    assert len({row["evaluation_indices_sha256"] for row in rows}) == 1
    assert len({row["evaluation_samples"] for row in rows}) == 1

    # Shared protocol/case/method identity across all budgets.
    for key in ("protocol_sha256", "case_authority_sha256", "method_spec_sha256"):
        assert len({row[key] for row in rows}) == 1
