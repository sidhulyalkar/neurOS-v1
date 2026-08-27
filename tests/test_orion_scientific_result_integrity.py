from __future__ import annotations

import pytest

from orion.scientific_authority import (
    CaseOutcome,
    CaseStatus,
    FailureAggregationPolicy,
    FailurePreservingResultSet,
    MetricDirection,
    MetricSpec,
    ProbabilityRequirement,
)


def _metric(**overrides):
    kwargs = {
        "metric_id": "balanced_accuracy",
        "version": "sklearn-v1",
        "direction": MetricDirection.HIGHER_IS_BETTER,
        "averaging": "macro recall across declared classes",
        "class_semantics": "left vs right; equal class weighting",
        "probability_requirement": ProbabilityRequirement.NONE,
        "estimator": "sklearn.metrics.balanced_accuracy_score",
        "estimator_version": "1.x",
        "aggregation_unit": "participant-session case",
        "failure_policy": FailureAggregationPolicy.PRESERVE,
        "uncertainty_method": "participant-cluster bootstrap",
        "primary": True,
    }
    kwargs.update(overrides)
    return MetricSpec(**kwargs)


@pytest.mark.parametrize(
    "status",
    [CaseStatus.FAILED, CaseStatus.SKIPPED, CaseStatus.OOM, CaseStatus.NONCONVERGED, CaseStatus.UNAVAILABLE],
)
def test_non_success_rows_cannot_carry_scientific_metric_values(status):
    with pytest.raises(ValueError, match="cannot carry scientific metric values"):
        CaseOutcome(
            case_id="subject-1/session-2",
            method_id="eegnet",
            status=status,
            metrics={"balanced_accuracy": 0.71},
            reason="incomplete case",
        )


def test_failed_row_can_preserve_partial_diagnostics_in_metadata_without_becoming_score():
    row = CaseOutcome(
        case_id="subject-1/session-2",
        method_id="eegnet",
        status=CaseStatus.NONCONVERGED,
        reason="optimizer exhausted declared iterations",
        metadata={"last_loss": 0.42, "iterations": 100},
    )
    assert dict(row.metrics) == {}
    assert row.metadata["last_loss"] == 0.42
    assert row.metadata["iterations"] == 100


def test_result_set_detaches_from_caller_owned_row_list_and_keeps_stable_sha():
    original_rows = [
        CaseOutcome(
            case_id="case-1",
            method_id="eegnet",
            status=CaseStatus.OK,
            metrics={"balanced_accuracy": 0.75},
        )
    ]
    result = FailurePreservingResultSet(
        declared_case_ids=("case-1",),
        method_ids=("eegnet",),
        rows=original_rows,  # type: ignore[arg-type]
    )
    original_sha = result.result_sha256

    original_rows.clear()
    original_rows.append(
        CaseOutcome(
            case_id="case-1",
            method_id="eegnet",
            status=CaseStatus.OK,
            metrics={"balanced_accuracy": 0.01},
        )
    )

    assert isinstance(result.rows, tuple)
    assert len(result.rows) == 1
    assert result.rows[0].metrics["balanced_accuracy"] == 0.75
    assert result.result_sha256 == original_sha


def test_result_set_rejects_non_case_outcome_rows():
    with pytest.raises(TypeError, match="CaseOutcome"):
        FailurePreservingResultSet(
            declared_case_ids=("case-1",),
            method_ids=("eegnet",),
            rows=({"case_id": "case-1"},),  # type: ignore[arg-type]
        )


def test_metric_spec_requires_enum_authority_not_free_form_strings():
    with pytest.raises(TypeError, match="MetricDirection"):
        _metric(direction="higher_is_better")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ProbabilityRequirement"):
        _metric(probability_requirement="none")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="FailureAggregationPolicy"):
        _metric(failure_policy="preserve")  # type: ignore[arg-type]


def test_target_metric_rejects_boolean_target_value():
    with pytest.raises(ValueError, match="finite numeric target_value"):
        _metric(
            metric_id="distance-to-target",
            direction=MetricDirection.TARGET_IS_BEST,
            target_value=True,
        )
