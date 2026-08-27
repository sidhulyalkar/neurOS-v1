from __future__ import annotations

import pytest

from orion.scientific_authority import (
    ObservationConsumption,
    ObservationRole,
    ObservationSetAuthority,
    OperationKind,
    PreprocessingFitAuthority,
    TargetObservationBudget,
    TransformFitKind,
)

SHA = "a" * 64


def _source_observation() -> ObservationSetAuthority:
    return ObservationSetAuthority(
        authority_id="source-history",
        dataset_lineage_sha256=SHA,
        role=ObservationRole.SOURCE_HISTORY,
        observation_ids=("trial-1", "trial-2"),
        domain_id="subject-1/session-1",
    )


def test_data_fitted_consumption_cannot_be_empty():
    with pytest.raises(ValueError, match="at least one observation authority"):
        ObservationConsumption(
            operation_id="fit-standardizer",
            operation=OperationKind.PREPROCESSING_FIT,
            observation_authority_sha256s=(),
            roles=(),
        )


def test_data_fitted_preprocessing_requires_real_nonempty_consumption():
    source = _source_observation()
    consumption = ObservationConsumption.bind(
        operation_id="fit-standardizer",
        operation=OperationKind.PREPROCESSING_FIT,
        observations=(source,),
    )
    authority = PreprocessingFitAuthority(
        transform_id="standardizer",
        fit_kind=TransformFitKind.DATA_FITTED,
        implementation="neuros.standardize",
        implementation_version="2",
        state_sha256="b" * 64,
        consumption=consumption,
    )
    assert authority.consumption is consumption
    assert consumption.roles == (ObservationRole.SOURCE_HISTORY,)


def test_target_budget_cannot_hide_labels_in_per_class_field():
    with pytest.raises(ValueError, match="inconsistent with zero labeled_examples"):
        TargetObservationBudget(
            labeled_examples=0,
            labeled_examples_per_class=5,
        )


def test_per_class_budget_cannot_exceed_total_labeled_examples():
    with pytest.raises(ValueError, match="cannot exceed total"):
        TargetObservationBudget(
            labeled_examples=2,
            labeled_examples_per_class=3,
        )


def test_target_information_predicate_covers_every_budget_dimension():
    assert not TargetObservationBudget().has_target_information
    assert TargetObservationBudget(labeled_examples=2, labeled_examples_per_class=1).has_target_information
    assert TargetObservationBudget(unlabeled_examples=1).has_target_information
    assert TargetObservationBudget(unlabeled_seconds=0.25).has_target_information


def test_observation_role_and_operation_are_typed_authority_not_free_form_strings():
    with pytest.raises(TypeError, match="ObservationRole"):
        ObservationSetAuthority(
            authority_id="bad-role",
            dataset_lineage_sha256=SHA,
            role="source_history",  # type: ignore[arg-type]
            observation_ids=("trial-1",),
            domain_id="subject-1/session-1",
        )

    source = _source_observation()
    with pytest.raises(TypeError, match="OperationKind"):
        ObservationConsumption.bind(
            operation_id="bad-operation",
            operation="preprocessing_fit",  # type: ignore[arg-type]
            observations=(source,),
        )
