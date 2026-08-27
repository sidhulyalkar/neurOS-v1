from __future__ import annotations

import pytest

from orion.scientific_authority import (
    DatasetLineage,
    FailureAggregationPolicy,
    IdentityAvailability,
    IdentitySet,
    LineageCompleteness,
    MetricDirection,
    MetricSpec,
    ModelLineage,
    OverlapStatus,
    PretrainingOverlapAudit,
    ProbabilityRequirement,
    RepeatedMeasuresAuthority,
    ScientificStudyAuthority,
    audit_pretraining_overlap,
)


def _participant(*ids: str) -> IdentitySet:
    return IdentitySet(
        level="participant",
        availability=IdentityAvailability.AVAILABLE,
        identifiers=ids,
    )


def _metric() -> MetricSpec:
    return MetricSpec(
        metric_id="balanced_accuracy",
        version="fixture-v1",
        direction=MetricDirection.HIGHER_IS_BETTER,
        averaging="macro recall",
        class_semantics="two declared classes with equal weighting",
        probability_requirement=ProbabilityRequirement.NONE,
        estimator="fixture",
        estimator_version="1",
        aggregation_unit="participant-session case",
        failure_policy=FailureAggregationPolicy.PRESERVE,
        uncertainty_method="participant-cluster bootstrap",
        primary=True,
    )


def _repeated() -> RepeatedMeasuresAuthority:
    return RepeatedMeasuresAuthority(
        hierarchy=("participant", "session", "trial"),
        independent_unit="participant",
        case_unit="participant-session",
        cluster_units=("participant",),
        inference_method="participant-cluster bootstrap",
    )


def test_dataset_lineage_detaches_from_caller_owned_identity_list():
    identity_sets = [_participant("p-1")]
    dataset = DatasetLineage(
        dataset_id="evaluation",
        upstream_source="fixture",
        identity_sets=identity_sets,  # type: ignore[arg-type]
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    original_sha = dataset.lineage_sha256

    identity_sets.clear()
    identity_sets.append(_participant("p-999"))

    assert isinstance(dataset.identity_sets, tuple)
    assert dataset.identity_sets[0].identifiers == ("p-1",)
    assert dataset.lineage_sha256 == original_sha


def test_model_lineage_detaches_from_caller_owned_identity_list():
    identity_sets = [_participant("p-1")]
    model = ModelLineage(
        model_id="pretrained-model",
        upstream_source="fixture",
        pretraining_identity_sets=identity_sets,  # type: ignore[arg-type]
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    original_sha = model.lineage_sha256

    identity_sets.clear()
    assert isinstance(model.pretraining_identity_sets, tuple)
    assert model.pretraining_identity_sets[0].identifiers == ("p-1",)
    assert model.lineage_sha256 == original_sha


def test_participant_overlap_declared_on_evaluation_ancestor_is_detected():
    parent = DatasetLineage(
        dataset_id="parent-corpus",
        upstream_source="parent",
        identity_sets=(_participant("p-7"),),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    child = DatasetLineage(
        dataset_id="derived-evaluation",
        upstream_source="derived",
        parent_dataset_ids=("parent-corpus",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="other-named-pretraining",
        upstream_source="checkpoint",
        pretraining_dataset_ids=("different-corpus-name",),
        pretraining_identity_sets=(_participant("p-7"),),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )

    audit = audit_pretraining_overlap(
        model,
        child,
        known_datasets={"parent-corpus": parent},
    )
    assert audit.status is OverlapStatus.OVERLAP_DETECTED
    assert audit.matched_dataset_ids == ()
    assert audit.matched_identity_levels == ("participant",)


def test_unresolved_ancestor_prevents_false_disjoint_verification():
    child = DatasetLineage(
        dataset_id="derived-evaluation",
        upstream_source="derived",
        parent_dataset_ids=("missing-parent",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="pretrained-model",
        upstream_source="checkpoint",
        pretraining_dataset_ids=("other-corpus",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(model, child)
    assert audit.status is OverlapStatus.POSSIBLE_OVERLAP
    assert audit.unresolved_ancestor_ids == ("missing-parent",)


def test_known_dataset_mapping_cannot_relabel_a_different_lineage_object():
    parent = DatasetLineage(
        dataset_id="actual-parent",
        upstream_source="parent",
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    child = DatasetLineage(
        dataset_id="child",
        upstream_source="child",
        parent_dataset_ids=("declared-parent",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="model",
        upstream_source="checkpoint",
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    with pytest.raises(ValueError, match="key must match"):
        audit_pretraining_overlap(
            model,
            child,
            known_datasets={"declared-parent": parent},
        )


def test_study_recomputes_overlap_and_rejects_forged_disjoint_verdict():
    dataset = DatasetLineage(
        dataset_id="shared-domain",
        upstream_source="evaluation",
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="pretrained-model",
        upstream_source="checkpoint",
        pretraining_dataset_ids=("shared-domain",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    forged = PretrainingOverlapAudit(
        status=OverlapStatus.DISJOINT_VERIFIED,
        model_id=model.model_id,
        evaluation_dataset_id=dataset.dataset_id,
        model_lineage_sha256=model.lineage_sha256,
        evaluation_dataset_lineage_sha256=dataset.lineage_sha256,
        reason="caller-supplied clean verdict",
    )

    with pytest.raises(ValueError, match="independently recomputed study lineage"):
        ScientificStudyAuthority(
            study_id="forged-overlap-verdict",
            protocol_sha256="d" * 64,
            datasets=(dataset,),
            models=(model,),
            observations=(),
            preprocessing=(),
            metrics=(_metric(),),
            repeated_measures=_repeated(),
            overlap_audits=(forged,),
        )


def test_lineage_completeness_is_typed_authority_not_free_form_string():
    with pytest.raises(TypeError, match="LineageCompleteness"):
        DatasetLineage(
            dataset_id="bad",
            upstream_source="fixture",
            lineage_completeness="complete",  # type: ignore[arg-type]
        )
    with pytest.raises(TypeError, match="LineageCompleteness"):
        ModelLineage(
            model_id="bad-model",
            upstream_source="fixture",
            pretraining_lineage_completeness="complete",  # type: ignore[arg-type]
        )
