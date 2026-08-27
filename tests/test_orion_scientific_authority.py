from __future__ import annotations

from types import MappingProxyType

import pytest

from orion.scientific_authority import (
    CaseOutcome,
    CaseStatus,
    ClaimQualification,
    DatasetLineage,
    EvidenceClaim,
    EvidenceDomain,
    FailureAggregationPolicy,
    FailurePreservingResultSet,
    IdentityAvailability,
    IdentitySet,
    LineageCompleteness,
    MetricDirection,
    MetricSpec,
    ModelLineage,
    ObservationConsumption,
    ObservationRole,
    ObservationSetAuthority,
    OperationKind,
    OverlapStatus,
    PreprocessingFitAuthority,
    ProbabilityRequirement,
    RepeatedMeasuresAuthority,
    ScientificStudyAuthority,
    TargetObservationBudget,
    TransformFitKind,
    audit_pretraining_overlap,
    bind_longitudinal_case_authority,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64


def _kumar() -> DatasetLineage:
    return DatasetLineage(
        dataset_id="kumar2024",
        upstream_source="MOABB:Kumar2024",
        version="1.5",
        revision="fixture",
        content_sha256=SHA_A,
        identity_sets=(
            IdentitySet(
                level="participant",
                availability=IdentityAvailability.AVAILABLE,
                identifiers=("1", "10"),
            ),
            IdentitySet(
                level="site",
                availability=IdentityAvailability.UNAVAILABLE,
                unavailable_reason="upstream public loader does not expose site IDs",
            ),
        ),
        sampling_assumptions={"sampling_rate_hz": 512.0, "channels": 22},
        lineage_completeness=LineageCompleteness.COMPLETE,
    )


def _balanced_accuracy() -> MetricSpec:
    return MetricSpec(
        metric_id="balanced_accuracy",
        version="sklearn-v1",
        direction=MetricDirection.HIGHER_IS_BETTER,
        averaging="macro recall across declared classes",
        class_semantics="left_hand vs right_hand; equal class weighting",
        probability_requirement=ProbabilityRequirement.NONE,
        estimator="sklearn.metrics.balanced_accuracy_score",
        estimator_version="1.x",
        aggregation_unit="participant-session case",
        failure_policy=FailureAggregationPolicy.PRESERVE,
        uncertainty_method="participant-cluster bootstrap stratified by GR/PAR",
        primary=True,
    )


def _repeated() -> RepeatedMeasuresAuthority:
    return RepeatedMeasuresAuthority(
        hierarchy=("participant", "session", "run", "trial"),
        independent_unit="participant",
        case_unit="participant-session",
        cluster_units=("participant",),
        inference_method="participant-cluster bootstrap",
        strata=("original_protocol", "held_out_session"),
    )


def _observation(role: ObservationRole, suffix: str, ids=("0", "1")) -> ObservationSetAuthority:
    return ObservationSetAuthority(
        authority_id=f"obs-{suffix}",
        dataset_lineage_sha256=_kumar().lineage_sha256,
        role=role,
        observation_ids=tuple(ids),
        domain_id="kumar2024:subject-1:session-1",
    )


def test_dataset_lineage_has_full_identity_and_deterministic_display_fingerprint():
    first = DatasetLineage(
        dataset_id="example",
        upstream_source="upstream",
        content_sha256=SHA_A,
        sampling_assumptions={"b": 2, "a": [1, 2]},
        metadata={"z": {"x": 1}, "a": True},
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    second = DatasetLineage(
        dataset_id="example",
        upstream_source="upstream",
        content_sha256=SHA_A,
        sampling_assumptions={"a": [1, 2], "b": 2},
        metadata={"a": True, "z": {"x": 1}},
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    assert first.lineage_sha256 == second.lineage_sha256
    assert len(first.lineage_sha256) == 64
    assert first.display_fingerprint == first.lineage_sha256[:16]
    assert isinstance(first.metadata, MappingProxyType)
    with pytest.raises(TypeError):
        first.metadata["new"] = "mutation"  # type: ignore[index]


def test_lineage_rejects_unordered_or_nonfinite_provenance():
    with pytest.raises(TypeError, match="unordered sets"):
        DatasetLineage(
            dataset_id="bad",
            upstream_source="upstream",
            metadata={"ids": {"a", "b"}},
        )
    with pytest.raises(ValueError, match="NaN or infinity"):
        DatasetLineage(
            dataset_id="bad",
            upstream_source="upstream",
            sampling_assumptions={"rate": float("nan")},
        )


def test_bendr_tueg_overlap_is_machine_visible_for_tuab_and_tuev():
    tueg = DatasetLineage(
        dataset_id="tueg",
        upstream_source="Temple University EEG Corpus",
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    tuab = DatasetLineage(
        dataset_id="tuab",
        upstream_source="TUAB",
        parent_dataset_ids=("tueg",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    tuev = DatasetLineage(
        dataset_id="tuev",
        upstream_source="TUEV",
        parent_dataset_ids=("tueg",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    bendr = ModelLineage(
        model_id="bendr",
        upstream_source="BENDR pretrained checkpoint",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=("tueg",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )

    for evaluation in (tuab, tuev):
        audit = audit_pretraining_overlap(
            bendr,
            evaluation,
            known_datasets={"tueg": tueg},
        )
        assert audit.status is OverlapStatus.OVERLAP_DETECTED
        assert audit.matched_dataset_ids == ("tueg",)
        assert len(audit.audit_sha256) == 64


def test_complete_kumar_domain_is_verified_disjoint_from_tueg_pretraining():
    bendr = ModelLineage(
        model_id="bendr",
        upstream_source="BENDR",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=("tueg",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(bendr, _kumar())
    assert audit.status is OverlapStatus.DISJOINT_VERIFIED
    assert audit.matched_dataset_ids == ()


def test_unknown_or_partial_lineage_never_becomes_disjoint():
    unknown = ModelLineage(
        model_id="mystery",
        upstream_source="unknown checkpoint",
        pretraining_lineage_completeness=LineageCompleteness.UNKNOWN,
    )
    partial = ModelLineage(
        model_id="partial",
        upstream_source="partial checkpoint",
        pretraining_dataset_ids=("known-source",),
        pretraining_lineage_completeness=LineageCompleteness.PARTIAL,
    )
    assert audit_pretraining_overlap(unknown, _kumar()).status is OverlapStatus.UNKNOWN_LINEAGE
    assert audit_pretraining_overlap(partial, _kumar()).status is OverlapStatus.POSSIBLE_OVERLAP


def test_preprocessing_cannot_consume_final_assessment_or_qualification_rows():
    final = _observation(ObservationRole.FINAL_ASSESSMENT, "final")
    qualification = _observation(ObservationRole.QUALIFICATION, "qualification")
    with pytest.raises(ValueError, match="preprocessing_fit cannot consume"):
        ObservationConsumption.bind(
            operation_id="normalize",
            operation=OperationKind.PREPROCESSING_FIT,
            observations=(final,),
        )
    with pytest.raises(ValueError, match="model_training cannot consume"):
        ObservationConsumption.bind(
            operation_id="train",
            operation=OperationKind.MODEL_TRAINING,
            observations=(qualification,),
        )


def test_model_selection_and_final_assessment_have_distinct_authority():
    qualification = _observation(ObservationRole.QUALIFICATION, "qualification")
    final = _observation(ObservationRole.FINAL_ASSESSMENT, "final")
    selection = ObservationConsumption.bind(
        operation_id="select-state",
        operation=OperationKind.MODEL_SELECTION,
        observations=(qualification,),
    )
    assessment = ObservationConsumption.bind(
        operation_id="final-score",
        operation=OperationKind.FINAL_ASSESSMENT,
        observations=(final,),
    )
    assert selection.roles == (ObservationRole.QUALIFICATION,)
    assert assessment.roles == (ObservationRole.FINAL_ASSESSMENT,)
    with pytest.raises(ValueError, match="final_assessment cannot consume"):
        ObservationConsumption.bind(
            operation_id="bad-final-score",
            operation=OperationKind.FINAL_ASSESSMENT,
            observations=(qualification,),
        )


def test_data_fitted_preprocessing_requires_authorized_fit_consumption():
    source = _observation(ObservationRole.SOURCE_HISTORY, "source")
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
        state_sha256=SHA_C,
        consumption=consumption,
    )
    assert authority.consumption is consumption
    assert len(authority.authority_sha256) == 64

    with pytest.raises(ValueError, match="require observation consumption"):
        PreprocessingFitAuthority(
            transform_id="bad",
            fit_kind=TransformFitKind.DATA_FITTED,
            implementation="x",
            implementation_version="1",
            state_sha256=SHA_C,
        )
    with pytest.raises(ValueError, match="cannot claim data-fitted consumption"):
        PreprocessingFitAuthority(
            transform_id="fixed",
            fit_kind=TransformFitKind.PREDECLARED_FIXED,
            implementation="fixed-bandpass",
            implementation_version="1",
            state_sha256=SHA_C,
            consumption=consumption,
        )


def test_labeled_and_unlabeled_target_budgets_are_separate():
    budget = TargetObservationBudget(
        labeled_examples=10,
        unlabeled_examples=250,
        unlabeled_seconds=1.0,
    )
    assert budget.to_dict() == {
        "labeled_examples": 10,
        "unlabeled_examples": 250,
        "unlabeled_seconds": 1.0,
    }
    with pytest.raises(ValueError, match="labeled_examples"):
        TargetObservationBudget(labeled_examples=-1)


def test_metric_spec_is_not_just_a_metric_name():
    metric = _balanced_accuracy()
    payload = metric.to_dict()
    assert payload["direction"] == "higher_is_better"
    assert payload["aggregation_unit"] == "participant-session case"
    assert payload["failure_policy"] == "preserve"
    assert len(payload["metric_sha256"]) == 64
    assert len(payload["display_fingerprint"]) == 16

    with pytest.raises(ValueError, match="require a finite target_value"):
        MetricSpec(
            metric_id="distance-to-target",
            version="1",
            direction=MetricDirection.TARGET_IS_BEST,
            averaging="none",
            class_semantics="continuous",
            probability_requirement=ProbabilityRequirement.NONE,
            estimator="fixture",
            estimator_version="1",
            aggregation_unit="participant",
            failure_policy=FailureAggregationPolicy.PRESERVE,
            uncertainty_method="bootstrap",
        )


def test_repeated_measures_authority_cannot_promote_session_as_hidden_independent_unit():
    repeated = _repeated()
    assert repeated.independent_unit == "participant"
    assert repeated.cluster_units == ("participant",)
    with pytest.raises(ValueError, match="independent_unit must be present"):
        RepeatedMeasuresAuthority(
            hierarchy=("session", "trial"),
            independent_unit="participant",
            case_unit="session",
            cluster_units=("session",),
            inference_method="naive",
        )


def test_failure_preservation_requires_complete_method_case_cartesian_product():
    rows = (
        CaseOutcome("s1-t1", "eegnet", CaseStatus.OK, {"balanced_accuracy": 0.7}),
        CaseOutcome("s2-t1", "eegnet", CaseStatus.OOM, reason="GPU memory exhausted"),
        CaseOutcome("s1-t1", "csp", CaseStatus.OK, {"balanced_accuracy": 0.6}),
        CaseOutcome("s2-t1", "csp", CaseStatus.NONCONVERGED, reason="solver did not converge"),
    )
    result = FailurePreservingResultSet(
        declared_case_ids=("s1-t1", "s2-t1"),
        method_ids=("eegnet", "csp"),
        rows=rows,
    )
    assert result.status_counts()["oom"] == 1
    assert result.status_counts()["nonconverged"] == 1
    assert len(result.result_sha256) == 64

    with pytest.raises(ValueError, match="complete declared method/case matrix"):
        FailurePreservingResultSet(
            declared_case_ids=("s1-t1", "s2-t1"),
            method_ids=("eegnet", "csp"),
            rows=rows[:-1],
        )


def test_longitudinal_binding_preserves_frozen_protocol_and_blocks_evaluation_leakage():
    payload = {
        "dataset_id": "kumar2024",
        "case_id": "subject-1-session-1",
        "source_train_indices": [0, 1, 2, 3],
        "evaluation_indices": [8, 9, 10, 11],
        "calibration_order_by_class": {
            "left_hand": [4, 6],
            "right_hand": [5, 7],
        },
        "authority_fingerprint": "display-only",
        "partition_fingerprint": "partition-display",
        "calibration_split_fingerprint": "calibration-display",
        "processed_data_sha256": SHA_A,
        "history_policy": "prior",
        "held_out_values": ["1"],
    }
    observations, budget = bind_longitudinal_case_authority(
        payload,
        dataset_lineage=_kumar(),
        calibration_per_class=1,
    )
    roles = {item.role: item for item in observations}
    assert roles[ObservationRole.SOURCE_HISTORY].observation_ids == ("0", "1", "2", "3")
    assert roles[ObservationRole.LABELED_TARGET_CALIBRATION].observation_ids == ("4", "5")
    assert roles[ObservationRole.FINAL_ASSESSMENT].observation_ids == ("8", "9", "10", "11")
    assert budget.labeled_examples == 2
    assert budget.unlabeled_examples == 0

    with pytest.raises(ValueError, match="cannot borrow untouched final-assessment rows"):
        bind_longitudinal_case_authority(
            payload,
            dataset_lineage=_kumar(),
            calibration_per_class=1,
            unlabeled_target_observation_indices=(8,),
        )


def test_zero_label_longitudinal_case_does_not_fake_unlabeled_target_observation():
    payload = {
        "dataset_id": "kumar2024",
        "case_id": "subject-1-session-1",
        "source_train_indices": [0, 1],
        "evaluation_indices": [6, 7],
        "calibration_order_by_class": {"left": [2, 4], "right": [3, 5]},
        "processed_data_sha256": SHA_A,
        "history_policy": "prior",
        "held_out_values": ["1"],
    }
    observations, budget = bind_longitudinal_case_authority(
        payload,
        dataset_lineage=_kumar(),
        calibration_per_class=0,
    )
    calibration = next(
        item for item in observations if item.role is ObservationRole.LABELED_TARGET_CALIBRATION
    )
    assert calibration.observation_ids == ()
    assert budget.labeled_examples == 0
    assert budget.unlabeled_examples == 0


def test_clean_claim_requires_disjoint_audit_and_overlap_claim_must_be_contaminated():
    tueg = DatasetLineage(
        dataset_id="tueg",
        upstream_source="TUEG",
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    tuab = DatasetLineage(
        dataset_id="tuab",
        upstream_source="TUAB",
        parent_dataset_ids=("tueg",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    bendr = ModelLineage(
        model_id="bendr",
        upstream_source="BENDR",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=("tueg",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    overlap = audit_pretraining_overlap(bendr, tuab, known_datasets={"tueg": tueg})
    evidence_sha = _balanced_accuracy().metric_sha256

    with pytest.raises(ValueError, match="must be labeled contaminated"):
        ScientificStudyAuthority(
            study_id="bad-clean-claim",
            protocol_sha256=SHA_D,
            datasets=(tuab,),
            models=(bendr,),
            observations=(),
            preprocessing=(),
            metrics=(_balanced_accuracy(),),
            repeated_measures=_repeated(),
            overlap_audits=(overlap,),
            claims=(
                EvidenceClaim(
                    claim_id="bendr-tuab",
                    domain=EvidenceDomain.TASK_UTILITY,
                    scope="TUAB downstream classification",
                    qualification=ClaimQualification.CLEAN,
                    evidence_sha256s=(evidence_sha,),
                    model_id="bendr",
                    evaluation_dataset_id="tuab",
                ),
            ),
        )

    valid = ScientificStudyAuthority(
        study_id="qualified-overlap",
        protocol_sha256=SHA_D,
        datasets=(tuab,),
        models=(bendr,),
        observations=(),
        preprocessing=(),
        metrics=(_balanced_accuracy(),),
        repeated_measures=_repeated(),
        overlap_audits=(overlap,),
        claims=(
            EvidenceClaim(
                claim_id="bendr-tuab",
                domain=EvidenceDomain.TASK_UTILITY,
                scope="TUAB downstream classification",
                qualification=ClaimQualification.CONTAMINATED_PRETRAINING_OVERLAP,
                evidence_sha256s=(evidence_sha,),
                model_id="bendr",
                evaluation_dataset_id="tuab",
            ),
        ),
    )
    assert valid.claims[0].qualification is ClaimQualification.CONTAMINATED_PRETRAINING_OVERLAP


def test_machine_readable_report_separates_evidence_domains_and_claim_scope():
    dataset = _kumar()
    model = ModelLineage(
        model_id="specialist",
        upstream_source="neurOS EEGNet",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=(),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(model, dataset)
    metric = _balanced_accuracy()
    claim = EvidenceClaim(
        claim_id="offline-task-utility",
        domain=EvidenceDomain.TASK_UTILITY,
        scope="offline prospective next-session MI classification",
        qualification=ClaimQualification.CLEAN,
        evidence_sha256s=(metric.metric_sha256,),
        model_id="specialist",
        evaluation_dataset_id="kumar2024",
    )
    study = ScientificStudyAuthority(
        study_id="kumar2024-frontier",
        protocol_sha256=SHA_D,
        datasets=(dataset,),
        models=(model,),
        observations=(),
        preprocessing=(),
        metrics=(metric,),
        repeated_measures=_repeated(),
        overlap_audits=(audit,),
        claims=(claim,),
    )
    report = study.report()
    assert report["schema"] == "orion.scientific_authority.v2"
    assert len(report["study_sha256"]) == 64
    assert len(report["display_fingerprint"]) == 16
    assert report["evidence_domains"]["task_utility"][0]["claim_id"] == "offline-task-utility"
    assert report["evidence_domains"]["representation_geometry"] == []
    assert report["evidence_domains"]["mechanism"] == []
    assert report["evidence_domains"]["runtime"] == []
    assert report["evidence_domains"]["hardware"] == []
    assert report["claim_scope"] == [claim.to_dict()]
