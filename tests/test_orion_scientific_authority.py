from __future__ import annotations

from dataclasses import replace
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


def _observation(
    role: ObservationRole,
    suffix: str,
    ids: tuple[str, ...] = ("0", "1"),
    *,
    dataset: DatasetLineage | None = None,
) -> ObservationSetAuthority:
    dataset = _kumar() if dataset is None else dataset
    return ObservationSetAuthority(
        authority_id=f"obs-{suffix}",
        dataset_lineage_sha256=dataset.lineage_sha256,
        role=role,
        observation_ids=ids,
        domain_id=f"{dataset.dataset_id}:subject-1:session-1",
    )


def _result(method_id: str, value: float = 0.65) -> FailurePreservingResultSet:
    return FailurePreservingResultSet(
        declared_case_ids=("case-1",),
        method_ids=(method_id,),
        rows=(
            CaseOutcome(
                "case-1",
                method_id,
                CaseStatus.OK,
                {"balanced_accuracy": value},
            ),
        ),
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
        DatasetLineage(dataset_id="bad", upstream_source="upstream", metadata={"ids": {"a", "b"}})
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
    bendr = ModelLineage(
        model_id="bendr",
        upstream_source="BENDR pretrained checkpoint",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=("tueg",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    for dataset_id in ("tuab", "tuev"):
        evaluation = DatasetLineage(
            dataset_id=dataset_id,
            upstream_source=dataset_id.upper(),
            parent_dataset_ids=("tueg",),
            lineage_completeness=LineageCompleteness.COMPLETE,
        )
        audit = audit_pretraining_overlap(
            bendr,
            evaluation,
            known_datasets={"tueg": tueg},
        )
        assert audit.status is OverlapStatus.OVERLAP_DETECTED
        assert audit.matched_dataset_ids == ("tueg",)
        assert len(audit.audit_sha256) == 64


def test_overlap_walks_transitive_dataset_ancestry():
    root = DatasetLineage(
        dataset_id="tueg",
        upstream_source="TUEG",
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    middle = DatasetLineage(
        dataset_id="temple-derived",
        upstream_source="derived",
        parent_dataset_ids=("tueg",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    leaf = DatasetLineage(
        dataset_id="evaluation-leaf",
        upstream_source="derived-evaluation",
        parent_dataset_ids=("temple-derived",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="pretrained",
        upstream_source="checkpoint",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=("tueg",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(
        model,
        leaf,
        known_datasets={"tueg": root, "temple-derived": middle},
    )
    assert audit.status is OverlapStatus.OVERLAP_DETECTED
    assert audit.matched_dataset_ids == ("tueg",)


def test_unresolved_ancestor_is_possible_overlap_not_verified_disjoint():
    evaluation = DatasetLineage(
        dataset_id="derived",
        upstream_source="evaluation",
        parent_dataset_ids=("unresolved-parent",),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="model",
        upstream_source="checkpoint",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=("another-domain",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(model, evaluation)
    assert audit.status is OverlapStatus.POSSIBLE_OVERLAP
    assert audit.unresolved_ancestor_ids == ("unresolved-parent",)


def test_declared_participant_overlap_is_detected_across_dataset_ids():
    evaluation = DatasetLineage(
        dataset_id="evaluation",
        upstream_source="eval",
        identity_sets=(
            IdentitySet(
                level="participant",
                availability=IdentityAvailability.AVAILABLE,
                identifiers=("P1", "P2"),
            ),
        ),
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="pretrained",
        upstream_source="checkpoint",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=("other-dataset",),
        pretraining_identity_sets=(
            IdentitySet(
                level="participant",
                availability=IdentityAvailability.AVAILABLE,
                identifiers=("P1",),
            ),
        ),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(model, evaluation)
    assert audit.status is OverlapStatus.OVERLAP_DETECTED
    assert audit.matched_identity_levels == ("participant",)


def test_complete_kumar_domain_is_verified_disjoint_from_tueg_pretraining():
    model = ModelLineage(
        model_id="bendr",
        upstream_source="BENDR",
        checkpoint_sha256=SHA_B,
        pretraining_dataset_ids=("tueg",),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(model, _kumar())
    assert audit.status is OverlapStatus.DISJOINT_VERIFIED


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


def test_preprocessing_and_training_cannot_consume_held_out_authority():
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
    assert len(authority.authority_sha256) == 64
    with pytest.raises(ValueError, match="require observation consumption"):
        PreprocessingFitAuthority(
            transform_id="bad",
            fit_kind=TransformFitKind.DATA_FITTED,
            implementation="x",
            implementation_version="1",
            state_sha256=SHA_C,
        )


def test_study_rejects_preprocessing_consumption_outside_declared_observation_universe():
    dataset = _kumar()
    source = _observation(ObservationRole.SOURCE_HISTORY, "source", dataset=dataset)
    consumption = ObservationConsumption.bind(
        operation_id="fit",
        operation=OperationKind.PREPROCESSING_FIT,
        observations=(source,),
    )
    transform = PreprocessingFitAuthority(
        transform_id="normalizer",
        fit_kind=TransformFitKind.DATA_FITTED,
        implementation="fixture",
        implementation_version="1",
        state_sha256=SHA_C,
        consumption=consumption,
    )
    with pytest.raises(ValueError, match="outside the declared study universe"):
        ScientificStudyAuthority(
            study_id="missing-observation",
            protocol_sha256=SHA_D,
            datasets=(dataset,),
            models=(),
            observations=(),
            preprocessing=(transform,),
            metrics=(_balanced_accuracy(),),
            repeated_measures=_repeated(),
        )


def test_labeled_and_unlabeled_target_budgets_are_separate():
    budget = TargetObservationBudget(
        labeled_examples=10,
        labeled_examples_per_class=5,
        unlabeled_examples=250,
        unlabeled_seconds=1.0,
    )
    assert budget.to_dict() == {
        "labeled_examples": 10,
        "labeled_examples_per_class": 5,
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
    with pytest.raises(ValueError, match="finite target_value"):
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


def test_repeated_measures_authority_requires_true_independent_cluster():
    assert _repeated().cluster_units == ("participant",)
    with pytest.raises(ValueError, match="independent_unit must be present"):
        RepeatedMeasuresAuthority(
            hierarchy=("session", "trial"),
            independent_unit="participant",
            case_unit="session",
            cluster_units=("session",),
            inference_method="naive",
        )
    with pytest.raises(ValueError, match="cluster_units must include"):
        RepeatedMeasuresAuthority(
            hierarchy=("participant", "session", "trial"),
            independent_unit="participant",
            case_unit="session",
            cluster_units=("session",),
            inference_method="session bootstrap",
        )


def test_failure_preservation_requires_complete_method_case_cartesian_product_and_metric_contract():
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
    result.require_metric_specs((_balanced_accuracy(),))
    assert result.status_counts()["oom"] == 1
    assert result.status_counts()["nonconverged"] == 1
    with pytest.raises(ValueError, match="complete declared method/case matrix"):
        FailurePreservingResultSet(
            declared_case_ids=("s1-t1", "s2-t1"),
            method_ids=("eegnet", "csp"),
            rows=rows[:-1],
        )
    with pytest.raises(ValueError, match="at least one metric"):
        CaseOutcome("s1", "bad", CaseStatus.OK, {})


def _longitudinal_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dataset_id": "kumar2024",
        "case_id": "subject-1-session-1",
        "split_unit": "session",
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
        "n_samples": 12,
    }


def test_longitudinal_binding_preserves_frozen_protocol_and_full_case_identity():
    payload = _longitudinal_payload()
    observations, budget = bind_longitudinal_case_authority(
        payload,
        dataset_lineage=_kumar(),
        calibration_per_class=1,
    )
    roles = {item.role: item for item in observations}
    assert roles[ObservationRole.SOURCE_HISTORY].observation_ids == ("0", "1", "2", "3")
    assert roles[ObservationRole.LABELED_TARGET_CALIBRATION].observation_ids == ("4", "5")
    assert roles[ObservationRole.FINAL_ASSESSMENT].observation_ids == ("8", "9", "10", "11")
    full_case_sha = roles[ObservationRole.SOURCE_HISTORY].metadata["source_authority_sha256"]
    assert isinstance(full_case_sha, str) and len(full_case_sha) == 64
    assert roles[ObservationRole.FINAL_ASSESSMENT].metadata["source_authority_sha256"] == full_case_sha
    assert roles[ObservationRole.SOURCE_HISTORY].metadata["legacy_source_authority_fingerprint"] == "display-only"
    assert budget.labeled_examples == 2
    assert budget.labeled_examples_per_class == 1
    assert budget.unlabeled_examples == 0


def test_longitudinal_binding_blocks_evaluation_source_and_coercive_index_leakage():
    payload = _longitudinal_payload()
    with pytest.raises(ValueError, match="cannot borrow untouched final-assessment rows"):
        bind_longitudinal_case_authority(
            payload,
            dataset_lineage=_kumar(),
            calibration_per_class=1,
            unlabeled_target_observation_indices=(8,),
        )
    with pytest.raises(ValueError, match="cannot borrow source-history rows"):
        bind_longitudinal_case_authority(
            payload,
            dataset_lineage=_kumar(),
            calibration_per_class=1,
            unlabeled_target_observation_indices=(0,),
        )
    malformed = _longitudinal_payload()
    malformed["source_train_indices"] = [0.0, 1, 2, 3]
    with pytest.raises(ValueError, match="actual integers without coercion"):
        bind_longitudinal_case_authority(
            malformed,
            dataset_lineage=_kumar(),
            calibration_per_class=1,
        )


def test_longitudinal_full_authority_sha_is_verified_when_supplied():
    payload = _longitudinal_payload()
    payload["authority_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="does not match its content"):
        bind_longitudinal_case_authority(
            payload,
            dataset_lineage=_kumar(),
            calibration_per_class=1,
        )


def test_zero_label_longitudinal_case_does_not_fake_unlabeled_target_observation():
    payload = _longitudinal_payload()
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
    assert budget.labeled_examples_per_class == 0
    assert budget.unlabeled_examples == 0


def test_overlap_claim_must_be_contaminated_and_task_utility_cites_result_bundle():
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
    result = _result("bendr")
    metric = _balanced_accuracy()
    with pytest.raises(ValueError, match="must be labeled contaminated"):
        ScientificStudyAuthority(
            study_id="bad-clean-claim",
            protocol_sha256=SHA_D,
            datasets=(tuab,),
            models=(bendr,),
            observations=(),
            preprocessing=(),
            metrics=(metric,),
            repeated_measures=_repeated(),
            overlap_audits=(overlap,),
            result_sets=(result,),
            claims=(
                EvidenceClaim(
                    claim_id="bendr-tuab",
                    domain=EvidenceDomain.TASK_UTILITY,
                    scope="TUAB downstream classification",
                    qualification=ClaimQualification.CLEAN,
                    evidence_sha256s=(result.result_sha256,),
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
        metrics=(metric,),
        repeated_measures=_repeated(),
        overlap_audits=(overlap,),
        result_sets=(result,),
        claims=(
            EvidenceClaim(
                claim_id="bendr-tuab",
                domain=EvidenceDomain.TASK_UTILITY,
                scope="TUAB downstream classification",
                qualification=ClaimQualification.CONTAMINATED_PRETRAINING_OVERLAP,
                evidence_sha256s=(result.result_sha256,),
                model_id="bendr",
                evaluation_dataset_id="tuab",
            ),
        ),
    )
    assert valid.claims[0].qualification is ClaimQualification.CONTAMINATED_PRETRAINING_OVERLAP


def test_study_rejects_stale_or_forged_overlap_audit():
    dataset = _kumar()
    model = ModelLineage(
        model_id="model",
        upstream_source="checkpoint",
        checkpoint_sha256=SHA_B,
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(model, dataset)
    forged = replace(audit, model_lineage_sha256="0" * 64)
    with pytest.raises(ValueError, match="stale or forged"):
        ScientificStudyAuthority(
            study_id="forged",
            protocol_sha256=SHA_D,
            datasets=(dataset,),
            models=(model,),
            observations=(),
            preprocessing=(),
            metrics=(_balanced_accuracy(),),
            repeated_measures=_repeated(),
            overlap_audits=(forged,),
        )


def test_zero_shot_claim_requires_zero_labeled_and_unlabeled_target_information():
    dataset = _kumar()
    model = ModelLineage(
        model_id="specialist",
        upstream_source="neurOS EEGNet",
        checkpoint_sha256=SHA_B,
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(model, dataset)
    result = _result(model.model_id)
    claim = EvidenceClaim(
        claim_id="zero-shot",
        domain=EvidenceDomain.TASK_UTILITY,
        scope="no target-session observation",
        qualification=ClaimQualification.CLEAN,
        evidence_sha256s=(result.result_sha256,),
        model_id=model.model_id,
        evaluation_dataset_id=dataset.dataset_id,
        target_budget_id="zero",
        zero_shot_claim=True,
    )
    with pytest.raises(ValueError, match="target information budget is nonzero"):
        ScientificStudyAuthority(
            study_id="not-zero-shot",
            protocol_sha256=SHA_D,
            datasets=(dataset,),
            models=(model,),
            observations=(),
            preprocessing=(),
            metrics=(_balanced_accuracy(),),
            repeated_measures=_repeated(),
            overlap_audits=(audit,),
            result_sets=(result,),
            target_budgets={"zero": TargetObservationBudget(unlabeled_examples=50)},
            claims=(claim,),
        )
    valid = ScientificStudyAuthority(
        study_id="true-zero-shot",
        protocol_sha256=SHA_D,
        datasets=(dataset,),
        models=(model,),
        observations=(),
        preprocessing=(),
        metrics=(_balanced_accuracy(),),
        repeated_measures=_repeated(),
        overlap_audits=(audit,),
        result_sets=(result,),
        target_budgets={"zero": TargetObservationBudget()},
        claims=(claim,),
    )
    assert valid.target_budgets["zero"].unlabeled_examples == 0


def test_machine_readable_report_separates_evidence_domains_and_claim_scope():
    dataset = _kumar()
    model = ModelLineage(
        model_id="specialist",
        upstream_source="neurOS EEGNet",
        checkpoint_sha256=SHA_B,
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    audit = audit_pretraining_overlap(model, dataset)
    metric = _balanced_accuracy()
    result = _result(model.model_id)
    claim = EvidenceClaim(
        claim_id="offline-task-utility",
        domain=EvidenceDomain.TASK_UTILITY,
        scope="offline prospective next-session MI classification",
        qualification=ClaimQualification.CLEAN,
        evidence_sha256s=(result.result_sha256,),
        model_id=model.model_id,
        evaluation_dataset_id=dataset.dataset_id,
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
        result_sets=(result,),
        claims=(claim,),
    )
    report = study.report()
    assert report["schema"] == "orion.scientific_authority.v2"
    assert len(report["study_sha256"]) == 64
    assert report["result_sets"][0]["result_sha256"] == result.result_sha256
    assert report["evidence_domains"]["task_utility"][0]["claim_id"] == "offline-task-utility"
    assert report["evidence_domains"]["representation_geometry"] == []
    assert report["evidence_domains"]["mechanism"] == []
    assert report["evidence_domains"]["runtime"] == []
    assert report["evidence_domains"]["hardware"] == []
    assert report["evidence_domains"]["clinical"] == []
    assert report["claim_scope"] == [claim.to_dict()]
