"""Emit one deterministic Scientific Authority v2 report."""

from __future__ import annotations

import json

from orion import (
    CaseOutcome,
    CaseStatus,
    ClaimQualification,
    DatasetLineage,
    EvidenceClaim,
    EvidenceDomain,
    FailureAggregationPolicy,
    FailurePreservingResultSet,
    LineageCompleteness,
    MetricDirection,
    MetricSpec,
    ModelLineage,
    ProbabilityRequirement,
    RepeatedMeasuresAuthority,
    ScientificStudyAuthority,
    TargetObservationBudget,
    audit_pretraining_overlap,
)


def main() -> None:
    dataset = DatasetLineage(
        dataset_id="kumar2024",
        upstream_source="MOABB:Kumar2024",
        version="1.5",
        revision="example",
        content_sha256="a" * 64,
        sampling_assumptions={"sampling_rate_hz": 512.0, "channels": 22},
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="eegnet-from-declared-history",
        upstream_source="neurOS EEGNet",
        revision="example",
        checkpoint_sha256="b" * 64,
        pretraining_dataset_ids=(),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    overlap = audit_pretraining_overlap(model, dataset)
    metric = MetricSpec(
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
    repeated = RepeatedMeasuresAuthority(
        hierarchy=("participant", "session", "run", "trial"),
        independent_unit="participant",
        case_unit="participant-session",
        cluster_units=("participant",),
        inference_method="participant-cluster bootstrap",
        strata=("original_protocol", "held_out_session"),
    )
    result = FailurePreservingResultSet(
        declared_case_ids=("subject-1-session-1",),
        method_ids=(model.model_id,),
        rows=(
            CaseOutcome(
                case_id="subject-1-session-1",
                method_id=model.model_id,
                status=CaseStatus.OK,
                metrics={"balanced_accuracy": 0.68},
                metadata={"fixture": True},
            ),
        ),
    )
    zero_target = TargetObservationBudget(
        labeled_examples=0,
        labeled_examples_per_class=0,
        unlabeled_examples=0,
        unlabeled_seconds=0.0,
    )
    claim = EvidenceClaim(
        claim_id="prospective-session-task-utility",
        domain=EvidenceDomain.TASK_UTILITY,
        scope="offline prospective next-session motor-imagery classification",
        qualification=ClaimQualification.CLEAN,
        evidence_sha256s=(result.result_sha256,),
        model_id=model.model_id,
        evaluation_dataset_id=dataset.dataset_id,
        target_budget_id="zero-target-observation",
        zero_shot_claim=True,
    )
    study = ScientificStudyAuthority(
        study_id="kumar2024-scientific-authority-example",
        protocol_sha256="d" * 64,
        datasets=(dataset,),
        models=(model,),
        observations=(),
        preprocessing=(),
        metrics=(metric,),
        repeated_measures=repeated,
        overlap_audits=(overlap,),
        result_sets=(result,),
        target_budgets={"zero-target-observation": zero_target},
        claims=(claim,),
        metadata={
            "evidence_boundary": (
                "synthetic report-shape example only; the numeric result is not real-data "
                "evidence and does not support hardware, closed-loop, physiological, or clinical claims"
            )
        },
    )
    print(json.dumps(study.report(), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
