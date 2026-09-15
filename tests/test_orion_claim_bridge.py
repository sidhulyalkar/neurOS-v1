from __future__ import annotations

from dataclasses import replace
from importlib.util import find_spec

import pytest

from neuros.evidence import (
    EvidenceRelation,
    EvidenceRequirement,
    EvidenceTier,
    ScientificClaimSpec,
)
from neuros.evidence.orion_bridge import bind_orion_study_claim

ORION_AVAILABLE = find_spec("orion") is not None
if ORION_AVAILABLE:
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
        audit_pretraining_overlap,
    )

requires_orion = pytest.mark.skipif(
    not ORION_AVAILABLE,
    reason="ORION integration contracts require the neuros-orion distribution",
)

SHA_DATA = "a" * 64
SHA_MODEL = "b" * 64
SHA_PROTOCOL = "d" * 64


def _study() -> tuple[ScientificStudyAuthority, FailurePreservingResultSet]:
    dataset = DatasetLineage(
        dataset_id="kumar2024",
        upstream_source="MOABB:Kumar2024",
        content_sha256=SHA_DATA,
        lineage_completeness=LineageCompleteness.COMPLETE,
    )
    model = ModelLineage(
        model_id="eegnet-declared-history",
        upstream_source="neurOS EEGNet",
        checkpoint_sha256=SHA_MODEL,
        pretraining_dataset_ids=(),
        pretraining_lineage_completeness=LineageCompleteness.COMPLETE,
    )
    result = FailurePreservingResultSet(
        declared_case_ids=("case-1",),
        method_ids=(model.model_id,),
        rows=(
            CaseOutcome(
                case_id="case-1",
                method_id=model.model_id,
                status=CaseStatus.OK,
                metrics={"balanced_accuracy": 0.68},
            ),
        ),
    )
    metric = MetricSpec(
        metric_id="balanced_accuracy",
        version="fixture-v1",
        direction=MetricDirection.HIGHER_IS_BETTER,
        averaging="macro recall",
        class_semantics="two declared classes",
        probability_requirement=ProbabilityRequirement.NONE,
        estimator="fixture",
        estimator_version="1",
        aggregation_unit="participant-session case",
        failure_policy=FailureAggregationPolicy.PRESERVE,
        uncertainty_method="participant-cluster bootstrap",
        primary=True,
    )
    repeated = RepeatedMeasuresAuthority(
        hierarchy=("participant", "session"),
        independent_unit="participant",
        case_unit="participant-session",
        cluster_units=("participant",),
        inference_method="participant-cluster bootstrap",
    )
    claim = EvidenceClaim(
        claim_id="prospective-session-task-utility",
        domain=EvidenceDomain.TASK_UTILITY,
        scope="offline prospective next-session motor-imagery classification",
        qualification=ClaimQualification.CLEAN,
        evidence_sha256s=(result.result_sha256,),
        model_id=model.model_id,
        evaluation_dataset_id=dataset.dataset_id,
    )
    study = ScientificStudyAuthority(
        study_id="bridge-fixture",
        protocol_sha256=SHA_PROTOCOL,
        datasets=(dataset,),
        models=(model,),
        observations=(),
        preprocessing=(),
        metrics=(metric,),
        repeated_measures=repeated,
        overlap_audits=(audit_pretraining_overlap(model, dataset),),
        result_sets=(result,),
        claims=(claim,),
    )
    return study, result


def _public_claim(
    *,
    claim_id: str = "prospective-session-task-utility",
    domain: str = "task_utility",
    scope: str = "offline prospective next-session motor-imagery classification",
) -> ScientificClaimSpec:
    return ScientificClaimSpec(
        claim_id=claim_id,
        statement="A declared-history decoder has task-utility evidence in the frozen study.",
        domain=domain,
        scope=scope,
        inference_unit="participant-session",
        target_evidence_tier=EvidenceTier.REAL_DATA,
        requirements=(
            EvidenceRequirement(
                requirement_id="orion-study-authority",
                authority_type="orion.scientific_authority.v2",
                description="ORION study authority must admit the exact claim and evidence.",
                required_tier=EvidenceTier.SOFTWARE_CONTRACT,
            ),
        ),
    )


def test_bridge_surface_imports_without_eager_orion_dependency() -> None:
    assert callable(bind_orion_study_claim)


@requires_orion
def test_bridge_binds_only_embedded_orion_claim_and_exact_evidence() -> None:
    study, result = _study()
    claim = _public_claim()

    receipt = bind_orion_study_claim(
        claim,
        study,
        relation=EvidenceRelation.SUPPORTS,
        declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
    )

    assert receipt.public_claim_sha256 == claim.claim_sha256
    assert receipt.orion_study_sha256 == study.study_sha256
    assert receipt.orion_protocol_sha256 == study.protocol_sha256
    assert receipt.orion_claim_qualification == ClaimQualification.CLEAN.value
    assert [ref.evidence_sha256 for ref in receipt.evidence_refs] == [result.result_sha256]
    assert receipt.evidence_refs[0].declared_evidence_tier is EvidenceTier.SOFTWARE_CONTRACT
    assert len(receipt.binding_sha256) == 64

    bundle = receipt.to_bundle(claim)
    assert bundle.claim.claim_sha256 == claim.claim_sha256
    assert bundle.study_sha256s == (study.study_sha256,)
    assert bundle.protocol_sha256 == study.protocol_sha256
    assert bundle.metadata["orion_binding_sha256"] == receipt.binding_sha256


@requires_orion
def test_clean_orion_qualification_never_maps_to_an_evidence_tier() -> None:
    study, _ = _study()
    claim = _public_claim()

    software = bind_orion_study_claim(
        claim,
        study,
        relation=EvidenceRelation.CONTEXT,
        declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
    )
    real_data = bind_orion_study_claim(
        claim,
        study,
        relation=EvidenceRelation.CONTEXT,
        declared_evidence_tier=EvidenceTier.REAL_DATA,
    )

    assert software.orion_claim_qualification == real_data.orion_claim_qualification == "clean"
    assert software.declared_evidence_tier is EvidenceTier.SOFTWARE_CONTRACT
    assert real_data.declared_evidence_tier is EvidenceTier.REAL_DATA
    assert software.binding_sha256 != real_data.binding_sha256
    manifest = software.to_manifest()
    assert manifest["authority_boundary"] == {
        "orion_claim_qualification_is_evidence_tier": False,
        "declared_evidence_tier_is_earned_qualification": False,
        "relation_is_inferred": False,
    }


@requires_orion
def test_relation_is_explicit_and_changes_binding_identity() -> None:
    study, _ = _study()
    claim = _public_claim()
    support = bind_orion_study_claim(
        claim,
        study,
        relation=EvidenceRelation.SUPPORTS,
        declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
    )
    contradiction = bind_orion_study_claim(
        claim,
        study,
        relation=EvidenceRelation.CONTRADICTS,
        declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
    )
    assert support.binding_sha256 != contradiction.binding_sha256
    assert support.evidence_refs[0].relation is EvidenceRelation.SUPPORTS
    assert contradiction.evidence_refs[0].relation is EvidenceRelation.CONTRADICTS


@requires_orion
def test_bridge_rejects_public_domain_and_scope_drift() -> None:
    study, _ = _study()
    with pytest.raises(ValueError, match="domain drift"):
        bind_orion_study_claim(
            _public_claim(domain="mechanism"),
            study,
            relation=EvidenceRelation.CONTEXT,
            declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        )
    with pytest.raises(ValueError, match="scope drift"):
        bind_orion_study_claim(
            _public_claim(scope="different population"),
            study,
            relation=EvidenceRelation.CONTEXT,
            declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        )


@requires_orion
def test_bridge_cannot_bind_claim_orion_never_admitted() -> None:
    study, _ = _study()
    with pytest.raises(ValueError, match="does not contain claim_id"):
        bind_orion_study_claim(
            _public_claim(claim_id="invented-after-the-fact"),
            study,
            relation=EvidenceRelation.SUPPORTS,
            declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        )


@requires_orion
def test_receipt_cannot_be_reused_for_a_different_public_claim() -> None:
    study, _ = _study()
    claim = _public_claim()
    receipt = bind_orion_study_claim(
        claim,
        study,
        relation=EvidenceRelation.SUPPORTS,
        declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
    )
    changed = replace(claim, statement=claim.statement + " Changed proposition.")
    with pytest.raises(ValueError, match="claim identity"):
        receipt.to_bundle(changed)


@requires_orion
def test_bridge_requires_real_orion_study_authority() -> None:
    with pytest.raises(TypeError, match="ScientificStudyAuthority"):
        bind_orion_study_claim(
            _public_claim(),
            {"claims": []},
            relation=EvidenceRelation.CONTEXT,
            declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        )


@requires_orion
def test_binding_is_deterministic_for_same_authority() -> None:
    study, _ = _study()
    claim = _public_claim()
    kwargs = {
        "relation": EvidenceRelation.REPLICATES,
        "declared_evidence_tier": EvidenceTier.REPLAY_OR_SYNTHETIC,
    }
    first = bind_orion_study_claim(claim, study, **kwargs)
    second = bind_orion_study_claim(claim, study, **kwargs)
    assert first.to_dict() == second.to_dict()
    assert first.binding_sha256 == second.binding_sha256
