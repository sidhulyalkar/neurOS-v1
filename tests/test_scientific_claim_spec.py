from __future__ import annotations

import pytest

from neuros.evidence import (
    ClaimEvidenceRef,
    EvidenceRelation,
    EvidenceRequirement,
    EvidenceTier,
    ScientificClaimBundle,
    ScientificClaimSpec,
)


def _claim(**metadata):
    return ScientificClaimSpec(
        claim_id="kumar2024-calibration-frontier",
        statement=(
            "EEGNet requires less participant-specific labeled calibration than "
            "CSP+LDA under the frozen longitudinal Kumar2024 protocol."
        ),
        domain="task_utility",
        scope="MOABB Kumar2024 bar-feedback subset; prospective prior-session history",
        inference_unit="participant",
        target_evidence_tier=EvidenceTier.REAL_DATA,
        requirements=(
            EvidenceRequirement(
                requirement_id="prospective-authority",
                authority_type="LongitudinalCaseAuthority",
                description="Final-assessment observations remain untouched.",
                required_tier=EvidenceTier.REAL_DATA,
            ),
            EvidenceRequirement(
                requirement_id="failure-preserving-results",
                authority_type="FailurePreservingResultSet",
                description="All declared participant/session/budget cases remain visible.",
                required_tier=EvidenceTier.REAL_DATA,
            ),
        ),
        metadata=metadata,
    )


def test_claim_identity_is_stable_across_mapping_order():
    first = _claim(beta=2, alpha={"y": 2, "x": 1})
    second = _claim(alpha={"x": 1, "y": 2}, beta=2)
    assert first.claim_sha256 == second.claim_sha256
    assert first.to_dict()["schema"] == "neuros.scientific_claim.v1"


def test_claim_rejects_duplicate_requirement_ids():
    requirement = EvidenceRequirement(
        requirement_id="same",
        authority_type="Authority",
        description="first",
        required_tier=EvidenceTier.SOFTWARE_CONTRACT,
    )
    with pytest.raises(ValueError, match="duplicate requirement_id"):
        ScientificClaimSpec(
            claim_id="claim",
            statement="statement",
            domain="task_utility",
            scope="scope",
            inference_unit="participant",
            target_evidence_tier=EvidenceTier.REAL_DATA,
            requirements=(requirement, requirement),
        )


def test_claim_metadata_is_immutable_and_rejects_unordered_sets():
    claim = _claim(nested={"items": [1, 2]})
    with pytest.raises(TypeError):
        claim.metadata["new"] = "value"
    with pytest.raises(TypeError, match="unordered sets"):
        _claim(bad={"x", "y"})


def test_evidence_ref_requires_content_identity():
    with pytest.raises(ValueError, match="64-character"):
        ClaimEvidenceRef(
            evidence_sha256="abc",
            relation=EvidenceRelation.SUPPORTS,
            evidence_tier=EvidenceTier.REAL_DATA,
            source_kind="orion.failure_preserving_result_set",
            source_id="result-set-1",
        )


def test_bundle_binds_claim_without_promoting_evidence():
    claim = _claim()
    evidence = ClaimEvidenceRef(
        evidence_sha256="a" * 64,
        relation=EvidenceRelation.SUPPORTS,
        evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        source_kind="orion.failure_preserving_result_set",
        source_id="result-set-1",
    )
    bundle = ScientificClaimBundle(
        claim=claim,
        evidence=(evidence,),
        protocol_sha256="b" * 64,
        study_sha256s=("c" * 64,),
    )

    payload = bundle.to_dict()
    assert payload["claim"]["target_evidence_tier"] == EvidenceTier.REAL_DATA.value
    assert payload["evidence"][0]["evidence_tier"] == EvidenceTier.SOFTWARE_CONTRACT.value
    assert "achieved_evidence_tier" not in payload
    assert len(bundle.bundle_sha256) == 64


def test_bundle_rejects_duplicate_evidence_reference():
    claim = _claim()
    evidence = ClaimEvidenceRef(
        evidence_sha256="d" * 64,
        relation=EvidenceRelation.CONTEXT,
        evidence_tier=EvidenceTier.REAL_DATA,
        source_kind="orion.scientific_authority.v2",
        source_id="study",
    )
    with pytest.raises(ValueError, match="duplicate references"):
        ScientificClaimBundle(claim=claim, evidence=(evidence, evidence))
