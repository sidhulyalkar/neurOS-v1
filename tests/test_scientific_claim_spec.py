from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from neuros.evidence.claims import (
    ClaimEvidenceRef,
    EvidenceRelation,
    EvidenceRequirement,
    EvidenceTier,
    ScientificClaimBundle,
    ScientificClaimSpec,
)


def _sha(char: str) -> str:
    return char * 64


def _requirement(
    requirement_id: str,
    tier: EvidenceTier = EvidenceTier.SOFTWARE_CONTRACT,
) -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=requirement_id,
        authority_type="test_authority",
        description=f"require {requirement_id}",
        required_tier=tier,
        metadata={"nested": {"b": 2, "a": 1}},
    )


def _claim(*requirements: EvidenceRequirement) -> ScientificClaimSpec:
    return ScientificClaimSpec(
        claim_id="decoder-transfer",
        statement="A frozen decoder transfers across held-out subjects.",
        domain="neural_decoding",
        scope="held_out_subjects",
        inference_unit="subject",
        target_evidence_tier=EvidenceTier.REAL_DATA,
        requirements=tuple(requirements) or (_requirement("software"),),
        metadata={"protocol_family": "subject_disjoint", "version": 1},
    )


def _evidence(
    char: str,
    *,
    relation: EvidenceRelation = EvidenceRelation.SUPPORTS,
    tier: EvidenceTier = EvidenceTier.SOFTWARE_CONTRACT,
    source_id: str | None = None,
) -> ClaimEvidenceRef:
    return ClaimEvidenceRef(
        evidence_sha256=_sha(char),
        relation=relation,
        declared_evidence_tier=tier,
        source_kind="qualification_receipt",
        source_id=source_id or f"receipt-{char}",
        metadata={"provider": "test", "attempt": 1},
    )


def test_claim_identity_is_independent_of_requirement_and_mapping_order() -> None:
    left = ScientificClaimSpec(
        claim_id="decoder-transfer",
        statement="A frozen decoder transfers across held-out subjects.",
        domain="neural_decoding",
        scope="held_out_subjects",
        inference_unit="subject",
        target_evidence_tier=EvidenceTier.REAL_DATA,
        requirements=(_requirement("z-last"), _requirement("a-first")),
        metadata={"outer": {"z": 2, "a": 1}, "name": "same"},
    )
    right = ScientificClaimSpec(
        claim_id="decoder-transfer",
        statement="A frozen decoder transfers across held-out subjects.",
        domain="neural_decoding",
        scope="held_out_subjects",
        inference_unit="subject",
        target_evidence_tier=EvidenceTier.REAL_DATA,
        requirements=(_requirement("a-first"), _requirement("z-last")),
        metadata={"name": "same", "outer": {"a": 1, "z": 2}},
    )

    assert left.claim_sha256 == right.claim_sha256
    assert [item.requirement_id for item in left.requirements] == ["a-first", "z-last"]
    assert left.to_dict() == right.to_dict()


def test_bundle_identity_is_independent_of_evidence_and_study_order() -> None:
    claim = _claim(_requirement("software"), _requirement("real-data", EvidenceTier.REAL_DATA))
    first = _evidence("a")
    second = _evidence("b", tier=EvidenceTier.REAL_DATA)

    left = ScientificClaimBundle(
        claim=claim,
        evidence=(second, first),
        protocol_sha256=_sha("c"),
        study_sha256s=(_sha("e"), _sha("d")),
        metadata={"site": "multi"},
    )
    right = ScientificClaimBundle(
        claim=claim,
        evidence=(first, second),
        protocol_sha256=_sha("c"),
        study_sha256s=(_sha("d"), _sha("e")),
        metadata={"site": "multi"},
    )

    assert left.bundle_sha256 == right.bundle_sha256
    assert left.to_dict() == right.to_dict()
    assert left.study_sha256s == (_sha("d"), _sha("e"))


def test_bundle_rejects_duplicate_underlying_evidence_even_if_reference_differs() -> None:
    claim = _claim()
    support = _evidence("a", relation=EvidenceRelation.SUPPORTS, source_id="support-view")
    context = _evidence("a", relation=EvidenceRelation.CONTEXT, source_id="context-view")

    with pytest.raises(ValueError, match="same underlying evidence_sha256"):
        ScientificClaimBundle(claim=claim, evidence=(support, context))


def test_target_and_declared_tiers_never_become_earned_qualification() -> None:
    claim = _claim(_requirement("real-data", EvidenceTier.REAL_DATA))
    ref = _evidence("a", tier=EvidenceTier.SOFTWARE_CONTRACT)
    bundle = ScientificClaimBundle(claim=claim, evidence=(ref,))
    payload = bundle.to_dict()

    assert payload["claim"]["target_evidence_tier"] == "real_data"
    assert payload["evidence"][0]["declared_evidence_tier"] == "software_contract"
    serialized = repr(payload)
    assert "achieved_evidence_tier" not in serialized
    assert "truth" not in serialized
    assert "confidence" not in serialized
    assert "winner" not in serialized


def test_duplicate_requirement_ids_fail_closed() -> None:
    with pytest.raises(ValueError, match="duplicate requirement_id"):
        _claim(_requirement("same"), _requirement("same"))


def test_metadata_is_deeply_frozen_and_rejects_unordered_values() -> None:
    requirement = _requirement("immutable")
    with pytest.raises(TypeError):
        requirement.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        requirement.metadata["nested"]["new"] = 3  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        requirement.description = "changed"  # type: ignore[misc]

    with pytest.raises(TypeError, match="unordered sets"):
        EvidenceRequirement(
            requirement_id="bad-set",
            authority_type="test",
            description="reject nondeterministic metadata",
            required_tier=EvidenceTier.SOFTWARE_CONTRACT,
            metadata={"bad": {"unordered", "set"}},
        )

    with pytest.raises(TypeError, match="mapping keys must be strings"):
        EvidenceRequirement(
            requirement_id="bad-key",
            authority_type="test",
            description="reject cross-language key coercion",
            required_tier=EvidenceTier.SOFTWARE_CONTRACT,
            metadata={1: "not portable"},  # type: ignore[dict-item]
        )


def test_nonfinite_metadata_and_malformed_digests_fail_closed() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        EvidenceRequirement(
            requirement_id="nan",
            authority_type="test",
            description="reject nonfinite value",
            required_tier=EvidenceTier.SOFTWARE_CONTRACT,
            metadata={"value": float("nan")},
        )

    with pytest.raises(ValueError, match="64-character SHA-256"):
        ClaimEvidenceRef(
            evidence_sha256="not-a-digest",
            relation=EvidenceRelation.SUPPORTS,
            declared_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
            source_kind="receipt",
            source_id="bad",
        )


def test_negative_zero_is_canonicalized_for_cross_runtime_identity() -> None:
    left = ScientificClaimSpec(
        claim_id="zero",
        statement="Zero-valued metadata is canonical.",
        domain="contract_test",
        scope="unit",
        inference_unit="case",
        target_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        requirements=(_requirement("software"),),
        metadata={"value": -0.0},
    )
    right = ScientificClaimSpec(
        claim_id="zero",
        statement="Zero-valued metadata is canonical.",
        domain="contract_test",
        scope="unit",
        inference_unit="case",
        target_evidence_tier=EvidenceTier.SOFTWARE_CONTRACT,
        requirements=(_requirement("software"),),
        metadata={"value": 0.0},
    )

    assert left.claim_sha256 == right.claim_sha256
    assert left.to_dict()["metadata"]["value"] == 0.0


def test_content_changes_change_domain_separated_identities() -> None:
    claim = _claim()
    changed = ScientificClaimSpec(
        claim_id=claim.claim_id,
        statement=claim.statement + " Strictly.",
        domain=claim.domain,
        scope=claim.scope,
        inference_unit=claim.inference_unit,
        target_evidence_tier=claim.target_evidence_tier,
        requirements=claim.requirements,
        metadata=claim.metadata,
    )
    ref = _evidence("a")

    assert claim.claim_sha256 != changed.claim_sha256
    assert ref.reference_sha256 != claim.claim_sha256
    assert ScientificClaimBundle(claim=claim, evidence=(ref,)).bundle_sha256 not in {
        claim.claim_sha256,
        ref.reference_sha256,
    }
