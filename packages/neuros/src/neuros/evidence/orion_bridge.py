"""Bind validated ORION study claims into portable neurOS evidence references.

The bridge is deliberately asymmetric. ORION remains the authority for whether an
``EvidenceClaim`` is admissible inside a ``ScientificStudyAuthority``. neurOS only
binds that already-validated claim to the public, dependency-light claim contract.

No ORION ``ClaimQualification`` value is interpreted as a neurOS ``EvidenceTier``.
No evidence relation is inferred. Both relation and declared tier must be supplied
explicitly by the caller and remain declarations rather than earned qualification.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import string
from typing import Any

from .claims import (
    ClaimEvidenceRef,
    EvidenceRelation,
    EvidenceTier,
    ScientificClaimBundle,
    ScientificClaimSpec,
)

_SCHEMA_VERSION = 1
_DIGEST_DOMAIN = b"neuros.orion-claim-binding.v1\0"


def _digest(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(
        char not in string.hexdigits for char in value
    ):
        raise ValueError(f"{name} must be a 64-character SHA-256 hex digest")
    return value.lower()


def _text(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    return value


def _canonical_json(payload: dict[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


@dataclass(frozen=True, slots=True)
class OrionClaimBindingReceipt:
    """Immutable receipt linking one public claim to one validated ORION claim.

    ``orion_claim_qualification`` records the study-condition qualification ORION
    assigned to the embedded claim. It is intentionally separate from every
    ``ClaimEvidenceRef.declared_evidence_tier``.
    """

    public_claim_sha256: str
    orion_study_sha256: str
    orion_protocol_sha256: str
    orion_study_id: str
    orion_claim_id: str
    orion_domain: str
    orion_scope: str
    orion_claim_qualification: str
    relation: EvidenceRelation
    declared_evidence_tier: EvidenceTier
    evidence_refs: tuple[ClaimEvidenceRef, ...]
    schema_version: int = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "public_claim_sha256",
            _digest(self.public_claim_sha256, name="public_claim_sha256"),
        )
        object.__setattr__(
            self,
            "orion_study_sha256",
            _digest(self.orion_study_sha256, name="orion_study_sha256"),
        )
        object.__setattr__(
            self,
            "orion_protocol_sha256",
            _digest(self.orion_protocol_sha256, name="orion_protocol_sha256"),
        )
        for field_name in (
            "orion_study_id",
            "orion_claim_id",
            "orion_domain",
            "orion_scope",
            "orion_claim_qualification",
        ):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), name=field_name))

        relation = EvidenceRelation(self.relation)
        tier = EvidenceTier(self.declared_evidence_tier)
        object.__setattr__(self, "relation", relation)
        object.__setattr__(self, "declared_evidence_tier", tier)

        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION}")

        refs = tuple(self.evidence_refs)
        if not refs:
            raise ValueError("evidence_refs cannot be empty")
        if any(not isinstance(ref, ClaimEvidenceRef) for ref in refs):
            raise TypeError("evidence_refs must contain ClaimEvidenceRef values")
        if any(ref.relation is not relation for ref in refs):
            raise ValueError("every evidence ref must use the receipt relation")
        if any(ref.declared_evidence_tier is not tier for ref in refs):
            raise ValueError("every evidence ref must use the receipt declared_evidence_tier")
        if len({ref.evidence_sha256 for ref in refs}) != len(refs):
            raise ValueError("evidence_refs cannot repeat underlying evidence_sha256 values")
        object.__setattr__(
            self,
            "evidence_refs",
            tuple(sorted(refs, key=lambda ref: ref.reference_sha256)),
        )

    def to_manifest(self) -> dict[str, Any]:
        """Return the binding identity payload without self-referential digest fields."""

        return {
            "schema": "neuros.orion_claim_binding.v1",
            "schema_version": self.schema_version,
            "public_claim_sha256": self.public_claim_sha256,
            "orion_study_sha256": self.orion_study_sha256,
            "orion_protocol_sha256": self.orion_protocol_sha256,
            "orion_study_id": self.orion_study_id,
            "orion_claim_id": self.orion_claim_id,
            "orion_domain": self.orion_domain,
            "orion_scope": self.orion_scope,
            "orion_claim_qualification": self.orion_claim_qualification,
            "relation": self.relation.value,
            "declared_evidence_tier": self.declared_evidence_tier.value,
            "evidence": [
                {
                    "evidence_sha256": ref.evidence_sha256,
                    "reference_sha256": ref.reference_sha256,
                }
                for ref in self.evidence_refs
            ],
            "authority_boundary": {
                "orion_claim_qualification_is_evidence_tier": False,
                "declared_evidence_tier_is_earned_qualification": False,
                "relation_is_inferred": False,
            },
        }

    @property
    def binding_sha256(self) -> str:
        return sha256(_DIGEST_DOMAIN + _canonical_json(self.to_manifest())).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.to_manifest(),
            "binding_sha256": self.binding_sha256,
            "evidence_refs": [ref.to_dict() for ref in self.evidence_refs],
        }

    def to_bundle(self, claim: ScientificClaimSpec) -> ScientificClaimBundle:
        """Compose the receipt into a public claim bundle without upgrading authority."""

        if not isinstance(claim, ScientificClaimSpec):
            raise TypeError("claim must be a ScientificClaimSpec")
        if claim.claim_sha256 != self.public_claim_sha256:
            raise ValueError("claim identity does not match this ORION binding receipt")
        return ScientificClaimBundle(
            claim=claim,
            evidence=self.evidence_refs,
            protocol_sha256=self.orion_protocol_sha256,
            study_sha256s=(self.orion_study_sha256,),
            metadata={
                "orion_binding_sha256": self.binding_sha256,
                "orion_claim_qualification": self.orion_claim_qualification,
                "orion_claim_qualification_semantics": "study_condition_not_evidence_tier",
            },
        )


def bind_orion_study_claim(
    claim: ScientificClaimSpec,
    study: Any,
    *,
    relation: EvidenceRelation,
    declared_evidence_tier: EvidenceTier,
) -> OrionClaimBindingReceipt:
    """Bind an embedded ORION study claim to the matching public neurOS claim.

    The function intentionally accepts no external ``EvidenceClaim`` object. It
    selects the claim by ``claim.claim_id`` from the already-constructed study,
    preventing a caller from presenting a claim that ORION never admitted.
    """

    if not isinstance(claim, ScientificClaimSpec):
        raise TypeError("claim must be a ScientificClaimSpec")

    try:
        from orion.scientific_authority import ScientificStudyAuthority
    except ModuleNotFoundError as exc:  # pragma: no cover - exercised via import-failure regression
        if exc.name != "orion":
            raise
        raise RuntimeError(
            "ORION claim binding requires the neuros evidence extra / neuros-orion"
        ) from exc

    if not isinstance(study, ScientificStudyAuthority):
        raise TypeError("study must be an ORION ScientificStudyAuthority")

    matches = tuple(item for item in study.claims if item.claim_id == claim.claim_id)
    if not matches:
        raise ValueError(f"ORION study does not contain claim_id {claim.claim_id!r}")
    if len(matches) != 1:
        raise ValueError(f"ORION study contains duplicate claim_id {claim.claim_id!r}")
    embedded = matches[0]

    orion_domain = embedded.domain.value
    if claim.domain != orion_domain:
        raise ValueError(
            f"claim domain drift: public={claim.domain!r}, ORION={orion_domain!r}"
        )
    if claim.scope != embedded.scope:
        raise ValueError(
            f"claim scope drift: public={claim.scope!r}, ORION={embedded.scope!r}"
        )
    if not embedded.evidence_sha256s:
        raise ValueError("embedded ORION claim has no evidence SHA-256 values to bind")

    relation_value = EvidenceRelation(relation)
    tier_value = EvidenceTier(declared_evidence_tier)
    study_sha = _digest(study.study_sha256, name="study.study_sha256")
    protocol_sha = _digest(study.protocol_sha256, name="study.protocol_sha256")
    qualification = embedded.qualification.value
    source_id = f"{study.study_id}:{embedded.claim_id}"

    refs = tuple(
        ClaimEvidenceRef(
            evidence_sha256=_digest(evidence_sha, name="ORION evidence_sha256"),
            relation=relation_value,
            declared_evidence_tier=tier_value,
            source_kind="orion.scientific_authority.v2",
            source_id=source_id,
            metadata={
                "orion_study_sha256": study_sha,
                "orion_protocol_sha256": protocol_sha,
                "orion_claim_id": embedded.claim_id,
                "orion_evidence_domain": orion_domain,
                "orion_claim_qualification": qualification,
                "orion_zero_shot_claim": bool(embedded.zero_shot_claim),
                "authority_boundary": "declared_reference_not_earned_truth",
            },
        )
        for evidence_sha in embedded.evidence_sha256s
    )

    return OrionClaimBindingReceipt(
        public_claim_sha256=claim.claim_sha256,
        orion_study_sha256=study_sha,
        orion_protocol_sha256=protocol_sha,
        orion_study_id=study.study_id,
        orion_claim_id=embedded.claim_id,
        orion_domain=orion_domain,
        orion_scope=embedded.scope,
        orion_claim_qualification=qualification,
        relation=relation_value,
        declared_evidence_tier=tier_value,
        evidence_refs=refs,
    )
