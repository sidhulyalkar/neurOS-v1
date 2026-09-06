"""Machine-readable scientific claim manifests for the public neurOS evidence layer.

This module is intentionally dependency-light. It defines the proposition and
evidence-reference layer above Scientific Authority / NSQ without duplicating
their qualification logic. A claim manifest can say what evidence would be
required and which immutable artifacts are being cited; it cannot promote
itself to a stronger scientific evidence tier.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _nonempty(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    return normalized


def _require_sha256(name: str, value: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a SHA-256 string")
    normalized = value.strip().lower()
    if not _SHA256_RE.fullmatch(normalized):
        raise ValueError(f"{name} must be a 64-character lowercase SHA-256 digest")
    return normalized


def _canonical_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("claim manifests cannot contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0])):
            normalized_key = str(key)
            if not normalized_key.strip():
                raise ValueError("claim manifest mapping keys must be non-empty")
            if normalized_key in normalized:
                raise ValueError(
                    "claim manifest mapping keys collide after string normalization: "
                    f"{normalized_key!r}"
                )
            normalized[normalized_key] = _canonical_json(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_canonical_json(item) for item in value]
    if isinstance(value, (set, frozenset)):
        raise TypeError("unordered sets are not valid claim manifest values")
    raise TypeError(
        "claim manifest values must be deterministic JSON-compatible primitives, "
        f"mappings, lists, or tuples; got {type(value).__name__}"
    )


def _freeze_json(value: Any) -> Any:
    normalized = _canonical_json(value)
    if isinstance(normalized, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in normalized.items()})
    if isinstance(normalized, list):
        return tuple(_freeze_json(item) for item in normalized)
    return normalized


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        _canonical_json(value),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


class EvidenceTier(str, Enum):
    """Evidence strata. They are labels, not automatically comparable scores."""

    SOFTWARE_CONTRACT = "software_contract"
    INTEGRATION = "integration"
    REPLAY_OR_SYNTHETIC = "replay_or_synthetic"
    REAL_DATA = "real_data"
    PHYSICAL_HARDWARE = "physical_hardware"
    CLOSED_LOOP = "closed_loop"
    CLINICAL = "clinical"


class EvidenceRelation(str, Enum):
    """How one cited evidence object relates to a claim."""

    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    REPLICATES = "replicates"
    CONTEXT = "context"


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """One predeclared requirement for evaluating a scientific claim."""

    requirement_id: str
    authority_type: str
    description: str
    required_tier: EvidenceTier
    required: bool = True
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("EvidenceRequirement schema_version must be 1")
        if not isinstance(self.required_tier, EvidenceTier):
            raise TypeError("required_tier must be EvidenceTier")
        if not isinstance(self.required, bool):
            raise TypeError("required must be boolean")
        object.__setattr__(self, "requirement_id", _nonempty("requirement_id", self.requirement_id))
        object.__setattr__(self, "authority_type", _nonempty("authority_type", self.authority_type))
        object.__setattr__(self, "description", _nonempty("description", self.description))
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requirement_id": self.requirement_id,
            "authority_type": self.authority_type,
            "description": self.description,
            "required_tier": self.required_tier.value,
            "required": self.required,
            "metadata": _thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ScientificClaimSpec:
    """Canonical proposition-level contract.

    This is intentionally upstream of Scientific Authority / NSQ qualification.
    ``target_evidence_tier`` describes the level of evidence the protocol intends
    to earn. It never implies that the claim has earned that tier.
    """

    claim_id: str
    statement: str
    domain: str
    scope: str
    inference_unit: str
    target_evidence_tier: EvidenceTier
    requirements: tuple[EvidenceRequirement, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ScientificClaimSpec schema_version must be 1")
        if not isinstance(self.target_evidence_tier, EvidenceTier):
            raise TypeError("target_evidence_tier must be EvidenceTier")
        object.__setattr__(self, "claim_id", _nonempty("claim_id", self.claim_id))
        object.__setattr__(self, "statement", _nonempty("statement", self.statement))
        object.__setattr__(self, "domain", _nonempty("domain", self.domain))
        object.__setattr__(self, "scope", _nonempty("scope", self.scope))
        object.__setattr__(self, "inference_unit", _nonempty("inference_unit", self.inference_unit))

        requirements = tuple(self.requirements)
        if not requirements:
            raise ValueError("requirements must contain at least one EvidenceRequirement")
        if any(not isinstance(item, EvidenceRequirement) for item in requirements):
            raise TypeError("requirements must contain only EvidenceRequirement objects")
        ids = [item.requirement_id for item in requirements]
        if len(set(ids)) != len(ids):
            raise ValueError("requirements cannot contain duplicate requirement_id values")
        object.__setattr__(self, "requirements", requirements)

        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def claim_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return self.claim_sha256[:16]

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": "neuros.scientific_claim.v1",
            "schema_version": self.schema_version,
            "claim_id": self.claim_id,
            "statement": self.statement,
            "domain": self.domain,
            "scope": self.scope,
            "inference_unit": self.inference_unit,
            "target_evidence_tier": self.target_evidence_tier.value,
            "requirements": [item.to_dict() for item in self.requirements],
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["claim_sha256"] = self.claim_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload


@dataclass(frozen=True, slots=True)
class ClaimEvidenceRef:
    """Immutable reference from a claim to one content-addressed evidence object."""

    evidence_sha256: str
    relation: EvidenceRelation
    evidence_tier: EvidenceTier
    source_kind: str
    source_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ClaimEvidenceRef schema_version must be 1")
        if not isinstance(self.relation, EvidenceRelation):
            raise TypeError("relation must be EvidenceRelation")
        if not isinstance(self.evidence_tier, EvidenceTier):
            raise TypeError("evidence_tier must be EvidenceTier")
        object.__setattr__(
            self, "evidence_sha256", _require_sha256("evidence_sha256", self.evidence_sha256)
        )
        object.__setattr__(self, "source_kind", _nonempty("source_kind", self.source_kind))
        object.__setattr__(self, "source_id", _nonempty("source_id", self.source_id))
        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def reference_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "evidence_sha256": self.evidence_sha256,
            "relation": self.relation.value,
            "evidence_tier": self.evidence_tier.value,
            "source_kind": self.source_kind,
            "source_id": self.source_id,
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["reference_sha256"] = self.reference_sha256
        return payload


@dataclass(frozen=True, slots=True)
class ScientificClaimBundle:
    """Portable manifest binding one claim spec to cited evidence.

    The bundle records evidence labels exactly as supplied. It deliberately does
    not infer an achieved tier, resolve contradictions, or convert cited support
    into truth. Scientific Authority / NSQ remain responsible for qualification.
    """

    claim: ScientificClaimSpec
    evidence: tuple[ClaimEvidenceRef, ...]
    protocol_sha256: str | None = None
    study_sha256s: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("ScientificClaimBundle schema_version must be 1")
        if not isinstance(self.claim, ScientificClaimSpec):
            raise TypeError("claim must be ScientificClaimSpec")

        evidence = tuple(self.evidence)
        if not evidence:
            raise ValueError("evidence must contain at least one ClaimEvidenceRef")
        if any(not isinstance(item, ClaimEvidenceRef) for item in evidence):
            raise TypeError("evidence must contain only ClaimEvidenceRef objects")
        ref_ids = [item.reference_sha256 for item in evidence]
        if len(set(ref_ids)) != len(ref_ids):
            raise ValueError("evidence cannot contain duplicate references")
        object.__setattr__(self, "evidence", evidence)

        if self.protocol_sha256 is not None:
            object.__setattr__(
                self,
                "protocol_sha256",
                _require_sha256("protocol_sha256", self.protocol_sha256),
            )
        study_sha256s = tuple(
            _require_sha256("study_sha256", value) for value in self.study_sha256s
        )
        if len(set(study_sha256s)) != len(study_sha256s):
            raise ValueError("study_sha256s cannot contain duplicates")
        object.__setattr__(self, "study_sha256s", study_sha256s)

        metadata = _freeze_json(self.metadata)
        if not isinstance(metadata, Mapping):
            raise TypeError("metadata must be a mapping")
        object.__setattr__(self, "metadata", metadata)

    @property
    def bundle_sha256(self) -> str:
        return _canonical_sha256(self.to_dict(include_identity=False))

    @property
    def display_fingerprint(self) -> str:
        return self.bundle_sha256[:16]

    def to_dict(self, *, include_identity: bool = True) -> dict[str, Any]:
        payload = {
            "schema": "neuros.scientific_claim_bundle.v1",
            "schema_version": self.schema_version,
            "claim": self.claim.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "protocol_sha256": self.protocol_sha256,
            "study_sha256s": list(self.study_sha256s),
            "metadata": _thaw_json(self.metadata),
        }
        if include_identity:
            payload["bundle_sha256"] = self.bundle_sha256
            payload["display_fingerprint"] = self.display_fingerprint
        return payload
