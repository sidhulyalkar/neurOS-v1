"""v1 release and empirical-evidence closure status."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class EvidenceRequirementState(str, Enum):
    IMPLEMENTED = "implemented"
    VERIFIED = "verified"
    PENDING = "pending"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EvidenceClosureRequirement:
    requirement_id: str
    category: str
    description: str
    state: EvidenceRequirementState
    artifact_fingerprints: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def satisfied(self) -> bool:
        return self.state in {
            EvidenceRequirementState.IMPLEMENTED,
            EvidenceRequirementState.VERIFIED,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "requirement_id": self.requirement_id,
            "category": self.category,
            "description": self.description,
            "state": self.state.value,
            "satisfied": self.satisfied,
            "artifact_fingerprints": list(self.artifact_fingerprints),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class V1EvidenceStatus:
    package_version: str
    requirements: tuple[EvidenceClosureRequirement, ...]

    @property
    def software_contract_ready(self) -> bool:
        software = [item for item in self.requirements if item.category == "software"]
        return bool(software) and all(item.satisfied for item in software)

    @property
    def empirical_evidence_complete(self) -> bool:
        empirical = [item for item in self.requirements if item.category == "empirical"]
        return bool(empirical) and all(item.satisfied for item in empirical)

    @property
    def pending_empirical_requirements(self) -> tuple[str, ...]:
        return tuple(
            item.requirement_id
            for item in self.requirements
            if item.category == "empirical" and not item.satisfied
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "package_version": self.package_version,
            "software_contract_ready": self.software_contract_ready,
            "empirical_evidence_complete": self.empirical_evidence_complete,
            "pending_empirical_requirements": list(self.pending_empirical_requirements),
            "claim_boundary": (
                "Software readiness certifies schemas, migrations, falsification gates, tutorial "
                "execution, and reproduction machinery. It does not certify empirical neural "
                "mechanisms until real evidence artifacts are attached."
            ),
            "requirements": [item.to_dict() for item in self.requirements],
        }


def default_v1_evidence_status() -> V1EvidenceStatus:
    """Return the repository-declared v1 status without inventing real evidence."""

    software = (
        EvidenceClosureRequirement(
            "versioned-schema-freeze",
            "software",
            "Frozen manifest and artifact schemas with backwards migration.",
            EvidenceRequirementState.IMPLEMENTED,
        ),
        EvidenceClosureRequirement(
            "scientific-vs-run-identity",
            "software",
            "Deterministic scientific fingerprints are separate from execution run hashes.",
            EvidenceRequirementState.IMPLEMENTED,
        ),
        EvidenceClosureRequirement(
            "independent-reproduction-contract",
            "software",
            "Independent reruns are compared by qualitative decision and preregistered tolerances.",
            EvidenceRequirementState.IMPLEMENTED,
        ),
        EvidenceClosureRequirement(
            "cpu-tutorial-evidence-ci",
            "software",
            "Maintained CPU tutorials execute in evidence CI rather than receiving JSON-only checks.",
            EvidenceRequirementState.IMPLEMENTED,
        ),
        EvidenceClosureRequirement(
            "synthetic-falsification-suite",
            "software",
            "Known-positive and known-negative scientific gates remain executable.",
            EvidenceRequirementState.IMPLEMENTED,
        ),
    )
    empirical = tuple(
        EvidenceClosureRequirement(requirement_id, "empirical", description, EvidenceRequirementState.PENDING)
        for requirement_id, description in (
            (
                "real-model-faithfulness-pack",
                "At least one held-out real-model circuit-faithfulness evidence pack.",
            ),
            (
                "real-neural-factorial-study",
                "Matched Transformer x SSM crossed with Event x Relative-ISI on real neural data.",
            ),
            (
                "held-out-causal-correspondence",
                "At least one held-out causal feature correspondence from the real study.",
            ),
            (
                "multi-seed-correspondence-replication",
                "At least one correspondence replicated across independent model-training seeds.",
            ),
            (
                "subject-session-uncertainty",
                "Subject/session-level uncertainty where the chosen dataset supports that claim.",
            ),
            (
                "cross-context-causal-transfer",
                "At least one real cross-session or cross-dataset causal transfer study.",
            ),
            (
                "real-dose-response",
                "At least one real intervention dose-response study.",
            ),
            (
                "manifold-aware-control",
                "At least one stronger empirical/conditional/manifold-aware substitution control.",
            ),
            (
                "independent-artifact-reproduction",
                "At least one artifact family reproduced through an independent execution path.",
            ),
            (
                "published-negative-results",
                "Negative discovery/alignment/correspondence/replication results retained and published.",
            ),
        )
    )
    return V1EvidenceStatus(package_version="1.0.0", requirements=(*software, *empirical))
