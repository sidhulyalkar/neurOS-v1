"""Cross-plane scientific evidence orchestration for neurOS.

This namespace lives in the user-facing ``neuros`` distribution so studies may
compose optional ORION scientific authority with optional ecosystem adapters
without reversing dependency direction in lower-level packages.

The claim contracts are intentionally dependency-light. They describe claim
intent and immutable evidence references; ORION and qualified study systems
remain responsible for earned scientific authority.
"""

from .claims import (
    ClaimEvidenceRef,
    EvidenceRelation,
    EvidenceRequirement,
    EvidenceTier,
    ScientificClaimBundle,
    ScientificClaimSpec,
)

__all__ = [
    "ClaimEvidenceRef",
    "EvidenceRelation",
    "EvidenceRequirement",
    "EvidenceTier",
    "ScientificClaimBundle",
    "ScientificClaimSpec",
]
