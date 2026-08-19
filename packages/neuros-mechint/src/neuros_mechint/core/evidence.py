"""Repository-aligned evidence tiers for mechanistic claims."""

from __future__ import annotations

from enum import IntEnum


class EvidenceTier(IntEnum):
    """Weakest evidence tier that directly supports a result.

    Values intentionally match the repository-wide evidence ladder documented
    in ``CONTRIBUTING.md``. A method's maturity and a result's evidence tier are
    separate concepts: a mature method can still be run on weak evidence.
    """

    UNIT = 1
    CONTRACT = 2
    INTEGRATION = 3
    REPLAY = 4
    SCIENTIFIC_SYNTHETIC = 5
    REAL_DATASET = 6
    HARDWARE_QUALIFICATION = 7
    CLOSED_LOOP_QUALIFICATION = 8
    CLINICAL_EVIDENCE = 9

    @property
    def label(self) -> str:
        return {
            EvidenceTier.UNIT: "unit",
            EvidenceTier.CONTRACT: "contract",
            EvidenceTier.INTEGRATION: "integration",
            EvidenceTier.REPLAY: "replay",
            EvidenceTier.SCIENTIFIC_SYNTHETIC: "scientific_synthetic",
            EvidenceTier.REAL_DATASET: "real_dataset",
            EvidenceTier.HARDWARE_QUALIFICATION: "hardware_qualification",
            EvidenceTier.CLOSED_LOOP_QUALIFICATION: "closed_loop_qualification",
            EvidenceTier.CLINICAL_EVIDENCE: "clinical_evidence",
        }[self]

    @classmethod
    def coerce(cls, value: EvidenceTier | int | str) -> EvidenceTier:
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        for tier in cls:
            if normalized in {tier.label, tier.name.lower(), str(int(tier))}:
                return tier
        raise ValueError(f"unknown evidence tier: {value!r}")
