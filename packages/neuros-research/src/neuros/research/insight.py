"""Lineage-bound cross-pollination artifacts for distributed research."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ._canonical import canonical_sha256, require_nonempty
from .arbiter import PromotionDecision
from .contracts import ExperimentPacket
from .evidence import ExperimentEvidence


@dataclass(frozen=True, slots=True)
class InsightCard:
    """Compact shareable lesson derived from qualified evidence."""

    insight_id: str
    source_experiment_id: str
    source_packet_fingerprint: str
    source_evidence_fingerprint: str
    source_decision_fingerprint: str
    hypothesis: str
    evidence_summary: tuple[str, ...]
    failure_modes: tuple[str, ...]
    suggested_next_tests: tuple[str, ...]

    def __post_init__(self) -> None:
        for name in (
            "insight_id",
            "source_experiment_id",
            "source_packet_fingerprint",
            "source_evidence_fingerprint",
            "source_decision_fingerprint",
            "hypothesis",
        ):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), name=name))
        for name in ("evidence_summary", "failure_modes", "suggested_next_tests"):
            values = tuple(require_nonempty(value, name=name) for value in getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} values must be unique")
            object.__setattr__(self, name, values)

    @classmethod
    def from_promoted(
        cls,
        *,
        insight_id: str,
        packet: ExperimentPacket,
        evidence: ExperimentEvidence,
        decision: PromotionDecision,
        hypothesis: str,
        evidence_summary: tuple[str, ...],
        failure_modes: tuple[str, ...] = (),
        suggested_next_tests: tuple[str, ...] = (),
    ) -> InsightCard:
        if decision.state != "promoted":
            raise ValueError("insight cards may only be published from promoted evidence")
        if decision.experiment_id != packet.experiment_id:
            raise ValueError("decision experiment_id does not match packet")
        if decision.packet_fingerprint != packet.fingerprint:
            raise ValueError("decision packet fingerprint does not match packet")
        if decision.evidence_fingerprint != evidence.fingerprint:
            raise ValueError("decision evidence fingerprint does not match evidence")
        return cls(
            insight_id=insight_id,
            source_experiment_id=packet.experiment_id,
            source_packet_fingerprint=packet.fingerprint,
            source_evidence_fingerprint=evidence.fingerprint,
            source_decision_fingerprint=decision.fingerprint,
            hypothesis=hypothesis,
            evidence_summary=evidence_summary,
            failure_modes=failure_modes,
            suggested_next_tests=suggested_next_tests,
        )

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "insight_id": self.insight_id,
            "source_experiment_id": self.source_experiment_id,
            "source_packet_fingerprint": self.source_packet_fingerprint,
            "source_evidence_fingerprint": self.source_evidence_fingerprint,
            "source_decision_fingerprint": self.source_decision_fingerprint,
            "hypothesis": self.hypothesis,
            "evidence_summary": list(self.evidence_summary),
            "failure_modes": list(self.failure_modes),
            "suggested_next_tests": list(self.suggested_next_tests),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict(include_fingerprint=False))
