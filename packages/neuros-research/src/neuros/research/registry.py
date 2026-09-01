"""Semantic experiment registry backed by the tamper-evident ledger."""

from __future__ import annotations

from .arbiter import PromotionDecision
from .contracts import ExperimentPacket
from .evidence import ExperimentEvidence
from .insight import InsightCard
from .ledger import EvidenceLedger


class ResearchRegistry:
    """Fail-closed experiment DAG and evidence attachment authority."""

    def __init__(self) -> None:
        self._packets: dict[str, ExperimentPacket] = {}
        self._evidence: dict[str, ExperimentEvidence] = {}
        self._decisions: dict[str, PromotionDecision] = {}
        self._insights: dict[str, InsightCard] = {}
        self._ledger = EvidenceLedger()

    @property
    def ledger(self) -> EvidenceLedger:
        return self._ledger

    @property
    def experiment_ids(self) -> tuple[str, ...]:
        return tuple(self._packets)

    @property
    def fingerprint(self) -> str:
        return self._ledger.head_hash

    def register_packet(self, packet: ExperimentPacket) -> None:
        if packet.experiment_id in self._packets:
            raise ValueError(f"experiment_id {packet.experiment_id!r} is already registered")
        missing = [
            parent
            for parent in packet.hypothesis.parent_experiment_ids
            if parent not in self._packets
        ]
        if missing:
            raise ValueError(
                "parent experiments must be registered before child experiments: "
                + ", ".join(missing)
            )
        self._packets[packet.experiment_id] = packet
        self._ledger.append("packet_registered", packet.experiment_id, packet.to_dict())

    def attach_evidence(self, evidence: ExperimentEvidence) -> None:
        packet = self._packets.get(evidence.experiment_id)
        if packet is None:
            raise ValueError(f"experiment {evidence.experiment_id!r} is not registered")
        if evidence.experiment_id in self._evidence:
            raise ValueError(f"evidence already attached for {evidence.experiment_id!r}")
        if evidence.packet_fingerprint != packet.fingerprint:
            raise ValueError("evidence packet fingerprint does not match registered packet")
        if evidence.evaluation_fingerprint != packet.evaluation.fingerprint:
            raise ValueError("evidence evaluation fingerprint does not match registered authority")
        self._evidence[evidence.experiment_id] = evidence
        self._ledger.append("evidence_attached", evidence.experiment_id, evidence.to_dict())

    def attach_decision(self, decision: PromotionDecision) -> None:
        packet = self._packets.get(decision.experiment_id)
        evidence = self._evidence.get(decision.experiment_id)
        if packet is None or evidence is None:
            raise ValueError("decision requires registered packet and attached evidence")
        if decision.experiment_id in self._decisions:
            raise ValueError(f"decision already attached for {decision.experiment_id!r}")
        if decision.packet_fingerprint != packet.fingerprint:
            raise ValueError("decision packet fingerprint does not match registered packet")
        if decision.evidence_fingerprint != evidence.fingerprint:
            raise ValueError("decision evidence fingerprint does not match attached evidence")
        self._decisions[decision.experiment_id] = decision
        self._ledger.append("decision_attached", decision.experiment_id, decision.to_dict())

    def publish_insight(self, insight: InsightCard) -> None:
        if insight.insight_id in self._insights:
            raise ValueError(f"insight_id {insight.insight_id!r} is already registered")
        packet = self._packets.get(insight.source_experiment_id)
        evidence = self._evidence.get(insight.source_experiment_id)
        decision = self._decisions.get(insight.source_experiment_id)
        if packet is None or evidence is None or decision is None:
            raise ValueError("insight requires registered packet, evidence, and decision")
        if decision.state != "promoted":
            raise ValueError("insight may only be published from a promoted decision")
        if insight.source_packet_fingerprint != packet.fingerprint:
            raise ValueError("insight packet fingerprint does not match registered packet")
        if insight.source_evidence_fingerprint != evidence.fingerprint:
            raise ValueError("insight evidence fingerprint does not match registered evidence")
        if insight.source_decision_fingerprint != decision.fingerprint:
            raise ValueError("insight decision fingerprint does not match registered decision")
        self._insights[insight.insight_id] = insight
        self._ledger.append(
            "insight_published",
            insight.source_experiment_id,
            insight.to_dict(),
        )
