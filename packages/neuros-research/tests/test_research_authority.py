from __future__ import annotations

from dataclasses import replace

import pytest
from neuros.research import (
    AdversarialCheck,
    DatasetAuthority,
    EvaluationAuthority,
    EvidenceArbiter,
    EvidenceLedger,
    ExperimentEvidence,
    ExperimentPacket,
    Hypothesis,
    InsightCard,
    MetricGate,
    MetricObservation,
    PromotionPolicy,
    ResearchAgent,
)


def digest(char: str) -> str:
    return char * 64


def make_packet(*, experiment_id: str = "exp-vjepa-001") -> ExperimentPacket:
    return ExperimentPacket(
        experiment_id=experiment_id,
        dataset=DatasetAuthority(
            dataset_id="cneuromod-mario-sub01",
            source_fingerprint=digest("a"),
            access="authorized_restricted",
            source_revision="manifest-v1",
        ),
        evaluation=EvaluationAuthority(
            evaluator_id="algonaut-g2",
            split_fingerprint=digest("b"),
            metric_names=("pearson", "rsa_spearman", "runtime_seconds"),
            evaluation_domains=("validation", "g2_ood", "geometry", "operational"),
            forbidden_feedback=(
                "hidden_test_targets",
                "private_leaderboard",
                "g2_ood_for_model_selection",
            ),
        ),
        agent=ResearchAgent(
            agent_id="sol-representation-scout-01",
            kind="frontier_model",
            provider="openai",
            model="gpt-5.6-sol",
            version="2026-08",
            prompt_sha256=digest("c"),
            role="representation_scout",
        ),
        hypothesis=Hypothesis(
            hypothesis_id="h-dense-temporal",
            statement="Dense temporal video tokens improve unseen-level encoding.",
            changed_variables=("representation.pooling",),
        ),
        code_revision="3ec682e10c02c98fced4eff7a39d7da9321b81cf",
        seeds=(7, 19, 41),
        information_regimes=("external_pretrained", "train_only_inductive"),
        claim_ceiling="predictive_ood",
        representation_fingerprint=digest("d"),
        compute_budget={"gpu_hours": 8.0, "max_cache_gb": 100},
    )


def make_evidence(packet: ExperimentPacket, *, pearson: float = 0.23) -> ExperimentEvidence:
    return ExperimentEvidence(
        experiment_id=packet.experiment_id,
        packet_fingerprint=packet.fingerprint,
        evaluation_fingerprint=packet.evaluation.fingerprint,
        status="completed",
        metrics=(
            MetricObservation("pearson", "g2_ood", pearson),
            MetricObservation("rsa_spearman", "geometry", 0.31),
            MetricObservation("runtime_seconds", "operational", 52.0),
        ),
        checks=(
            AdversarialCheck("split_firewall", "pass"),
            AdversarialCheck("temporal_shift_null", "pass"),
            AdversarialCheck("source_revalidation", "pass"),
        ),
        artifact_fingerprints={"prediction_bundle": digest("e")},
    )


def test_packet_identity_is_stable_and_metadata_is_detached() -> None:
    metadata = {"nested": {"items": [1, 2]}}
    packet = replace(make_packet(), metadata=metadata)
    fingerprint = packet.fingerprint
    metadata["nested"]["items"].append(3)
    assert packet.fingerprint == fingerprint
    assert packet.to_dict()["metadata"] == {"nested": {"items": [1, 2]}}


def test_evidence_cannot_bind_to_a_different_packet() -> None:
    packet = make_packet()
    evidence = make_evidence(packet)
    mutated = replace(packet, seeds=(99,))
    policy = PromotionPolicy(
        policy_id="g2",
        metric_gates=(MetricGate("pearson", "g2_ood", "ge", threshold=0.1),),
    )
    decision = EvidenceArbiter().evaluate(mutated, evidence, policy)
    assert decision.state == "non_evaluable"
    assert any("packet fingerprint" in reason for reason in decision.reasons)


def test_failed_evidence_cannot_carry_metrics() -> None:
    packet = make_packet()
    with pytest.raises(ValueError, match="cannot carry promoted metric"):
        ExperimentEvidence(
            experiment_id=packet.experiment_id,
            packet_fingerprint=packet.fingerprint,
            evaluation_fingerprint=packet.evaluation.fingerprint,
            status="failed",
            metrics=(MetricObservation("pearson", "g2_ood", 0.9),),
            failure_reason="out of memory",
        )


def test_vector_policy_requires_metric_and_adversarial_gates() -> None:
    packet = make_packet()
    evidence = make_evidence(packet, pearson=0.23)
    baseline_packet = make_packet(experiment_id="baseline")
    baseline = make_evidence(baseline_packet, pearson=0.20)
    policy = PromotionPolicy(
        policy_id="g2-representation-promotion",
        metric_gates=(
            MetricGate(
                "pearson",
                "g2_ood",
                "ge",
                threshold=0.15,
                min_delta_from_baseline=0.02,
            ),
            MetricGate("rsa_spearman", "geometry", "ge", threshold=0.25),
        ),
        required_checks=("split_firewall", "temporal_shift_null", "source_revalidation"),
    )
    decision = EvidenceArbiter().evaluate(
        packet,
        evidence,
        policy,
        baseline_packet=baseline_packet,
        baseline=baseline,
    )
    assert decision.state == "promoted"


def test_policy_rejects_when_one_evidence_domain_fails() -> None:
    packet = make_packet()
    evidence = make_evidence(packet, pearson=0.21)
    policy = PromotionPolicy(
        policy_id="strict",
        metric_gates=(
            MetricGate("pearson", "g2_ood", "ge", threshold=0.2),
            MetricGate("rsa_spearman", "geometry", "ge", threshold=0.5),
        ),
        required_checks=("split_firewall",),
    )
    decision = EvidenceArbiter().evaluate(packet, evidence, policy)
    assert decision.state == "rejected"
    assert any("rsa_spearman" in reason for reason in decision.reasons)


def test_insight_cards_require_promoted_source_evidence() -> None:
    packet = make_packet()
    evidence = make_evidence(packet)
    policy = PromotionPolicy(
        policy_id="g2",
        metric_gates=(MetricGate("pearson", "g2_ood", "ge", threshold=0.1),),
    )
    decision = EvidenceArbiter().evaluate(packet, evidence, policy)
    insight = InsightCard.from_promoted(
        insight_id="insight-dense-temporal-01",
        packet=packet,
        evidence=evidence,
        decision=decision,
        hypothesis="Dense temporal pooling may carry complementary OOD signal.",
        evidence_summary=("G2 Pearson cleared the frozen gate.",),
        suggested_next_tests=("Repeat on held-out subject.",),
    )
    assert insight.source_experiment_id == packet.experiment_id

    rejected_policy = PromotionPolicy(
        policy_id="too-strict",
        metric_gates=(MetricGate("pearson", "g2_ood", "ge", threshold=0.99),),
    )
    rejected = EvidenceArbiter().evaluate(packet, evidence, rejected_policy)
    with pytest.raises(ValueError, match="promoted"):
        InsightCard.from_promoted(
            insight_id="bad",
            packet=packet,
            evidence=evidence,
            decision=rejected,
            hypothesis="bad",
            evidence_summary=("bad",),
        )


def test_hash_chain_detects_payload_tampering() -> None:
    packet = make_packet()
    evidence = make_evidence(packet)
    ledger = EvidenceLedger()
    ledger.append("packet_registered", packet.experiment_id, packet.to_dict())
    ledger.append("evidence_attached", packet.experiment_id, evidence.to_dict())
    raw = ledger.to_jsonl()
    roundtrip = EvidenceLedger.from_jsonl(raw)
    assert roundtrip.head_hash == ledger.head_hash

    tampered = raw.replace('"value":0.23', '"value":0.99')
    with pytest.raises(ValueError, match="event hash mismatch"):
        EvidenceLedger.from_jsonl(tampered)


def test_dispatch_policy_rejects_allowed_prohibited_overlap() -> None:
    from neuros.research import ExternalDispatchPolicy

    with pytest.raises(ValueError, match="both allowed and prohibited"):
        ExternalDispatchPolicy(
            allowed_payload_classes=("raw_participant_data",),
            prohibited_payload_classes=("raw_participant_data",),
        )


def test_arbiter_rejects_undeclared_metric_authority() -> None:
    packet = make_packet()
    evidence = replace(
        make_evidence(packet),
        metrics=(MetricObservation("secret_lb", "g2_ood", 1.0),),
    )
    policy = PromotionPolicy(
        policy_id="bad",
        metric_gates=(MetricGate("secret_lb", "g2_ood", "ge", threshold=0.1),),
    )
    decision = EvidenceArbiter().evaluate(packet, evidence, policy)
    assert decision.state == "non_evaluable"
    assert any("not declared" in reason or "outside" in reason for reason in decision.reasons)


def test_registry_requires_explicit_parent_lineage() -> None:
    from neuros.research import ResearchRegistry

    registry = ResearchRegistry()
    parent = make_packet(experiment_id="parent")
    child = replace(
        make_packet(experiment_id="child"),
        hypothesis=Hypothesis(
            hypothesis_id="child-h",
            statement="Apply a promoted temporal insight.",
            changed_variables=("temporal_pooling",),
            parent_experiment_ids=("parent",),
        ),
    )
    with pytest.raises(ValueError, match="parent experiments"):
        registry.register_packet(child)
    registry.register_packet(parent)
    registry.register_packet(child)
    assert registry.experiment_ids == ("parent", "child")


def test_registry_owns_adjudication_and_insight_chain() -> None:
    from neuros.research import ResearchRegistry

    packet = make_packet()
    evidence = make_evidence(packet)
    policy = PromotionPolicy(
        policy_id="g2",
        metric_gates=(MetricGate("pearson", "g2_ood", "ge", threshold=0.1),),
    )

    registry = ResearchRegistry()
    registry.register_packet(packet)
    registry.attach_evidence(evidence)
    decision = registry.adjudicate(packet.experiment_id, policy)
    assert decision.state == "promoted"

    insight = InsightCard.from_promoted(
        insight_id="share-1",
        packet=packet,
        evidence=evidence,
        decision=decision,
        hypothesis="Dense temporal tokens survived the frozen G2 gate.",
        evidence_summary=("G2 gate passed.",),
    )
    registry.publish_insight(insight)

    assert len(registry.ledger.events) == 4
    registry.ledger.verify()
    assert registry.fingerprint == registry.ledger.head_hash


def test_registry_baseline_adjudication_uses_registered_evidence() -> None:
    from neuros.research import ResearchRegistry

    baseline_packet = make_packet(experiment_id="baseline")
    candidate_packet = make_packet(experiment_id="candidate")
    baseline_evidence = make_evidence(baseline_packet, pearson=0.20)
    candidate_evidence = make_evidence(candidate_packet, pearson=0.23)
    policy = PromotionPolicy(
        policy_id="delta-gate",
        metric_gates=(
            MetricGate(
                "pearson",
                "g2_ood",
                "ge",
                threshold=0.15,
                min_delta_from_baseline=0.02,
            ),
        ),
    )

    registry = ResearchRegistry()
    registry.register_packet(baseline_packet)
    registry.register_packet(candidate_packet)
    registry.attach_evidence(baseline_evidence)
    registry.attach_evidence(candidate_evidence)
    decision = registry.adjudicate(
        candidate_packet.experiment_id,
        policy,
        baseline_experiment_id=baseline_packet.experiment_id,
    )
    assert decision.state == "promoted"


def test_registry_cannot_adjudicate_against_missing_baseline_evidence() -> None:
    from neuros.research import ResearchRegistry

    baseline_packet = make_packet(experiment_id="baseline")
    candidate_packet = make_packet(experiment_id="candidate")
    candidate_evidence = make_evidence(candidate_packet, pearson=0.23)
    policy = PromotionPolicy(
        policy_id="delta-gate",
        metric_gates=(
            MetricGate(
                "pearson",
                "g2_ood",
                "ge",
                min_delta_from_baseline=0.02,
            ),
        ),
    )

    registry = ResearchRegistry()
    registry.register_packet(baseline_packet)
    registry.register_packet(candidate_packet)
    registry.attach_evidence(candidate_evidence)
    with pytest.raises(ValueError, match="baseline"):
        registry.adjudicate(
            candidate_packet.experiment_id,
            policy,
            baseline_experiment_id=baseline_packet.experiment_id,
        )
