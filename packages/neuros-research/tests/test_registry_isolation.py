from neuros.research import (
    DatasetAuthority,
    EvaluationAuthority,
    ExperimentPacket,
    Hypothesis,
    ResearchAgent,
    ResearchRegistry,
)


def test_exported_ledger_is_a_detached_snapshot() -> None:
    packet = ExperimentPacket(
        experiment_id="registry-isolation",
        dataset=DatasetAuthority(
            dataset_id="synthetic",
            source_fingerprint="a" * 64,
            access="synthetic",
        ),
        evaluation=EvaluationAuthority(
            evaluator_id="synthetic-referee",
            split_fingerprint="b" * 64,
            metric_names=("score",),
            evaluation_domains=("validation",),
        ),
        agent=ResearchAgent(
            agent_id="program",
            kind="deterministic_program",
            provider="local",
            model="fixture",
        ),
        hypothesis=Hypothesis(
            hypothesis_id="isolation",
            statement="Exported ledger mutation cannot alter registry history.",
            changed_variables=("ledger_snapshot",),
        ),
        code_revision="fixture",
        seeds=(1,),
        information_regimes=("simulation_only",),
        claim_ceiling="software_only",
    )

    registry = ResearchRegistry()
    registry.register_packet(packet)
    authoritative_hash = registry.fingerprint

    snapshot = registry.ledger
    snapshot.append(
        "packet_registered",
        "forged-experiment",
        {"note": "this event exists only in the detached snapshot"},
    )

    assert snapshot.head_hash != authoritative_hash
    assert registry.fingerprint == authoritative_hash
    assert len(registry.ledger.events) == 1
