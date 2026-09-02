from __future__ import annotations

from dataclasses import replace

import pytest
from neuros.research import (
    AdversarialCheck,
    DatasetAuthority,
    EvaluationAuthority,
    ExperimentEvidence,
    ExperimentPacket,
    Hypothesis,
    MetricGate,
    ResearchAgent,
)


def digest(char: str) -> str:
    return char * 64


def packet() -> ExperimentPacket:
    return ExperimentPacket(
        experiment_id="vocab-probe",
        dataset=DatasetAuthority(
            dataset_id="synthetic",
            source_fingerprint=digest("a"),
            access="synthetic",
        ),
        evaluation=EvaluationAuthority(
            evaluator_id="probe",
            split_fingerprint=digest("b"),
            metric_names=("pearson",),
            evaluation_domains=("validation",),
        ),
        agent=ResearchAgent(
            agent_id="program",
            kind="deterministic_program",
            provider="local",
            model="fixture",
        ),
        hypothesis=Hypothesis(
            hypothesis_id="probe",
            statement="Vocabulary values fail closed.",
            changed_variables=("authority",),
        ),
        code_revision="fixture",
        seeds=(1,),
        information_regimes=("simulation_only",),
        claim_ceiling="software_only",
    )


def test_dataset_access_fails_closed() -> None:
    with pytest.raises(ValueError, match="dataset access"):
        DatasetAuthority(
            dataset_id="x",
            source_fingerprint=digest("a"),
            access="secret"  # type: ignore[arg-type]
        )


def test_optimization_boundary_fails_closed() -> None:
    with pytest.raises(ValueError, match="optimization boundary"):
        replace(packet().evaluation, optimization_boundary="test_oracle")  # type: ignore[arg-type]


def test_agent_kind_fails_closed() -> None:
    with pytest.raises(ValueError, match="agent kind"):
        replace(packet().agent, kind="untracked_agent")  # type: ignore[arg-type]


def test_information_regime_fails_closed() -> None:
    with pytest.raises(ValueError, match="information regimes"):
        replace(packet(), information_regimes=("hidden_target_observed",))  # type: ignore[arg-type]


def test_claim_ceiling_fails_closed() -> None:
    with pytest.raises(ValueError, match="claim ceiling"):
        replace(packet(), claim_ceiling="clinical_truth")  # type: ignore[arg-type]


def test_evidence_status_fails_closed() -> None:
    current = packet()
    with pytest.raises(ValueError, match="evidence status"):
        ExperimentEvidence(
            experiment_id=current.experiment_id,
            packet_fingerprint=current.fingerprint,
            evaluation_fingerprint=current.evaluation.fingerprint,
            status="partially_good",  # type: ignore[arg-type]
            failure_reason="invalid state probe",
        )


def test_check_status_fails_closed() -> None:
    with pytest.raises(ValueError, match="check status"):
        AdversarialCheck("leakage", "probably_pass")  # type: ignore[arg-type]


def test_metric_gate_comparator_and_threshold_fail_closed() -> None:
    with pytest.raises(ValueError, match="comparator"):
        MetricGate("pearson", "validation", "approximately_ge", threshold=0.1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="finite"):
        MetricGate("pearson", "validation", "ge", threshold=float("nan"))
