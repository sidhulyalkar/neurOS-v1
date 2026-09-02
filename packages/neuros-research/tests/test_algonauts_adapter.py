from __future__ import annotations

import pytest
from neuros.research import AlgonautsAuthoritySpec, Hypothesis, ResearchAgent


def agent() -> ResearchAgent:
    return ResearchAgent(
        agent_id="representation-scout",
        kind="frontier_model",
        provider="example",
        model="model",
    )


def hypothesis() -> Hypothesis:
    return Hypothesis(
        hypothesis_id="dense-temporal",
        statement="Dense temporal features improve unseen-level encoding.",
        changed_variables=("representation",),
    )


def spec(*, split_sha256: str = "b" * 64) -> AlgonautsAuthoritySpec:
    return AlgonautsAuthoritySpec(
        dataset_id="cneuromod-mario-sub01",
        source_sha256="a" * 64,
        source_revision="manifest-v1",
        split_sha256=split_sha256,
        evaluator_id="g2-neural-geometry-v1",
    )


def test_adapter_binds_temporal_metadata_into_evaluation_identity() -> None:
    first = spec().packet(
        experiment_id="first",
        agent=agent(),
        hypothesis=hypothesis(),
        code_revision="revision",
        seeds=(7,),
        evaluation_metadata={
            "temporal_alignment": {
                "model": "SPM",
                "hrf_oversampling": 50,
                "sample_times": "TR midpoints",
            }
        },
    )
    second = spec().packet(
        experiment_id="second",
        agent=agent(),
        hypothesis=hypothesis(),
        code_revision="revision",
        seeds=(7,),
        evaluation_metadata={
            "temporal_alignment": {
                "model": "SPM",
                "hrf_oversampling": 25,
                "sample_times": "TR midpoints",
            }
        },
    )
    assert first.evaluation.fingerprint != second.evaluation.fingerprint


def test_adapter_keeps_representation_out_of_evaluation_identity() -> None:
    shared = {
        "experiment_id": "candidate-a",
        "agent": agent(),
        "hypothesis": hypothesis(),
        "code_revision": "revision",
        "seeds": (7,),
        "evaluation_metadata": {"temporal_alignment": {"hrf_oversampling": 50}},
    }
    first = spec().packet(representation_sha256="c" * 64, **shared)
    shared["experiment_id"] = "candidate-b"
    second = spec().packet(representation_sha256="d" * 64, **shared)

    assert first.evaluation.fingerprint == second.evaluation.fingerprint
    assert first.representation_fingerprint != second.representation_fingerprint
    assert first.fingerprint != second.fingerprint


def test_adapter_forbids_ood_feedback_for_model_selection() -> None:
    packet = spec().packet(
        experiment_id="candidate",
        agent=agent(),
        hypothesis=hypothesis(),
        code_revision="revision",
        seeds=(7,),
    )
    forbidden = set(packet.evaluation.forbidden_feedback)
    assert "g2_ood_for_model_selection" in forbidden
    assert "g3_cross_game_for_model_selection" in forbidden
    assert "g4_held_subject_for_model_selection" in forbidden


def test_adapter_access_still_fails_closed_at_runtime() -> None:
    invalid = AlgonautsAuthoritySpec(
        dataset_id="x",
        source_sha256="a" * 64,
        source_revision="v1",
        split_sha256="b" * 64,
        evaluator_id="evaluator",
        access="send_anywhere",
    )
    with pytest.raises(ValueError, match="dataset access"):
        invalid.dataset_authority()
