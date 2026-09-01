from __future__ import annotations

import importlib

import pytest

semantics = importlib.import_module("neuros.research.semantics")

ALLOWED_PAYLOADS = ("source_code", "aggregate_metrics", "public_metadata")
ALLOWED_METRICS = tuple(sorted(semantics.ALGORITHMIC_METRIC_REGISTRY))


def proposal_payload() -> dict[str, object]:
    return {
        "candidate_id": "typed-control",
        "title": "Typed validation control",
        "statement": "The matched intervention improves development validation correlation.",
        "rationale": "A predeclared matched control separates representation effects from capacity.",
        "family": "representation",
        "changed_variables": ["representation.pooling"],
        "required_payload_classes": ["source_code", "aggregate_metrics"],
        "development_metrics": ["validation_pearson_delta", "runtime_seconds"],
        "primary_metric": "validation_pearson_delta",
        "supports_if": [
            {
                "metric": "validation_pearson_delta",
                "operator": ">=",
                "threshold": 0.02,
                "rationale": "practically meaningful development gain",
            }
        ],
        "rejects_if": [
            {
                "metric": "validation_pearson_delta",
                "operator": "<=",
                "threshold": 0.005,
                "rationale": "practical-null development gain",
            }
        ],
        "falsification_test": "Reject if the matched gain is at most 0.005.",
        "estimated_compute_tier": "low",
        "expected_failure_modes": ["The representation may be redundant with the control."],
    }


def parse(payload: dict[str, object]):
    return semantics.SemanticResearchProposal.from_dict(
        payload,
        allowed_payload_classes=ALLOWED_PAYLOADS,
        allowed_development_metrics=ALLOWED_METRICS,
    )


def test_semantic_proposal_accepts_directional_gap() -> None:
    proposal = parse(proposal_payload())
    assert proposal.primary_metric == "validation_pearson_delta"
    assert proposal.supports_if[0].threshold == 0.02
    assert proposal.rejects_if[0].threshold == 0.005
    assert len(proposal.fingerprint) == 64


def test_support_direction_inversion_is_rejected() -> None:
    payload = proposal_payload()
    payload["supports_if"] = [
        {
            "metric": "validation_pearson_delta",
            "operator": "<=",
            "threshold": 0.005,
            "rationale": "wrong direction",
        }
    ]
    with pytest.raises(ValueError, match="contradicts registered direction"):
        parse(payload)


def test_rejection_direction_inversion_is_rejected() -> None:
    payload = proposal_payload()
    payload["rejects_if"] = [
        {
            "metric": "validation_pearson_delta",
            "operator": ">=",
            "threshold": 0.02,
            "rationale": "wrong direction",
        }
    ]
    with pytest.raises(ValueError, match="contradicts registered direction"):
        parse(payload)


def test_overlapping_support_and_rejection_regions_are_rejected() -> None:
    payload = proposal_payload()
    payload["supports_if"] = [
        {
            "metric": "validation_pearson_delta",
            "operator": ">=",
            "threshold": 0.005,
            "rationale": "overlaps rejection",
        }
    ]
    with pytest.raises(ValueError, match="regions overlap"):
        parse(payload)


def test_primary_metric_requires_both_decision_regions() -> None:
    payload = proposal_payload()
    payload["development_metrics"] = [
        "validation_pearson_delta",
        "runtime_seconds",
        "validation_mse",
    ]
    payload["rejects_if"] = [
        {
            "metric": "runtime_seconds",
            "operator": ">=",
            "threshold": 100.0,
            "rationale": "runtime rejection only",
        }
    ]
    with pytest.raises(ValueError, match="primary_metric must have a rejection criterion"):
        parse(payload)


def test_unknown_semantic_metric_is_rejected() -> None:
    payload = proposal_payload()
    payload["development_metrics"] = ["validation_pearson_delta", "made_up_metric"]
    with pytest.raises(ValueError, match="outside frozen menu"):
        parse(payload)


def test_nonfinite_threshold_is_rejected() -> None:
    payload = proposal_payload()
    payload["supports_if"] = [
        {
            "metric": "validation_pearson_delta",
            "operator": ">=",
            "threshold": float("nan"),
            "rationale": "not a real threshold",
        }
    ]
    with pytest.raises(ValueError, match="must be finite"):
        parse(payload)


def test_neutrality_metric_uses_lower_is_supportive_semantics() -> None:
    payload = proposal_payload()
    payload["statement"] = "Ridge sensitivity remains inside a practical neutrality band."
    payload["development_metrics"] = ["validation_pearson_span", "runtime_seconds"]
    payload["primary_metric"] = "validation_pearson_span"
    payload["supports_if"] = [
        {
            "metric": "validation_pearson_span",
            "operator": "<=",
            "threshold": 0.01,
            "rationale": "small score span supports neutrality",
        }
    ]
    payload["rejects_if"] = [
        {
            "metric": "validation_pearson_span",
            "operator": ">=",
            "threshold": 0.02,
            "rationale": "large score span rejects neutrality",
        }
    ]
    proposal = parse(payload)
    assert proposal.primary_metric == "validation_pearson_span"


def test_execution_binding_fails_closed_without_real_hashes() -> None:
    with pytest.raises(ValueError, match="dataset_source_fingerprint"):
        semantics.ExecutionBinding(
            dataset_id="algonauts",
            dataset_source_fingerprint="not-a-hash",
            dataset_source_revision="dataset-v1",
            evaluator_id="frozen-dev",
            split_fingerprint="1" * 64,
            preprocessing_fingerprint="2" * 64,
            evaluation_domains=("development",),
            runner_entrypoint="python -m runner",
            code_revision="a" * 40,
            seeds=(17,),
        )


def test_materialized_packet_is_g1_only_and_binds_real_identities() -> None:
    proposal = parse(proposal_payload())
    binding = semantics.ExecutionBinding(
        dataset_id="algonauts-authorized-view",
        dataset_source_fingerprint="0" * 64,
        dataset_source_revision="dataset-v1",
        evaluator_id="frozen-development-evaluator",
        split_fingerprint="1" * 64,
        preprocessing_fingerprint="2" * 64,
        evaluation_domains=("development",),
        runner_entrypoint="python -m neuros.algonauts.execute",
        code_revision="a" * 40,
        seeds=(17, 29),
    )
    packet = semantics.materialize_g1_packet(
        proposal,
        binding,
        proposer_model="nvidia/nemotron-3-super-120b-a12b",
        proposer_prompt_sha256="3" * 64,
    )
    payload = packet.to_dict()
    assert payload["claim_ceiling"] == "predictive_id"
    assert payload["information_regimes"] == ["train_only_inductive"]
    assert payload["metadata"]["gate"] == "G1"
    assert payload["metadata"]["preprocessing_fingerprint"] == "2" * 64
    assert payload["evaluation"]["split_fingerprint"] == "1" * 64
