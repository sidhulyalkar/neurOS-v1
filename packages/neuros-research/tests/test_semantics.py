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
        "claim_relation": "matched_control",
        "control_description": "Capacity- and dimension-matched representation control.",
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


def _set_absolute_mse_claim(payload: dict[str, object]) -> None:
    payload["statement"] = "PCA reaches low development validation MSE."
    payload["rationale"] = "This is an absolute diagnostic threshold, not a comparison."
    payload["development_metrics"] = ["validation_mse", "runtime_seconds"]
    payload["primary_metric"] = "validation_mse"
    payload["claim_relation"] = "absolute"
    payload["control_description"] = ""
    payload["supports_if"] = [
        {
            "metric": "validation_mse",
            "operator": "<=",
            "threshold": 0.35,
            "rationale": "predeclared absolute development threshold",
        }
    ]
    payload["rejects_if"] = [
        {
            "metric": "validation_mse",
            "operator": ">=",
            "threshold": 0.45,
            "rationale": "predeclared absolute rejection threshold",
        }
    ]
    payload["falsification_test"] = "Reject if validation MSE is at least 0.45."


def test_semantic_proposal_accepts_directional_gap() -> None:
    proposal = parse(proposal_payload())
    assert proposal.primary_metric == "validation_pearson_delta"
    assert proposal.claim_relation == "matched_control"
    assert proposal.supports_if[0].threshold == 0.02
    assert proposal.rejects_if[0].threshold == 0.005
    assert len(proposal.fingerprint) == 64


def test_metric_registry_exposes_claim_relation_authority() -> None:
    registry = semantics.metric_registry_payload()
    assert registry["validation_mse"]["claim_relation"] == "absolute"
    assert registry["validation_mse_reduction"]["claim_relation"] == "matched_control"
    assert registry["temporal_shift_drop"]["claim_relation"] == "temporal_null"
    assert registry["complementarity_score"]["claim_relation"] == "complementarity"


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
    payload["claim_relation"] = "control_sweep"
    payload["control_description"] = "Predeclared ridge-hyperparameter control sweep."
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


def test_comparative_mse_claim_cannot_use_absolute_mse_metric() -> None:
    payload = proposal_payload()
    _set_absolute_mse_claim(payload)
    payload["statement"] = "PCA will achieve lower validation MSE than a matched-control baseline."
    payload["falsification_test"] = "Reject if PCA does not beat the matched-control baseline."
    with pytest.raises(ValueError, match="explicit comparative claim requires"):
        parse(payload)


def test_comparative_pearson_claim_cannot_use_absolute_pearson_metric() -> None:
    payload = proposal_payload()
    payload["statement"] = "Ridge will achieve higher validation Pearson than an untrained baseline."
    payload["development_metrics"] = ["validation_pearson", "cache_gb"]
    payload["primary_metric"] = "validation_pearson"
    payload["claim_relation"] = "absolute"
    payload["control_description"] = ""
    payload["supports_if"] = [
        {
            "metric": "validation_pearson",
            "operator": ">=",
            "threshold": 0.25,
            "rationale": "absolute threshold",
        }
    ]
    payload["rejects_if"] = [
        {
            "metric": "validation_pearson",
            "operator": "<=",
            "threshold": 0.10,
            "rationale": "absolute rejection",
        }
    ]
    payload["falsification_test"] = "Reject if ridge does not improve over the baseline."
    with pytest.raises(ValueError, match="explicit comparative claim requires"):
        parse(payload)


def test_comparative_mse_claim_passes_with_matched_reduction_metric() -> None:
    payload = proposal_payload()
    payload["statement"] = "PCA will achieve lower validation MSE than a matched-control baseline."
    payload["development_metrics"] = ["validation_mse_reduction", "runtime_seconds"]
    payload["primary_metric"] = "validation_mse_reduction"
    payload["claim_relation"] = "matched_control"
    payload["control_description"] = "Capacity- and dimension-matched random projection baseline."
    payload["supports_if"] = [
        {
            "metric": "validation_mse_reduction",
            "operator": ">=",
            "threshold": 0.02,
            "rationale": "meaningful MSE reduction",
        }
    ]
    payload["rejects_if"] = [
        {
            "metric": "validation_mse_reduction",
            "operator": "<=",
            "threshold": 0.005,
            "rationale": "practical-null reduction",
        }
    ]
    payload["falsification_test"] = "Reject if reduction versus the baseline is at most 0.005."
    proposal = parse(payload)
    assert proposal.claim_relation == "matched_control"


def test_claim_relation_must_match_primary_metric_registry() -> None:
    payload = proposal_payload()
    payload["claim_relation"] = "absolute"
    payload["control_description"] = ""
    with pytest.raises(ValueError, match="contradicts primary metric"):
        parse(payload)


def test_nonabsolute_claim_requires_explicit_control_description() -> None:
    payload = proposal_payload()
    payload["control_description"] = ""
    with pytest.raises(ValueError, match="require a non-empty control_description"):
        parse(payload)


def test_absolute_claim_without_comparison_is_allowed() -> None:
    payload = proposal_payload()
    _set_absolute_mse_claim(payload)
    proposal = parse(payload)
    assert proposal.claim_relation == "absolute"
    assert proposal.control_description == ""


def test_synthesis_stopping_authority_is_deterministic_and_candidate_independent() -> None:
    payload = {
        "priority_queue": ["a", "b"],
        "rounds": [{"round": 1, "candidate_ids": ["a", "b"], "reason": "cheap first"}],
        "stopping_rule": "Stop the whole program if candidate a fails.",
    }
    normalized = semantics.enforce_independent_synthesis_stopping_policy(payload)
    assert normalized["model_stopping_note"] == payload["stopping_rule"]
    assert normalized["stopping_rule"] == semantics.INDEPENDENT_CANDIDATE_STOPPING_RULE
    assert normalized["stopping_rule_authority"] == "deterministic_independent_candidate_policy"
    assert "does not terminate unrelated candidates" in normalized["stopping_rule"]


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
    assert payload["metadata"]["primary_metric"] == "validation_pearson_delta"
    assert payload["metadata"]["claim_relation"] == "matched_control"
    assert "matched" in payload["metadata"]["control_description"].lower()
    assert payload["evaluation"]["split_fingerprint"] == "1" * 64
