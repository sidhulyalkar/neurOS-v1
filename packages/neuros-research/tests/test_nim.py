from __future__ import annotations

import json

import pytest

from neuros.research.nim import (
    DEFAULT_NVIDIA_MODEL_PREFERENCES,
    NvidiaNimClient,
    ResearchProposal,
    _extract_json_object,
    _validate_endpoint,
    frozen_public_context,
    parse_proposals,
)


ALLOWED_PAYLOADS = ("source_code", "schemas", "aggregate_metrics", "public_metadata")
ALLOWED_METRICS = (
    "validation_pearson",
    "validation_mse",
    "rsa_spearman",
    "temporal_shift_null",
)


def proposal_payload(candidate_id: str = "rep-temporal-01") -> dict[str, object]:
    return {
        "candidate_id": candidate_id,
        "title": "Dense temporal representation control",
        "statement": "Dense temporal tokens improve validation encoding and geometry alignment.",
        "rationale": "The same frozen visual encoder may expose dynamics lost by aggressive pooling.",
        "family": "representation",
        "changed_variables": ["representation.pooling", "representation.frame_step"],
        "required_payload_classes": ["source_code", "aggregate_metrics"],
        "development_metrics": ["validation_pearson", "rsa_spearman"],
        "falsification_test": "Reject if validation gain disappears under matched feature dimension.",
        "estimated_compute_tier": "medium",
        "expected_failure_modes": ["Redundant temporal features may increase variance."],
    }


def test_nvidia_endpoint_is_credential_scoped() -> None:
    assert _validate_endpoint("https://integrate.api.nvidia.com/v1") == (
        "https://integrate.api.nvidia.com/v1"
    )
    with pytest.raises(ValueError, match="HTTPS"):
        _validate_endpoint("http://integrate.api.nvidia.com/v1")
    with pytest.raises(ValueError, match="may only be sent"):
        _validate_endpoint("https://example.com/v1")


def test_json_extraction_tolerates_reasoning_prefix() -> None:
    payload = _extract_json_object('analysis text\n{"candidates": [{"candidate_id": "a"}]}\n')
    assert payload["candidates"][0]["candidate_id"] == "a"


def test_proposal_rejects_frozen_authority_mutation() -> None:
    payload = proposal_payload()
    payload["changed_variables"] = ["split.ood_levels"]
    with pytest.raises(ValueError, match="frozen authority"):
        ResearchProposal.from_dict(
            payload,
            allowed_payload_classes=ALLOWED_PAYLOADS,
            allowed_development_metrics=ALLOWED_METRICS,
        )


def test_proposal_rejects_forbidden_payload_class() -> None:
    payload = proposal_payload()
    payload["required_payload_classes"] = ["raw_participant_data"]
    with pytest.raises(ValueError, match="outside dispatch policy"):
        ResearchProposal.from_dict(
            payload,
            allowed_payload_classes=ALLOWED_PAYLOADS,
            allowed_development_metrics=ALLOWED_METRICS,
        )


def test_proposal_rejects_forbidden_feedback_language() -> None:
    payload = proposal_payload()
    payload["rationale"] = "Tune this against the private leaderboard."
    with pytest.raises(ValueError, match="forbidden feedback"):
        ResearchProposal.from_dict(
            payload,
            allowed_payload_classes=ALLOWED_PAYLOADS,
            allowed_development_metrics=ALLOWED_METRICS,
        )


def test_parse_proposals_requires_unique_ids() -> None:
    payload = {"candidates": [proposal_payload("same") for _ in range(3)]}
    with pytest.raises(ValueError, match="unique"):
        parse_proposals(
            payload,
            allowed_payload_classes=ALLOWED_PAYLOADS,
            allowed_development_metrics=ALLOWED_METRICS,
        )


def test_model_selection_prefers_approved_nvidia_reasoners() -> None:
    available = (
        "other/model",
        DEFAULT_NVIDIA_MODEL_PREFERENCES[2],
        DEFAULT_NVIDIA_MODEL_PREFERENCES[0],
    )
    assert NvidiaNimClient.select_models(available, count=2) == (
        DEFAULT_NVIDIA_MODEL_PREFERENCES[0],
        DEFAULT_NVIDIA_MODEL_PREFERENCES[2],
    )


def test_public_context_is_detached_and_stable() -> None:
    source = {"metrics": ["validation_pearson"], "nested": {"allowed": True}}
    frozen = frozen_public_context(source)
    fingerprint = frozen["context_sha256"]
    source["metrics"].append("private")
    assert frozen["context"]["metrics"] == ["validation_pearson"]
    assert frozen["context_sha256"] == fingerprint
    json.dumps(frozen)
