from __future__ import annotations

import json

import pytest

import neuros.research.nim as nim


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
    assert nim._validate_endpoint("https://integrate.api.nvidia.com/v1") == (
        "https://integrate.api.nvidia.com/v1"
    )
    with pytest.raises(ValueError, match="HTTPS"):
        nim._validate_endpoint("http://integrate.api.nvidia.com/v1")
    with pytest.raises(ValueError, match="may only be sent"):
        nim._validate_endpoint("https://example.com/v1")


def test_json_extraction_tolerates_reasoning_prefix() -> None:
    payload = nim._extract_json_object(
        'analysis text\n{"candidates": [{"candidate_id": "a"}]}\n'
    )
    assert payload["candidates"][0]["candidate_id"] == "a"


def test_proposal_rejects_frozen_authority_mutation() -> None:
    payload = proposal_payload()
    payload["changed_variables"] = ["split.ood_levels"]
    with pytest.raises(ValueError, match="frozen authority"):
        nim.ResearchProposal.from_dict(
            payload,
            allowed_payload_classes=ALLOWED_PAYLOADS,
            allowed_development_metrics=ALLOWED_METRICS,
        )


def test_proposal_rejects_forbidden_payload_class() -> None:
    payload = proposal_payload()
    payload["required_payload_classes"] = ["raw_participant_data"]
    with pytest.raises(ValueError, match="outside dispatch policy"):
        nim.ResearchProposal.from_dict(
            payload,
            allowed_payload_classes=ALLOWED_PAYLOADS,
            allowed_development_metrics=ALLOWED_METRICS,
        )


def test_proposal_rejects_forbidden_feedback_language() -> None:
    payload = proposal_payload()
    payload["rationale"] = "Tune this against the private leaderboard."
    with pytest.raises(ValueError, match="forbidden feedback"):
        nim.ResearchProposal.from_dict(
            payload,
            allowed_payload_classes=ALLOWED_PAYLOADS,
            allowed_development_metrics=ALLOWED_METRICS,
        )


def test_parse_proposals_requires_unique_ids() -> None:
    payload = {"candidates": [proposal_payload("same") for _ in range(3)]}
    with pytest.raises(ValueError, match="unique"):
        nim.parse_proposals(
            payload,
            allowed_payload_classes=ALLOWED_PAYLOADS,
            allowed_development_metrics=ALLOWED_METRICS,
        )


def test_model_selection_uses_only_qualified_nvidia_reasoner() -> None:
    qualified = nim.DEFAULT_NVIDIA_MODEL_PREFERENCES[0]
    available = ("other/model", qualified, "nvidia/some-unqualified-nemotron")
    assert nim.NvidiaNimClient.select_models(available, count=3) == (qualified,)


def test_model_selection_fails_closed_without_qualified_model() -> None:
    with pytest.raises(ValueError, match="qualified NVIDIA Nemotron"):
        nim.NvidiaNimClient.select_models(("other/model",), count=1)


def test_public_context_is_detached_and_stable() -> None:
    source = {"metrics": ["validation_pearson"], "nested": {"allowed": True}}
    frozen = nim.frozen_public_context(source)
    fingerprint = frozen["context_sha256"]
    source["metrics"].append("private")
    assert frozen["context"]["metrics"] == ["validation_pearson"]
    assert frozen["context_sha256"] == fingerprint
    json.dumps(frozen)


def test_chat_json_disables_reasoning_for_machine_validated_output() -> None:
    captured: dict[str, object] = {}

    class StubNimClient(nim.NvidiaNimClient):
        def _request(self, path: str, *, payload=None):  # type: ignore[no-untyped-def]
            captured["path"] = path
            captured["payload"] = payload
            return {"choices": [{"message": {"content": '{"ok": true}'}}]}

    client = StubNimClient("secret-for-test-only")
    parsed, record = client.chat_json(
        role="test",
        model=nim.DEFAULT_NVIDIA_MODEL_PREFERENCES[0],
        system_prompt="Return JSON only.",
        user_prompt="Return ok.",
    )
    assert parsed == {"ok": True}
    assert captured["path"] == "chat/completions"
    request = captured["payload"]
    assert isinstance(request, dict)
    assert request["chat_template_kwargs"] == {"enable_thinking": False}
    assert record.model == nim.DEFAULT_NVIDIA_MODEL_PREFERENCES[0]
