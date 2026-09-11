from __future__ import annotations

import pytest
from neuros.research.nim_observed import (
    NimTokenUsage,
    ObservedNimResponseError,
    ObservedQualifiedNvidiaNimClient,
)


class _Stub(ObservedQualifiedNvidiaNimClient):
    def __init__(self, response):  # type: ignore[no-untyped-def]
        super().__init__("secret-for-test-only")
        self.response = response
        self.requests = []

    def _request(self, path: str, *, payload=None, **kwargs):  # type: ignore[no-untyped-def]
        del kwargs
        assert path == "chat/completions"
        self.requests.append(payload)
        return self.response


def _response(usage=None, *, content='{"findings":[]}'):  # type: ignore[no-untyped-def]
    payload = {"choices": [{"message": {"content": content}}]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def _kwargs():
    return {
        "role": "swarm:reviewer",
        "model": "model-a",
        "system_prompt": "system",
        "user_prompt": "user",
        "max_tokens": 900,
        "temperature": 0.1,
    }


def test_observed_call_preserves_existing_request_and_call_record_contract():
    standard = _Stub(_response({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}))
    observed = _Stub(_response({"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}))

    parsed_standard, record_standard = standard.chat_json(**_kwargs())
    parsed_observed, record_observed, usage = observed.chat_json_observed(**_kwargs())

    assert parsed_standard == parsed_observed == {"findings": []}
    assert standard.requests == observed.requests
    assert record_standard.to_dict() == record_observed.to_dict()
    assert record_standard.response_sha256 == record_observed.response_sha256
    assert set(record_observed.to_dict()) == {
        "role",
        "model",
        "endpoint",
        "prompt_sha256",
        "request_sha256",
        "response_sha256",
    }
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 4
    assert usage.total_tokens == 14
    assert usage.provider_reported is True
    assert len(usage.usage_sha256 or "") == 64
    assert len(observed.call_journal) == 1


def test_missing_usage_is_preserved_as_unavailable_not_estimated():
    client = _Stub(_response())
    _, _, usage = client.chat_json_observed(**_kwargs())
    assert usage.to_dict() == {
        "source": "unavailable",
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "usage_sha256": None,
    }


def test_malformed_usage_counts_are_not_coerced_into_fake_exact_counts():
    client = _Stub(
        _response(
            {
                "prompt_tokens": True,
                "completion_tokens": -1,
                "total_tokens": "14",
            }
        )
    )
    _, _, usage = client.chat_json_observed(**_kwargs())
    assert usage.provider_reported is True
    assert usage.prompt_tokens is None
    assert usage.completion_tokens is None
    assert usage.total_tokens is None
    assert len(usage.usage_sha256 or "") == 64


def test_token_usage_rejects_forged_counts_without_provider_fingerprint():
    with pytest.raises(ValueError, match="require an exact usage-object fingerprint"):
        NimTokenUsage(prompt_tokens=10)
    with pytest.raises(ValueError, match="non-negative"):
        NimTokenUsage(prompt_tokens=-1, usage_sha256="a" * 64)


def test_observed_parser_rejects_duplicate_json_keys_and_preserves_transport_evidence():
    client = _Stub(
        _response(
            {"prompt_tokens": 11, "completion_tokens": 3, "total_tokens": 14},
            content='{"findings":[],"findings":[]}',
        )
    )
    with pytest.raises(ObservedNimResponseError, match="strict JSON parsing") as error:
        client.chat_json_observed(**_kwargs())
    assert len(error.value.record.request_sha256) == 64
    assert len(error.value.record.response_sha256) == 64
    assert error.value.token_usage.total_tokens == 14
    assert error.value.token_usage.provider_reported is True
    assert client.call_journal == []


@pytest.mark.parametrize(
    "content",
    (
        'Here is the answer: {"findings":[]}',
        '{"findings":[]} trailing prose',
        '[{"findings":[]}]',
    ),
)
def test_observed_parser_requires_entire_response_to_be_one_json_object(content: str):
    client = _Stub(_response(content=content))
    with pytest.raises(ObservedNimResponseError, match="strict JSON parsing"):
        client.chat_json_observed(**_kwargs())
    assert client.call_journal == []


def test_observed_parser_accepts_surrounding_whitespace_only():
    client = _Stub(_response(content='  \n {"findings":[]} \t '))
    parsed, _, _ = client.chat_json_observed(**_kwargs())
    assert parsed == {"findings": []}


def test_invalid_provider_response_structure_preserves_usage_for_failure_accounting():
    client = _Stub(
        {
            "choices": [],
            "usage": {"prompt_tokens": 8, "completion_tokens": 0, "total_tokens": 8},
        }
    )
    with pytest.raises(ObservedNimResponseError, match="missing choices") as error:
        client.chat_json_observed(**_kwargs())
    assert error.value.token_usage.total_tokens == 8
    assert len(error.value.record.response_sha256) == 64
