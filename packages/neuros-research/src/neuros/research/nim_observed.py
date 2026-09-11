"""Additive observation surface for qualified NVIDIA NIM chat calls.

This module preserves the existing ``NimCallRecord`` contract. Telemetry is returned
separately so old request/response identities and serialized evidence do not change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ._canonical import canonical_sha256, require_nonempty, require_sha256
from .nim import NimCallRecord, _sha256_text
from .nim_provider import QualifiedNvidiaNimClient


def _optional_nonnegative_int(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


class _DuplicateObjectKeyError(ValueError):
    pass


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateObjectKeyError(f"duplicate JSON object key {key!r}")
        result[key] = value
    return result


def _extract_json_object_strict(text: str) -> dict[str, Any]:
    """Require the complete non-whitespace response to be exactly one JSON object."""
    stripped = text.strip()
    if not stripped:
        raise ValueError("NIM response did not contain a JSON object")
    decoder = json.JSONDecoder(object_pairs_hook=_reject_duplicate_pairs)
    try:
        value, end = decoder.raw_decode(stripped)
    except _DuplicateObjectKeyError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError("NIM response was not exactly one JSON object") from exc
    if not isinstance(value, dict):
        raise ValueError("NIM response must be exactly one JSON object")
    if stripped[end:].strip():
        raise ValueError("NIM response contained trailing content outside the JSON object")
    return value


@dataclass(frozen=True, slots=True)
class NimTokenUsage:
    """Provider-reported token counts, without estimating missing values."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    usage_sha256: str | None = None

    def __post_init__(self) -> None:
        counts = (self.prompt_tokens, self.completion_tokens, self.total_tokens)
        for value in counts:
            if value is not None and _optional_nonnegative_int(value) != value:
                raise ValueError("provider token counts must be non-negative integers or None")
        if self.usage_sha256 is None:
            if any(value is not None for value in counts):
                raise ValueError("provider token counts require an exact usage-object fingerprint")
        else:
            object.__setattr__(
                self,
                "usage_sha256",
                require_sha256(self.usage_sha256, name="usage_sha256"),
            )

    @classmethod
    def from_response(cls, response: dict[str, Any]) -> NimTokenUsage:
        usage = response.get("usage")
        if not isinstance(usage, dict):
            return cls()
        return cls(
            prompt_tokens=_optional_nonnegative_int(usage.get("prompt_tokens")),
            completion_tokens=_optional_nonnegative_int(usage.get("completion_tokens")),
            total_tokens=_optional_nonnegative_int(usage.get("total_tokens")),
            usage_sha256=canonical_sha256(usage),
        )

    @property
    def provider_reported(self) -> bool:
        return self.usage_sha256 is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": "provider_response" if self.provider_reported else "unavailable",
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "usage_sha256": self.usage_sha256,
        }


class ObservedNimResponseError(ValueError):
    """Hosted response reached us but could not become a valid strict review payload."""

    def __init__(
        self,
        message: str,
        *,
        record: NimCallRecord,
        token_usage: NimTokenUsage,
    ) -> None:
        self.record = record
        self.token_usage = token_usage
        super().__init__(message)


class ObservedQualifiedNvidiaNimClient(QualifiedNvidiaNimClient):
    """Qualified client that returns telemetry without altering ``NimCallRecord``."""

    def chat_json_observed(
        self,
        *,
        role: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], NimCallRecord, NimTokenUsage]:
        role = require_nonempty(role, name="role")
        model = require_nonempty(model, name="model")
        system_prompt = require_nonempty(system_prompt, name="system_prompt")
        user_prompt = require_nonempty(user_prompt, name="user_prompt")
        if max_tokens < 128 or max_tokens > 16384:
            raise ValueError("max_tokens must be in [128, 16384]")
        if temperature < 0.0 or temperature > 1.0:
            raise ValueError("temperature must be in [0, 1]")

        request_payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self._request("chat/completions", payload=request_payload)
        choices = response.get("choices")
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        content = message.get("content") if isinstance(message, dict) else None
        response_text = content if isinstance(content, str) else ""
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        record = NimCallRecord(
            role=role,
            model=model,
            endpoint=self.endpoint,
            prompt_sha256=_sha256_text(combined_prompt),
            request_sha256=canonical_sha256(request_payload),
            response_sha256=canonical_sha256(response),
            response_text=response_text,
        )
        token_usage = NimTokenUsage.from_response(response)

        if not isinstance(choices, list) or not choices:
            raise ObservedNimResponseError(
                "NIM chat response missing choices",
                record=record,
                token_usage=token_usage,
            )
        if not isinstance(content, str) or not content.strip():
            raise ObservedNimResponseError(
                "NIM chat response missing message content",
                record=record,
                token_usage=token_usage,
            )
        try:
            parsed = _extract_json_object_strict(content)
        except ValueError as exc:
            raise ObservedNimResponseError(
                f"NIM chat response failed strict JSON parsing: {type(exc).__name__}",
                record=record,
                token_usage=token_usage,
            ) from exc

        self.call_journal.append(record)
        return parsed, record, token_usage
