"""NVIDIA NIM proposal adapter kept outside neurOS evidence authority."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ._canonical import canonical_sha256, freeze_json, require_nonempty, thaw_json

CandidateFamily = Literal[
    "representation",
    "temporal_alignment",
    "neural_geometry",
    "readout",
    "fusion",
    "generalization",
]
ComputeTier = Literal["low", "medium", "high"]

_ALLOWED_FAMILIES = {
    "representation",
    "temporal_alignment",
    "neural_geometry",
    "readout",
    "fusion",
    "generalization",
}
_ALLOWED_COMPUTE_TIERS = {"low", "medium", "high"}
_FORBIDDEN_VARIABLE_PREFIXES = (
    "dataset.",
    "evaluation.",
    "split.",
    "hidden_target.",
    "leaderboard.",
)
_FORBIDDEN_TEXT = (
    "raw participant data",
    "participant identifier",
    "hidden test target",
    "private leaderboard",
)
DEFAULT_NVIDIA_ENDPOINT = "https://integrate.api.nvidia.com/v1"
# The first live calibration deliberately prefers one currently documented hosted model.
# Additional Nemotron variants can be admitted after their structured-output behavior is
# independently qualified under the same adapter contract.
DEFAULT_NVIDIA_MODEL_PREFERENCES = (
    "nvidia/nemotron-3-super-120b-a12b",
)


def _sha256_text(value: str) -> str:
    return sha256(value.encode("utf-8")).hexdigest()


def _validate_endpoint(endpoint: str) -> str:
    normalized = require_nonempty(endpoint, name="endpoint").rstrip("/")
    parsed = urlparse(normalized)
    if parsed.scheme != "https":
        raise ValueError("NVIDIA NIM endpoint must use HTTPS")
    if parsed.hostname != "integrate.api.nvidia.com":
        raise ValueError("NVIDIA API key may only be sent to integrate.api.nvidia.com")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError("NVIDIA NIM endpoint must not contain credentials, query, or fragment")
    return normalized


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("NIM response did not contain a JSON object")


@dataclass(frozen=True, slots=True)
class ResearchProposal:
    """One untrusted, structurally validated research hypothesis from a proposer."""

    candidate_id: str
    title: str
    statement: str
    rationale: str
    family: CandidateFamily
    changed_variables: tuple[str, ...]
    required_payload_classes: tuple[str, ...]
    development_metrics: tuple[str, ...]
    falsification_test: str
    estimated_compute_tier: ComputeTier
    expected_failure_modes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("candidate_id", "title", "statement", "rationale", "falsification_test"):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), name=name))
        if self.family not in _ALLOWED_FAMILIES:
            raise ValueError(f"unsupported candidate family {self.family!r}")
        if self.estimated_compute_tier not in _ALLOWED_COMPUTE_TIERS:
            raise ValueError(f"unsupported compute tier {self.estimated_compute_tier!r}")

        for name in (
            "changed_variables",
            "required_payload_classes",
            "development_metrics",
            "expected_failure_modes",
        ):
            values = tuple(require_nonempty(value, name=name) for value in getattr(self, name))
            if name != "expected_failure_modes" and not values:
                raise ValueError(f"{name} must contain at least one value")
            if len(set(values)) != len(values):
                raise ValueError(f"{name} values must be unique")
            object.__setattr__(self, name, values)

        for variable in self.changed_variables:
            if variable.startswith(_FORBIDDEN_VARIABLE_PREFIXES):
                raise ValueError(f"proposal cannot change frozen authority variable {variable!r}")

        searchable = " ".join(
            (
                self.title,
                self.statement,
                self.rationale,
                self.falsification_test,
                *self.expected_failure_modes,
            )
        ).lower()
        for forbidden in _FORBIDDEN_TEXT:
            if forbidden in searchable:
                raise ValueError(f"proposal references forbidden feedback/data class {forbidden!r}")

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        allowed_payload_classes: tuple[str, ...],
        allowed_development_metrics: tuple[str, ...],
    ) -> "ResearchProposal":
        proposal = cls(
            candidate_id=str(payload["candidate_id"]),
            title=str(payload["title"]),
            statement=str(payload["statement"]),
            rationale=str(payload["rationale"]),
            family=str(payload["family"]),  # type: ignore[arg-type]
            changed_variables=tuple(str(value) for value in payload["changed_variables"]),
            required_payload_classes=tuple(
                str(value) for value in payload["required_payload_classes"]
            ),
            development_metrics=tuple(str(value) for value in payload["development_metrics"]),
            falsification_test=str(payload["falsification_test"]),
            estimated_compute_tier=str(payload["estimated_compute_tier"]),  # type: ignore[arg-type]
            expected_failure_modes=tuple(
                str(value) for value in payload.get("expected_failure_modes", ())
            ),
        )
        unexpected_payloads = set(proposal.required_payload_classes) - set(allowed_payload_classes)
        if unexpected_payloads:
            raise ValueError(
                "proposal requires payload classes outside dispatch policy: "
                + ", ".join(sorted(unexpected_payloads))
            )
        unexpected_metrics = set(proposal.development_metrics) - set(allowed_development_metrics)
        if unexpected_metrics:
            raise ValueError(
                "proposal requests development metrics outside frozen menu: "
                + ", ".join(sorted(unexpected_metrics))
            )
        return proposal

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "title": self.title,
            "statement": self.statement,
            "rationale": self.rationale,
            "family": self.family,
            "changed_variables": list(self.changed_variables),
            "required_payload_classes": list(self.required_payload_classes),
            "development_metrics": list(self.development_metrics),
            "falsification_test": self.falsification_test,
            "estimated_compute_tier": self.estimated_compute_tier,
            "expected_failure_modes": list(self.expected_failure_modes),
        }

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class NimCallRecord:
    """Exact identity of one NIM proposal/critique call, excluding credentials."""

    role: str
    model: str
    endpoint: str
    prompt_sha256: str
    request_sha256: str
    response_sha256: str
    response_text: str = field(repr=False)

    def __post_init__(self) -> None:
        for name in (
            "role",
            "model",
            "endpoint",
            "prompt_sha256",
            "request_sha256",
            "response_sha256",
        ):
            object.__setattr__(self, name, require_nonempty(getattr(self, name), name=name))
        object.__setattr__(self, "response_text", str(self.response_text))

    def to_dict(self, *, include_response: bool = False) -> dict[str, Any]:
        payload = {
            "role": self.role,
            "model": self.model,
            "endpoint": self.endpoint,
            "prompt_sha256": self.prompt_sha256,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
        }
        if include_response:
            payload["response_text"] = self.response_text
        return payload


class NvidiaNimClient:
    """Minimal stdlib client for NVIDIA's OpenAI-compatible hosted NIM API."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = DEFAULT_NVIDIA_ENDPOINT,
        timeout_s: float = 90.0,
        max_attempts: int = 4,
    ) -> None:
        self._api_key = require_nonempty(api_key, name="api_key")
        self.endpoint = _validate_endpoint(endpoint)
        self.timeout_s = float(timeout_s)
        self.max_attempts = int(max_attempts)
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if self.max_attempts < 1 or self.max_attempts > 8:
            raise ValueError("max_attempts must be in [1, 8]")

    def _request(self, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.endpoint}/{path.lstrip('/')}",
            data=body,
            headers=headers,
            method="GET" if payload is None else "POST",
        )

        last_error: Exception | None = None
        for attempt in range(self.max_attempts):
            try:
                with urlopen(request, timeout=self.timeout_s) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError("NIM API returned a non-object JSON response")
                return decoded
            except HTTPError as exc:
                last_error = exc
                if exc.code not in {408, 409, 429, 500, 502, 503, 504}:
                    raise RuntimeError(f"NIM API request failed with HTTP {exc.code}") from exc
            except URLError as exc:
                last_error = exc
            if attempt + 1 < self.max_attempts:
                time.sleep(min(2**attempt, 8))
        raise RuntimeError("NIM API request exhausted retry budget") from last_error

    def list_models(self) -> tuple[str, ...]:
        payload = self._request("models")
        rows = payload.get("data")
        if not isinstance(rows, list):
            raise ValueError("NIM model catalog missing data list")
        models = tuple(
            str(row["id"])
            for row in rows
            if isinstance(row, dict) and isinstance(row.get("id"), str) and row["id"].strip()
        )
        if not models:
            raise ValueError("NIM model catalog is empty")
        return tuple(dict.fromkeys(models))

    def discover_models(self) -> tuple[tuple[str, ...], str]:
        """Use API discovery when available, otherwise retain a documented fixed fallback."""

        try:
            return self.list_models(), "api_models_endpoint"
        except (RuntimeError, ValueError):
            return DEFAULT_NVIDIA_MODEL_PREFERENCES, "documented_model_fallback"

    @staticmethod
    def select_models(
        available: tuple[str, ...],
        *,
        preferences: tuple[str, ...] = DEFAULT_NVIDIA_MODEL_PREFERENCES,
        count: int = 3,
    ) -> tuple[str, ...]:
        if count < 1:
            raise ValueError("count must be positive")
        available_set = set(available)
        selected = tuple(model for model in preferences if model in available_set)
        if not selected:
            raise ValueError("the qualified NVIDIA Nemotron model is not available")
        return selected[:count]

    def chat_json(
        self,
        *,
        role: str,
        model: str,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> tuple[dict[str, Any], NimCallRecord]:
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
            # Nemotron 3 Super defaults to reasoning mode. For this machine-validated
            # proposal channel we explicitly request final-answer generation so the
            # token budget is spent on the JSON contract rather than an opaque trace.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        response = self._request("chat/completions", payload=request_payload)
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("NIM chat response missing choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise ValueError("NIM chat response missing message content")

        parsed = _extract_json_object(content)
        combined_prompt = f"{system_prompt}\n\n{user_prompt}"
        record = NimCallRecord(
            role=role,
            model=model,
            endpoint=self.endpoint,
            prompt_sha256=_sha256_text(combined_prompt),
            request_sha256=canonical_sha256(request_payload),
            response_sha256=canonical_sha256(response),
            response_text=content,
        )
        return parsed, record


def parse_proposals(
    payload: dict[str, Any],
    *,
    allowed_payload_classes: tuple[str, ...],
    allowed_development_metrics: tuple[str, ...],
    min_candidates: int = 3,
    max_candidates: int = 8,
) -> tuple[ResearchProposal, ...]:
    rows = payload.get("candidates")
    if not isinstance(rows, list):
        raise ValueError("proposal response must contain a candidates list")
    if len(rows) < min_candidates or len(rows) > max_candidates:
        raise ValueError(
            f"proposal response must contain {min_candidates}..{max_candidates} candidates"
        )
    proposals = tuple(
        ResearchProposal.from_dict(
            row,
            allowed_payload_classes=allowed_payload_classes,
            allowed_development_metrics=allowed_development_metrics,
        )
        for row in rows
        if isinstance(row, dict)
    )
    if len(proposals) != len(rows):
        raise ValueError("every candidate must be a JSON object")
    ids = [proposal.candidate_id for proposal in proposals]
    if len(set(ids)) != len(ids):
        raise ValueError("candidate_id values must be unique")
    return proposals


def frozen_public_context(payload: dict[str, Any]) -> dict[str, Any]:
    """Detach and fingerprint the public-only context approved for NIM dispatch."""

    frozen = freeze_json(payload, path="nim.public_context")
    return {
        "context": thaw_json(frozen),
        "context_sha256": canonical_sha256(frozen),
    }
