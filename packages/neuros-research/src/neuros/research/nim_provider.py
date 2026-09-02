"""Fail-closed hosted NVIDIA transport qualification for research proposal calls."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, ClassVar
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from ._canonical import canonical_sha256, require_nonempty
from .nim import NimCallRecord, NvidiaNimClient

# Role preference is deliberate: the stronger models get the schema-heavy generator/critic
# positions when all routes qualify, while Lightning remains the fast workhorse fallback.
DOCUMENTED_NVIDIA_CHAT_MODELS = (
    "nvidia/nemotron-3-super-120b-a12b",
    "nvidia/nemotron-3-ultra-550b-a55b",
    "nvidia/nemotron-3.5-lightning-30b-a3b",
)
_RETRIABLE_HTTP = frozenset({408, 409, 429, 500, 502, 503, 504})
_MAX_PROVIDER_ERROR_BYTES = 4096
_MAX_PROVIDER_ERROR_CHARS = 768
_DISCOVERY_TIMEOUT_S = 20.0
_DISCOVERY_MAX_ATTEMPTS = 2


def _sanitize_provider_text(value: str, *, secret: str) -> str:
    """Return bounded one-line provider diagnostics with credential material removed."""

    text = str(value).replace(secret, "[REDACTED]")
    text = " ".join(text.split())
    return text[:_MAX_PROVIDER_ERROR_CHARS]


def _redact_secret(value: str, *, secret: str) -> str:
    return str(value).replace(secret, "[REDACTED]")


class NimProviderRequestError(RuntimeError):
    """Sanitized HTTP failure from the NVIDIA hosted API."""

    def __init__(self, *, path: str, status_code: int, response_excerpt: str) -> None:
        self.path = require_nonempty(path, name="path")
        self.status_code = int(status_code)
        self.response_excerpt = str(response_excerpt)
        super().__init__(f"NVIDIA NIM request failed with HTTP {self.status_code} at {self.path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "http_error",
            "path": self.path,
            "status_code": self.status_code,
            "response_excerpt": self.response_excerpt,
        }


@dataclass(frozen=True, slots=True)
class NimModelProbe:
    """One bounded model-route qualification result, excluding credentials and prompt text."""

    model: str
    status: str
    status_code: int | None = None
    response_sha256: str | None = None
    error_excerpt: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "model", require_nonempty(self.model, name="model"))
        if self.status not in {"qualified", "http_error", "transport_error", "invalid_response"}:
            raise ValueError(f"unsupported probe status {self.status!r}")
        if self.status == "qualified":
            if not self.response_sha256:
                raise ValueError("qualified probes require a response fingerprint")
            if self.status_code is not None or self.error_excerpt is not None:
                raise ValueError("qualified probes cannot carry failure evidence")
        elif not self.error_excerpt:
            raise ValueError("failed probes require bounded failure evidence")

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "status": self.status,
            "status_code": self.status_code,
            "response_sha256": self.response_sha256,
            "error_excerpt": self.error_excerpt,
        }


class QualifiedNvidiaNimClient(NvidiaNimClient):
    """NIM client that admits hosted models only after a bounded live chat probe."""

    latest_instance: ClassVar[QualifiedNvidiaNimClient | None] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.model_probes: tuple[NimModelProbe, ...] = ()
        self.catalog_models: tuple[str, ...] = ()
        self.catalog_error: dict[str, Any] | None = None
        self.discovery_mode = "not_run"
        self.call_journal: list[NimCallRecord] = []
        type(self).latest_instance = self

    def _request(
        self,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
        timeout_s: float | None = None,
        max_attempts: int | None = None,
    ) -> dict[str, Any]:
        """Use the base API contract while preserving sanitized, bounded transport evidence."""

        normalized_path = path.lstrip("/")
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self._api_key}",
        }
        if body is not None:
            headers["Content-Type"] = "application/json"
        request = Request(
            f"{self.endpoint}/{normalized_path}",
            data=body,
            headers=headers,
            method="GET" if payload is None else "POST",
        )
        request_timeout = self.timeout_s if timeout_s is None else float(timeout_s)
        attempts = self.max_attempts if max_attempts is None else int(max_attempts)
        if request_timeout <= 0.0:
            raise ValueError("request timeout must be positive")
        if attempts < 1 or attempts > self.max_attempts:
            raise ValueError("request attempts must be in [1, client.max_attempts]")

        last_error: Exception | None = None
        for attempt in range(attempts):
            try:
                with urlopen(request, timeout=request_timeout) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                if not isinstance(decoded, dict):
                    raise ValueError("NIM API returned a non-object JSON response")
                return decoded
            except HTTPError as exc:
                try:
                    raw = exc.read(_MAX_PROVIDER_ERROR_BYTES).decode("utf-8", errors="replace")
                except Exception:
                    raw = "provider response body unavailable"
                error = NimProviderRequestError(
                    path=normalized_path,
                    status_code=exc.code,
                    response_excerpt=_sanitize_provider_text(raw, secret=self._api_key),
                )
                last_error = error
                if exc.code not in _RETRIABLE_HTTP:
                    raise error from exc
            except (URLError, TimeoutError, OSError) as exc:
                reason = exc.reason if isinstance(exc, URLError) else exc
                last_error = RuntimeError(
                    "NVIDIA NIM transport error: "
                    + _sanitize_provider_text(str(reason), secret=self._api_key)
                )
            if attempt + 1 < attempts:
                time.sleep(min(2**attempt, 8))

        if isinstance(last_error, NimProviderRequestError):
            raise last_error
        raise RuntimeError("NVIDIA NIM API request exhausted retry budget") from last_error

    def _bounded_catalog_models(self) -> tuple[str, ...]:
        payload = self._request(
            "models",
            timeout_s=min(self.timeout_s, _DISCOVERY_TIMEOUT_S),
            max_attempts=min(self.max_attempts, _DISCOVERY_MAX_ATTEMPTS),
        )
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

    def probe_model(self, model: str) -> NimModelProbe:
        """Run a tiny bounded request using the same non-thinking chat mode as the tournament."""

        model = require_nonempty(model, name="model")
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Reply with READY."}],
            "temperature": 0.0,
            "max_tokens": 32,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        try:
            response = self._request(
                "chat/completions",
                payload=payload,
                timeout_s=min(self.timeout_s, _DISCOVERY_TIMEOUT_S),
                max_attempts=min(self.max_attempts, _DISCOVERY_MAX_ATTEMPTS),
            )
        except NimProviderRequestError as exc:
            return NimModelProbe(
                model=model,
                status="http_error",
                status_code=exc.status_code,
                error_excerpt=exc.response_excerpt or str(exc),
            )
        except (RuntimeError, TimeoutError, OSError) as exc:
            return NimModelProbe(
                model=model,
                status="transport_error",
                error_excerpt=_sanitize_provider_text(str(exc), secret=self._api_key),
            )
        except ValueError as exc:
            return NimModelProbe(
                model=model,
                status="invalid_response",
                error_excerpt=_sanitize_provider_text(str(exc), secret=self._api_key),
            )

        choices = response.get("choices")
        message = (
            choices[0].get("message")
            if isinstance(choices, list) and choices and isinstance(choices[0], dict)
            else None
        )
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            return NimModelProbe(
                model=model,
                status="invalid_response",
                error_excerpt="probe response did not contain non-empty message content",
            )
        return NimModelProbe(
            model=model,
            status="qualified",
            response_sha256=canonical_sha256(response),
        )

    def discover_models(self) -> tuple[tuple[str, ...], str]:
        """Treat catalog discovery as a hint; bounded chat execution is the authority."""

        try:
            self.catalog_models = self._bounded_catalog_models()
            self.catalog_error = None
            catalog_mode = "api_models_endpoint"
        except NimProviderRequestError as exc:
            self.catalog_models = ()
            self.catalog_error = exc.to_dict()
            catalog_mode = "catalog_unavailable"
        except (RuntimeError, ValueError, TimeoutError, OSError) as exc:
            self.catalog_models = ()
            self.catalog_error = {
                "kind": "catalog_error",
                "error_excerpt": _sanitize_provider_text(str(exc), secret=self._api_key),
            }
            catalog_mode = "catalog_unavailable"

        probes = tuple(self.probe_model(model) for model in DOCUMENTED_NVIDIA_CHAT_MODELS)
        self.model_probes = probes
        qualified = tuple(probe.model for probe in probes if probe.status == "qualified")
        self.discovery_mode = f"{catalog_mode}+bounded_chat_probe"
        if not qualified:
            diagnostics = [probe.to_dict() for probe in probes]
            raise RuntimeError(
                "no documented NVIDIA chat model passed bounded qualification: "
                + json.dumps(diagnostics, sort_keys=True, separators=(",", ":"))
            )
        return qualified, self.discovery_mode

    @staticmethod
    def select_models(
        available: tuple[str, ...],
        *,
        preferences: tuple[str, ...] = DOCUMENTED_NVIDIA_CHAT_MODELS,
        count: int = 3,
    ) -> tuple[str, ...]:
        """Select only live-qualified models, preserving documented role preference order."""

        if count < 1:
            raise ValueError("count must be positive")
        available_set = set(available)
        selected = tuple(model for model in preferences if model in available_set)
        if not selected:
            raise ValueError("no bounded-probe-qualified NVIDIA chat model is available")
        return selected[:count]

    def chat_json(self, **kwargs: Any) -> tuple[dict[str, Any], NimCallRecord]:
        """Journal every successfully parsed model response before semantic validation occurs."""

        parsed, record = super().chat_json(**kwargs)
        self.call_journal.append(record)
        return parsed, record

    def call_journal_payload(self) -> list[dict[str, Any]]:
        """Return credential-redacted call evidence, including failed semantic attempts."""

        rows: list[dict[str, Any]] = []
        for record in self.call_journal:
            row = record.to_dict(include_response=True)
            row["response_text"] = _redact_secret(row["response_text"], secret=self._api_key)
            rows.append(row)
        return rows

    def provider_qualification(self) -> dict[str, Any]:
        """Return the complete credential-free provider qualification record."""

        payload = {
            "schema_version": 1,
            "endpoint": self.endpoint,
            "discovery_mode": self.discovery_mode,
            "documented_candidates": list(DOCUMENTED_NVIDIA_CHAT_MODELS),
            "catalog_models_sha256": (
                canonical_sha256(list(self.catalog_models)) if self.catalog_models else None
            ),
            "catalog_error": self.catalog_error,
            "probes": [probe.to_dict() for probe in self.model_probes],
            "qualified_models": [
                probe.model for probe in self.model_probes if probe.status == "qualified"
            ],
            "discovery_budget": {
                "timeout_seconds_per_attempt": min(self.timeout_s, _DISCOVERY_TIMEOUT_S),
                "max_attempts_per_route": min(self.max_attempts, _DISCOVERY_MAX_ATTEMPTS),
            },
            "authority_boundary": (
                "A documented model is usable for this tournament only after the exact hosted "
                "chat route returns a non-empty response under the tournament's non-thinking mode. "
                "Timeout or failure of one candidate is preserved as probe evidence and cannot "
                "prevent independent qualification of another candidate."
            ),
        }
        payload["fingerprint"] = canonical_sha256(payload)
        return payload
