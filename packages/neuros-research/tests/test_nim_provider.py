from __future__ import annotations

from io import BytesIO
from urllib.error import HTTPError

import pytest

from neuros.research.nim_provider import (
    DOCUMENTED_NVIDIA_CHAT_MODELS,
    QualifiedNvidiaNimClient,
)


def _ok_response() -> dict[str, object]:
    return {"choices": [{"message": {"content": "READY"}}]}


def test_probe_requires_real_nonempty_chat_response() -> None:
    class Stub(QualifiedNvidiaNimClient):
        def _request(self, path: str, *, payload=None):  # type: ignore[no-untyped-def]
            assert path == "chat/completions"
            assert payload["chat_template_kwargs"] == {"enable_thinking": False}
            return _ok_response()

    probe = Stub("secret-for-test-only").probe_model(DOCUMENTED_NVIDIA_CHAT_MODELS[0])
    assert probe.status == "qualified"
    assert probe.response_sha256 is not None
    assert probe.error_excerpt is None


def test_probe_preserves_http_status_and_redacts_credential(monkeypatch: pytest.MonkeyPatch) -> None:
    secret = "secret-for-test-only"

    def fail(*args, **kwargs):  # type: ignore[no-untyped-def]
        del args, kwargs
        raise HTTPError(
            url="https://integrate.api.nvidia.com/v1/chat/completions",
            code=404,
            msg="Not Found",
            hdrs=None,
            fp=BytesIO(f'{{"detail":"route missing {secret}"}}'.encode()),
        )

    monkeypatch.setattr("neuros.research.nim_provider.urlopen", fail)
    client = QualifiedNvidiaNimClient(secret, max_attempts=1)
    probe = client.probe_model(DOCUMENTED_NVIDIA_CHAT_MODELS[0])
    assert probe.status == "http_error"
    assert probe.status_code == 404
    assert secret not in probe.error_excerpt
    assert "[REDACTED]" in probe.error_excerpt


def test_discovery_never_promotes_documented_fallback_without_successful_probe() -> None:
    class Stub(QualifiedNvidiaNimClient):
        def _request(self, path: str, *, payload=None):  # type: ignore[no-untyped-def]
            if path == "models":
                raise RuntimeError("catalog unavailable")
            raise RuntimeError("chat route unavailable")

    client = Stub("secret-for-test-only", max_attempts=1)
    with pytest.raises(RuntimeError, match="no documented NVIDIA chat model passed"):
        client.discover_models()
    assert len(client.model_probes) == len(DOCUMENTED_NVIDIA_CHAT_MODELS)
    assert {probe.status for probe in client.model_probes} == {"transport_error"}
    assert client.provider_qualification()["qualified_models"] == []


def test_discovery_falls_through_to_next_documented_model_only_after_live_probe() -> None:
    second = DOCUMENTED_NVIDIA_CHAT_MODELS[1]

    class Stub(QualifiedNvidiaNimClient):
        def _request(self, path: str, *, payload=None):  # type: ignore[no-untyped-def]
            if path == "models":
                return {"data": [{"id": model} for model in DOCUMENTED_NVIDIA_CHAT_MODELS]}
            if payload["model"] == second:
                return _ok_response()
            raise RuntimeError("candidate unavailable")

    client = Stub("secret-for-test-only", max_attempts=1)
    available, mode = client.discover_models()
    assert available == (second,)
    assert mode == "api_models_endpoint+bounded_chat_probe"
    assert [probe.status for probe in client.model_probes] == [
        "transport_error",
        "qualified",
        "transport_error",
    ]


def test_provider_qualification_fingerprint_is_stable() -> None:
    class Stub(QualifiedNvidiaNimClient):
        def _request(self, path: str, *, payload=None):  # type: ignore[no-untyped-def]
            if path == "models":
                return {"data": [{"id": DOCUMENTED_NVIDIA_CHAT_MODELS[0]}]}
            return _ok_response()

    first = Stub("secret-for-test-only")
    first.discover_models()
    second = Stub("secret-for-test-only")
    second.discover_models()
    assert first.provider_qualification() == second.provider_qualification()
