from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuros.research._canonical import canonical_sha256
from neuros.research.reviewed_context import load_reviewed_aggregate_context


def _payload() -> dict[str, object]:
    payload: dict[str, object] = {
        "kind": "neuros_reviewed_aggregate_evidence",
        "schema_version": 1,
        "evidence_id": "fixture",
        "source_revision": "a" * 40,
        "scientific_fingerprint": "b" * 64,
        "claim_scope": "synthetic fixture",
        "model_context_policy": {
            "payload_class": "aggregate_metrics",
            "context_only": True,
            "may_inform_hypothesis_generation": True,
            "may_not_be_used_as_scientific_promotion": True,
            "excluded": ["raw arrays", "hidden outcomes"],
        },
        "aggregate": {"noise": [0.0, 0.5], "mean": [0.9, 0.6]},
    }
    payload["review_fingerprint"] = canonical_sha256(payload)
    return payload


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    return path


def test_reviewed_aggregate_context_accepts_checksummed_aggregate_only_payload(
    tmp_path: Path,
) -> None:
    payload = _payload()
    loaded = load_reviewed_aggregate_context(_write(tmp_path / "evidence.json", payload))
    assert loaded == payload
    assert loaded["model_context_policy"]["context_only"] is True


def test_reviewed_aggregate_context_fails_closed_on_fingerprint_tampering(
    tmp_path: Path,
) -> None:
    payload = _payload()
    payload["aggregate"]["mean"][0] = 0.1
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_reviewed_aggregate_context(_write(tmp_path / "tampered.json", payload))


@pytest.mark.parametrize("prohibited", ["runtime_seconds", "raw_data", "winner", "api_key"])
def test_reviewed_aggregate_context_rejects_prohibited_dispatch_keys(
    tmp_path: Path,
    prohibited: str,
) -> None:
    payload = _payload()
    payload[prohibited] = "forbidden"
    payload["review_fingerprint"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "review_fingerprint"}
    )
    with pytest.raises(ValueError, match="prohibited key"):
        load_reviewed_aggregate_context(_write(tmp_path / f"{prohibited}.json", payload))


def test_reviewed_aggregate_context_requires_context_only_policy(tmp_path: Path) -> None:
    payload = _payload()
    payload["model_context_policy"]["context_only"] = False
    payload["review_fingerprint"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "review_fingerprint"}
    )
    with pytest.raises(ValueError, match="context-only"):
        load_reviewed_aggregate_context(_write(tmp_path / "policy.json", payload))


def test_reviewed_aggregate_context_rejects_nonfinite_json(tmp_path: Path) -> None:
    path = tmp_path / "nan.json"
    path.write_text(
        '{"kind":"neuros_reviewed_aggregate_evidence","schema_version":1,"x":NaN}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        load_reviewed_aggregate_context(path)
