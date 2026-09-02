"""Fail-closed loading of reviewed aggregate evidence for external model context."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from ._canonical import canonical_sha256, freeze_json, require_sha256, thaw_json

_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_CONTEXT_BYTES = 65_536
_PROHIBITED_CONTEXT_KEYS = frozenset(
    {
        "api_key",
        "credential",
        "credentials",
        "hidden_target",
        "hidden_targets",
        "operational",
        "operational_summaries",
        "participant_identifier",
        "participant_identifiers",
        "per_case_rows",
        "private_leaderboard",
        "raw_arrays",
        "raw_data",
        "runtime_seconds",
        "secret",
        "secrets",
        "token",
        "winner",
    }
)


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"reviewed evidence contains non-finite JSON constant {value!r}")


def _reject_prohibited_keys(value: Any, *, path: str = "$", depth: int = 0) -> None:
    if depth > 12:
        raise ValueError("reviewed evidence nesting exceeds the allowed depth")
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).strip().lower()
            if normalized in _PROHIBITED_CONTEXT_KEYS:
                raise ValueError(f"reviewed evidence contains prohibited key {path}.{key}")
            _reject_prohibited_keys(item, path=f"{path}.{key}", depth=depth + 1)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_prohibited_keys(item, path=f"{path}[{index}]", depth=depth + 1)


def load_reviewed_aggregate_context(path: str | Path) -> dict[str, Any]:
    """Load one checksummed, aggregate-only context document for model prompting."""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > _MAX_CONTEXT_BYTES:
        raise ValueError(
            f"reviewed evidence exceeds {_MAX_CONTEXT_BYTES} bytes and will not be dispatched"
        )
    payload = json.loads(raw.decode("utf-8"), parse_constant=_reject_json_constant)
    if not isinstance(payload, dict):
        raise TypeError("reviewed evidence must be a JSON object")
    if payload.get("kind") != "neuros_reviewed_aggregate_evidence":
        raise ValueError("unexpected reviewed evidence kind")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported reviewed evidence schema version")

    source_revision = str(payload.get("source_revision", "")).strip().lower()
    if not _GIT_SHA_RE.fullmatch(source_revision):
        raise ValueError("reviewed evidence source_revision must be a full lowercase Git SHA")
    require_sha256(
        str(payload.get("scientific_fingerprint", "")),
        name="reviewed evidence scientific_fingerprint",
    )
    review_fingerprint = require_sha256(
        str(payload.get("review_fingerprint", "")),
        name="reviewed evidence review_fingerprint",
    )

    unsigned = dict(payload)
    unsigned.pop("review_fingerprint", None)
    if canonical_sha256(unsigned) != review_fingerprint:
        raise ValueError("reviewed evidence fingerprint mismatch")

    policy = payload.get("model_context_policy")
    if not isinstance(policy, dict):
        raise TypeError("reviewed evidence model_context_policy must be an object")
    if policy.get("payload_class") != "aggregate_metrics":
        raise ValueError("reviewed evidence must declare aggregate_metrics payload class")
    if policy.get("context_only") is not True:
        raise ValueError("reviewed evidence must be explicitly context-only")
    if policy.get("may_inform_hypothesis_generation") is not True:
        raise ValueError("reviewed evidence must explicitly permit hypothesis generation")
    if policy.get("may_not_be_used_as_scientific_promotion") is not True:
        raise ValueError("reviewed evidence must explicitly deny scientific promotion authority")

    _reject_prohibited_keys(payload)
    return thaw_json(freeze_json(payload, path="reviewed_evidence"))
