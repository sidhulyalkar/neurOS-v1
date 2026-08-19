"""Versioned self-checking artifacts for intervention dose-response studies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from neuros_mechint.core.manifest import stable_hash
from neuros_mechint.core.schema import make_artifact_envelope, validate_artifact_envelope

from .dose_response import DoseResponseResult

DOSE_RESPONSE_ARTIFACT_SCHEMA = "neuros-mechint.dose-response-artifact.v1"
DOSE_RESPONSE_STUDY_SCHEMA = "neuros-mechint.dose-response-study.v1"


def serialize_dose_response(result: DoseResponseResult) -> dict[str, Any]:
    """Serialize the scientific result separately from execution/file identity."""

    payload = result.to_dict()
    scientific_payload = {
        "spec": payload["spec"],
        "unit_summaries": payload["unit_summaries"],
        "aggregate_doses": payload["aggregate_doses"],
        "aggregate_metrics": payload["aggregate_metrics"],
        "endpoint_effect": payload["endpoint_effect"],
        "mean_monotonic_fraction": payload["mean_monotonic_fraction"],
        "normalized_auc": payload["normalized_auc"],
        "passed": payload["passed"],
        "reasons": payload["reasons"],
    }
    return {
        "schema_version": DOSE_RESPONSE_STUDY_SCHEMA,
        **scientific_payload,
        "study_fingerprint": stable_hash(scientific_payload),
    }


def write_dose_response_artifact(result: DoseResponseResult, path: str | Path) -> Path:
    destination = Path(path)
    envelope = make_artifact_envelope("dose_response", serialize_dose_response(result))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(envelope, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def read_dose_response_artifact(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    envelope = validate_artifact_envelope(payload, family="dose_response")
    result = dict(envelope["result"])
    scientific_payload = {
        key: value
        for key, value in result.items()
        if key not in {"schema_version", "study_fingerprint"}
    }
    if stable_hash(scientific_payload) != result.get("study_fingerprint"):
        raise ValueError("dose-response scientific fingerprint mismatch")
    return result
