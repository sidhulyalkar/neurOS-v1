"""Frozen v1 schema catalog and backwards-compatible migration helpers."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .manifest import stable_hash

MANIFEST_SCHEMA_V2 = "2"
MANIFEST_SCHEMA_V3 = "3"
CURRENT_MANIFEST_SCHEMA = MANIFEST_SCHEMA_V3
ARTIFACT_ENVELOPE_SCHEMA = "neuros-mechint.artifact-envelope.v1"


@dataclass(frozen=True, slots=True)
class ArtifactSchemaSpec:
    """Machine-readable contract for one published artifact family."""

    family: str
    artifact_schema: str
    result_schema: str
    introduced_in: str

    def to_dict(self) -> dict[str, str]:
        return {
            "family": self.family,
            "artifact_schema": self.artifact_schema,
            "result_schema": self.result_schema,
            "introduced_in": self.introduced_in,
            "envelope_schema": ARTIFACT_ENVELOPE_SCHEMA,
        }


_ARTIFACT_SCHEMAS = {
    "evidence_pack": ArtifactSchemaSpec(
        family="evidence_pack",
        artifact_schema="neuros-mechint.evidence-pack-artifact.v1",
        result_schema="neuros-mechint.evidence-pack.v1",
        introduced_in="0.6.0",
    ),
    "factorial": ArtifactSchemaSpec(
        family="factorial",
        artifact_schema="neuros-mechint.factorial-mechanism-artifact.v1",
        result_schema="neuros-mechint.factorial-mechanism-study.v1",
        introduced_in="0.7.0",
    ),
    "correspondence": ArtifactSchemaSpec(
        family="correspondence",
        artifact_schema="neuros-mechint.feature-correspondence-artifact.v1",
        result_schema="neuros-mechint.feature-correspondence-study.v1",
        introduced_in="0.8.0",
    ),
    "replication": ArtifactSchemaSpec(
        family="replication",
        artifact_schema="neuros-mechint.hierarchical-replication-artifact.v1",
        result_schema="neuros-mechint.hierarchical-replication-study.v1",
        introduced_in="0.9.0",
    ),
    "dose_response": ArtifactSchemaSpec(
        family="dose_response",
        artifact_schema="neuros-mechint.dose-response-artifact.v1",
        result_schema="neuros-mechint.dose-response-study.v1",
        introduced_in="1.0.0",
    ),
}


def get_artifact_schema(family: str) -> ArtifactSchemaSpec:
    try:
        return _ARTIFACT_SCHEMAS[family]
    except KeyError as exc:
        raise KeyError(f"unknown artifact family: {family!r}") from exc


def schema_catalog() -> tuple[dict[str, str], ...]:
    """Return the frozen v1 artifact schema catalog."""

    return tuple(_ARTIFACT_SCHEMAS[name].to_dict() for name in sorted(_ARTIFACT_SCHEMAS))


def scientific_identity_from_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract deterministic scientific identity, excluding execution metadata."""

    evidence = payload.get("evidence_tier", {})
    if isinstance(evidence, Mapping):
        evidence_identity: Any = {
            "level": evidence.get("level"),
            "label": evidence.get("label"),
        }
    else:
        evidence_identity = evidence
    return {
        "experiment_name": payload.get("experiment_name"),
        "method": payload.get("method"),
        "method_version": payload.get("method_version"),
        "model_id": payload.get("model_id"),
        "model_revision": payload.get("model_revision"),
        "model_hash": payload.get("model_hash"),
        "dataset_id": payload.get("dataset_id"),
        "dataset_hash": payload.get("dataset_hash"),
        "parameters": deepcopy(payload.get("parameters", {})),
        "seed": payload.get("seed"),
        "evidence_tier": evidence_identity,
    }


def migrate_manifest_payload(
    payload: Mapping[str, Any], *, target_schema: str = CURRENT_MANIFEST_SCHEMA
) -> dict[str, Any]:
    """Migrate a serialized v0.x manifest into the frozen v1 manifest contract."""

    if target_schema != MANIFEST_SCHEMA_V3:
        raise ValueError(f"unsupported target manifest schema: {target_schema!r}")
    migrated = deepcopy(dict(payload))
    source = str(migrated.get("schema_version", MANIFEST_SCHEMA_V2))
    if source not in {MANIFEST_SCHEMA_V2, MANIFEST_SCHEMA_V3}:
        raise ValueError(f"unsupported manifest schema: {source!r}")
    if source == MANIFEST_SCHEMA_V2:
        identity = scientific_identity_from_manifest_payload(migrated)
        migrated["scientific_identity"] = identity
        migrated["scientific_fingerprint"] = stable_hash(identity)
        migrated["schema_version"] = MANIFEST_SCHEMA_V3
    return validate_manifest_payload(migrated)


def validate_manifest_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a v1 manifest payload and its deterministic identity."""

    validated = deepcopy(dict(payload))
    if str(validated.get("schema_version")) != MANIFEST_SCHEMA_V3:
        raise ValueError("manifest is not schema v3")
    identity = validated.get("scientific_identity")
    if not isinstance(identity, Mapping):
        raise TypeError("manifest scientific_identity must be an object")
    observed = stable_hash(dict(identity))
    if validated.get("scientific_fingerprint") != observed:
        raise ValueError("manifest scientific fingerprint mismatch")
    return validated


def make_artifact_envelope(family: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a result in the canonical v1 self-checking artifact envelope."""

    spec = get_artifact_schema(family)
    result_dict = deepcopy(dict(result))
    if result_dict.get("schema_version") != spec.result_schema:
        raise ValueError(
            f"{family} result schema must be {spec.result_schema!r}; "
            f"got {result_dict.get('schema_version')!r}"
        )
    return {
        "artifact_hash": stable_hash(result_dict),
        "artifact_schema": spec.artifact_schema,
        "contract": {
            "envelope_schema": ARTIFACT_ENVELOPE_SCHEMA,
            "family": family,
            "result_schema": spec.result_schema,
        },
        "result": result_dict,
    }


def migrate_artifact_envelope(payload: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    """Upgrade a pre-v1 envelope by attaching the frozen contract metadata.

    v0.6-v0.9 artifacts already carried family-specific schema IDs and full-content
    hashes. v1 keeps those IDs stable and adds a canonical contract block instead
    of rewriting the scientific result or changing its integrity hash.
    """

    spec = get_artifact_schema(family)
    migrated = deepcopy(dict(payload))
    if migrated.get("artifact_schema") != spec.artifact_schema:
        raise ValueError(f"unsupported {family} artifact schema")
    result = migrated.get("result")
    if not isinstance(result, Mapping):
        raise TypeError(f"{family} artifact result must be an object")
    result_dict = dict(result)
    if stable_hash(result_dict) != migrated.get("artifact_hash"):
        raise ValueError(f"{family} artifact hash mismatch")
    if result_dict.get("schema_version") != spec.result_schema:
        raise ValueError(f"unsupported {family} result schema")
    contract = migrated.get("contract")
    expected_contract = {
        "envelope_schema": ARTIFACT_ENVELOPE_SCHEMA,
        "family": family,
        "result_schema": spec.result_schema,
    }
    if contract is None:
        migrated["contract"] = expected_contract
    elif dict(contract) != expected_contract:
        raise ValueError(f"{family} artifact contract metadata mismatch")
    return migrated


def validate_artifact_envelope(payload: Mapping[str, Any], *, family: str) -> dict[str, Any]:
    """Validate either a historical v0.x envelope or a canonical v1 envelope."""

    return migrate_artifact_envelope(payload, family=family)
