"""Failure-preserving research evidence contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ._canonical import canonical_sha256, freeze_json, require_nonempty, thaw_json

EvidenceStatus = Literal["completed", "failed", "unavailable"]
CheckStatus = Literal["pass", "fail", "unavailable"]


@dataclass(frozen=True, slots=True)
class MetricObservation:
    """One named scientific or operational metric in one evidence domain."""

    name: str
    domain: str
    value: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", require_nonempty(self.name, name="metric.name"))
        object.__setattr__(self, "domain", require_nonempty(self.domain, name="metric.domain"))
        value = float(self.value)
        if not math.isfinite(value):
            raise ValueError("metric.value must be finite")
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "metadata", freeze_json(self.metadata, path="metric.metadata"))

    @property
    def key(self) -> tuple[str, str]:
        return self.domain, self.name

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "domain": self.domain,
            "value": self.value,
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class AdversarialCheck:
    """Explicit falsification, leakage, integrity, or null-control result."""

    check_id: str
    status: CheckStatus
    detail: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "check_id", require_nonempty(self.check_id, name="check_id"))
        object.__setattr__(self, "detail", str(self.detail).strip())
        object.__setattr__(self, "metadata", freeze_json(self.metadata, path="check.metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "status": self.status,
            "detail": self.detail,
            "metadata": thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ExperimentEvidence:
    """Evidence emitted for exactly one immutable experiment packet."""

    experiment_id: str
    packet_fingerprint: str
    evaluation_fingerprint: str
    status: EvidenceStatus
    metrics: tuple[MetricObservation, ...] = ()
    checks: tuple[AdversarialCheck, ...] = ()
    artifact_fingerprints: Mapping[str, str] = field(default_factory=dict)
    failure_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = "1.0"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "experiment_id", require_nonempty(self.experiment_id, name="experiment_id")
        )
        for name in ("packet_fingerprint", "evaluation_fingerprint"):
            value = require_nonempty(getattr(self, name), name=name).lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"{name} must be a full SHA-256")
            object.__setattr__(self, name, value)

        metric_keys = [metric.key for metric in self.metrics]
        if len(set(metric_keys)) != len(metric_keys):
            raise ValueError("metrics must not contain duplicate (domain, name) keys")
        check_ids = [check.check_id for check in self.checks]
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("checks must have unique check_id values")

        if self.status == "completed" and self.failure_reason is not None:
            raise ValueError("completed evidence cannot carry failure_reason")
        if self.status != "completed" and self.metrics:
            raise ValueError("failed/unavailable evidence cannot carry promoted metric values")
        if self.status != "completed" and not (self.failure_reason or "").strip():
            raise ValueError("failed/unavailable evidence must explain failure_reason")
        if self.failure_reason is not None:
            object.__setattr__(
                self,
                "failure_reason",
                require_nonempty(self.failure_reason, name="failure_reason"),
            )

        normalized_artifacts: dict[str, str] = {}
        for name, digest in self.artifact_fingerprints.items():
            key = require_nonempty(name, name="artifact name")
            value = require_nonempty(digest, name=f"artifact {key} fingerprint").lower()
            if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
                raise ValueError(f"artifact fingerprint for {key!r} must be a full SHA-256")
            normalized_artifacts[key] = value
        object.__setattr__(
            self,
            "artifact_fingerprints",
            freeze_json(normalized_artifacts, path="evidence.artifact_fingerprints"),
        )
        object.__setattr__(self, "metadata", freeze_json(self.metadata, path="evidence.metadata"))
        object.__setattr__(
            self, "schema_version", require_nonempty(self.schema_version, name="schema_version")
        )

    def metric_map(self) -> dict[tuple[str, str], float]:
        return {metric.key: metric.value for metric in self.metrics}

    def check_map(self) -> dict[str, AdversarialCheck]:
        return {check.check_id: check for check in self.checks}

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "experiment_id": self.experiment_id,
            "packet_fingerprint": self.packet_fingerprint,
            "evaluation_fingerprint": self.evaluation_fingerprint,
            "status": self.status,
            "metrics": [metric.to_dict() for metric in self.metrics],
            "checks": [check.to_dict() for check in self.checks],
            "artifact_fingerprints": thaw_json(self.artifact_fingerprints),
            "failure_reason": self.failure_reason,
            "metadata": thaw_json(self.metadata),
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict(include_fingerprint=False))
