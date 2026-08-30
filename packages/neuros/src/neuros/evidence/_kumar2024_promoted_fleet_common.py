"""Fail-closed orchestration authority for promoted Kumar2024 execution.

The scientific execution plan determines *what* may run. This module governs
only *how* a trusted execution transport may retry and collect those immutable
worker assignments without acquiring scientific authority of its own.

The central invariant is strict::

    infrastructure failures may retry; a valid worker artifact may not.

A valid worker artifact is terminal even when its preserved scientific rows
report failure, OOM, non-convergence, or poor performance. Retrying such a
result would let infrastructure policy become an undeclared model-selection
mechanism.

Nothing here dispatches a worker, touches neural data, interprets a score, or
permits ORION comparison.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from .kumar2024_comparison import Kumar2024ComparisonPlan
from .kumar2024_promoted_execution import (
    PromotedExecutionPlan,
    PromotedShardResult,
    _exact_nonnegative_int,
    _git_sha,
    _identity_sha256,
    _sha256,
    assemble_promoted_execution,
    validate_promoted_shard_result,
)

AttemptOutcome = Literal["infrastructure_failure", "worker_artifact"]

_CLAIM_BOUNDARY = {
    "numerical_result_interpretable": False,
    "global_analysis_performed": False,
    "external_floor_claim_generated": False,
    "orion_comparison_permitted": False,
}

# These are intentionally code-owned rather than scheduler-owned. Adding or
# changing a retryable failure class is a scientific-transport change and, since
# this module is a promoted-binding ownership path, forces a fresh binding.
_ALLOWED_INFRASTRUCTURE_FAILURE_CODES = (
    "binding_artifact_download_failed",
    "checkout_failed",
    "data_access_failed_before_worker_artifact",
    "environment_realization_failed",
    "runner_lost_before_worker_artifact",
    "runner_unavailable",
    "worker_process_lost_before_artifact",
    "worker_start_failed",
)
_FAILURE_CODE_RE = re.compile(r"^[a-z0-9][a-z0-9_.-]{0,63}$")


def _claim_boundary() -> dict[str, bool]:
    return dict(_CLAIM_BOUNDARY)


def _strict_text(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _failure_code(value: Any) -> str:
    code = _strict_text("infrastructure_failure_code", value)
    if _FAILURE_CODE_RE.fullmatch(code) is None:
        raise ValueError(
            "infrastructure_failure_code must be a stable lowercase machine code "
            "using only a-z, 0-9, '.', '_', or '-'"
        )
    return code


def _detail_sha256(value: str) -> str:
    raw = _strict_text("infrastructure failure detail", value).encode("utf-8")
    return hashlib.sha256(
        b"neuros.kumar2024_promoted_infrastructure_failure_detail.v1\0" + raw
    ).hexdigest()


def _require_serialized_payload(
    payload: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    object_name: str,
    digest_field: str | None = None,
    digest: str | None = None,
) -> None:
    """Reject misleading derived fields, missing fields, and unknown fields.

    Evidence JSON is reviewer-facing. Reconstructing a safe Python object while
    silently accepting a raw file whose derived status/claim fields were altered
    would be cryptographically correct but operationally misleading. Serialized
    evidence therefore has to agree field-for-field with canonical rendering.
    """

    allowed = set(expected)
    if digest_field is not None:
        allowed.add(digest_field)
    unexpected = sorted(set(payload) - allowed)
    if unexpected:
        raise ValueError(f"serialized {object_name} contains unexpected fields: {unexpected}")
    missing = sorted(set(expected) - set(payload))
    if missing:
        raise ValueError(f"serialized {object_name} is missing fields: {missing}")
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"serialized {object_name} field {key!r} differs from canonical value")
    if digest_field is not None and digest_field in payload:
        if digest is None or payload[digest_field] != digest:
            raise ValueError(f"serialized {object_name} SHA-256 mismatch")


@dataclass(frozen=True, slots=True)
class PromotedFleetAuthority:
    """Immutable scheduler authority derived from one verified binding."""

    execution_plan_sha256: str
    binding_bundle_sha256: str
    source_revision: str
    shard_spec_sha256_by_id: tuple[tuple[str, str], ...]
    max_infrastructure_retries: int = 2
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedFleetAuthority schema_version must be 1")
        object.__setattr__(
            self,
            "execution_plan_sha256",
            _sha256("execution_plan_sha256", self.execution_plan_sha256),
        )
        object.__setattr__(
            self,
            "binding_bundle_sha256",
            _sha256("binding_bundle_sha256", self.binding_bundle_sha256),
        )
        object.__setattr__(self, "source_revision", _git_sha(self.source_revision))
        object.__setattr__(
            self,
            "max_infrastructure_retries",
            _exact_nonnegative_int(
                "max_infrastructure_retries", self.max_infrastructure_retries
            ),
        )

        pairs: list[tuple[str, str]] = []
        for raw_id, raw_sha in self.shard_spec_sha256_by_id:
            pairs.append(
                (
                    _strict_text("shard_id", raw_id),
                    _sha256("shard_spec_sha256", raw_sha),
                )
            )
        pairs.sort(key=lambda item: item[0])
        if not pairs:
            raise ValueError("fleet authority requires at least one shard")
        ids = [item[0] for item in pairs]
        shas = [item[1] for item in pairs]
        if len(set(ids)) != len(ids):
            raise ValueError("fleet authority shard ids must be unique")
        if len(set(shas)) != len(shas):
            raise ValueError("fleet authority shard SHA-256 values must be unique")
        object.__setattr__(self, "shard_spec_sha256_by_id", tuple(pairs))

    @property
    def shard_map(self) -> dict[str, str]:
        return dict(self.shard_spec_sha256_by_id)

    @property
    def shard_id_by_sha256(self) -> dict[str, str]:
        return {sha: shard_id for shard_id, sha in self.shard_spec_sha256_by_id}

    @property
    def expected_shards(self) -> int:
        return len(self.shard_spec_sha256_by_id)

    @property
    def max_attempts_per_shard(self) -> int:
        return self.max_infrastructure_retries + 1

    @property
    def allowed_infrastructure_failure_codes(self) -> tuple[str, ...]:
        return _ALLOWED_INFRASTRUCTURE_FAILURE_CODES

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "execution_plan_sha256": self.execution_plan_sha256,
            "binding_bundle_sha256": self.binding_bundle_sha256,
            "source_revision": self.source_revision,
            "max_infrastructure_retries": self.max_infrastructure_retries,
            "max_attempts_per_shard": self.max_attempts_per_shard,
            "expected_shards": self.expected_shards,
            "shard_spec_sha256_by_id": {
                shard_id: sha for shard_id, sha in self.shard_spec_sha256_by_id
            },
            "allowed_infrastructure_failure_codes": list(
                self.allowed_infrastructure_failure_codes
            ),
            "retry_semantics": (
                "retry only when the trusted transport establishes that no valid "
                "worker artifact was produced; any valid worker artifact is terminal "
                "regardless of scientific row status"
            ),
            "claim_boundary": _claim_boundary(),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_fleet_authority.v1",
            self.to_dict(),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotedFleetAuthority":
        raw_map = payload.get("shard_spec_sha256_by_id")
        if not isinstance(raw_map, Mapping):
            raise ValueError("serialized fleet authority requires shard map")
        value = cls(
            execution_plan_sha256=payload["execution_plan_sha256"],
            binding_bundle_sha256=payload["binding_bundle_sha256"],
            source_revision=payload["source_revision"],
            shard_spec_sha256_by_id=tuple(raw_map.items()),
            max_infrastructure_retries=payload.get("max_infrastructure_retries", 2),
            schema_version=payload.get("schema_version", 1),
        )
        _require_serialized_payload(
            payload,
            value.to_dict(),
            object_name="fleet authority",
            digest_field="fleet_authority_sha256",
            digest=value.sha256,
        )
        return value


def _build_promoted_fleet_authority_from_plan(
    execution_plan: PromotedExecutionPlan,
    *,
    binding_bundle_sha256: str,
    max_infrastructure_retries: int = 2,
) -> PromotedFleetAuthority:
    """Internal constructor for already-verified bindings and synthetic tests."""

    if not isinstance(execution_plan, PromotedExecutionPlan):
        raise TypeError("execution_plan must be PromotedExecutionPlan")
    return PromotedFleetAuthority(
        execution_plan_sha256=execution_plan.sha256,
        binding_bundle_sha256=binding_bundle_sha256,
        source_revision=execution_plan.binding.source_revision,
        shard_spec_sha256_by_id=tuple(
            (shard.shard_id, shard.sha256) for shard in execution_plan.template.shards
        ),
        max_infrastructure_retries=max_infrastructure_retries,
    )


def build_promoted_fleet_authority_from_binding(
    binding_root: str | Path,
    *,
    max_infrastructure_retries: int = 2,
) -> PromotedFleetAuthority:
    """Verify a sealed binding before deriving any fleet authority from it."""

    from .kumar2024_promoted_binding import (
        _binding_from_payload,
        _template_from_payload,
        verify_promoted_binding_bundle,
    )

    root = Path(binding_root).resolve()
    receipt = verify_promoted_binding_bundle(root)
    raw = json.loads((root / "execution_plan.json").read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("binding execution_plan.json must contain a JSON object")
    execution_plan = PromotedExecutionPlan(
        template=_template_from_payload(raw["template"]),
        binding=_binding_from_payload(raw["binding"]),
    )
    if raw.get("execution_plan_sha256") != execution_plan.sha256:
        raise ValueError("binding execution-plan payload SHA-256 mismatch")
    if receipt.get("execution_plan_sha256") != execution_plan.sha256:
        raise ValueError("verified binding receipt and execution plan differ")
    return _build_promoted_fleet_authority_from_plan(
        execution_plan,
        binding_bundle_sha256=receipt["bundle_sha256"],
        max_infrastructure_retries=max_infrastructure_retries,
    )


@dataclass(frozen=True, slots=True)
class PromotedShardLease:
    """One content-addressed permission to attempt one immutable shard."""

    fleet_authority_sha256: str
    execution_plan_sha256: str
    shard_id: str
    shard_spec_sha256: str
    attempt_index: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedShardLease schema_version must be 1")
        object.__setattr__(
            self,
            "fleet_authority_sha256",
            _sha256("fleet_authority_sha256", self.fleet_authority_sha256),
        )
        object.__setattr__(
            self,
            "execution_plan_sha256",
            _sha256("execution_plan_sha256", self.execution_plan_sha256),
        )
        object.__setattr__(self, "shard_id", _strict_text("shard_id", self.shard_id))
        object.__setattr__(
            self,
            "shard_spec_sha256",
            _sha256("shard_spec_sha256", self.shard_spec_sha256),
        )
        object.__setattr__(
            self,
            "attempt_index",
            _exact_nonnegative_int("attempt_index", self.attempt_index),
        )

    def identity_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fleet_authority_sha256": self.fleet_authority_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "shard_id": self.shard_id,
            "shard_spec_sha256": self.shard_spec_sha256,
            "attempt_index": self.attempt_index,
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_shard_lease.v1",
            self.identity_dict(),
        )

    @property
    def artifact_key(self) -> str:
        return (
            "nsq-kumar2024/promoted/"
            f"fleet-{self.fleet_authority_sha256}/"
            f"shard-{self.shard_spec_sha256}/"
            f"attempt-{self.attempt_index:04d}/"
            f"lease-{self.sha256}"
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.identity_dict(),
            "lease_sha256": self.sha256,
            "artifact_key": self.artifact_key,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotedShardLease":
        value = cls(
            fleet_authority_sha256=payload["fleet_authority_sha256"],
            execution_plan_sha256=payload["execution_plan_sha256"],
            shard_id=payload["shard_id"],
            shard_spec_sha256=payload["shard_spec_sha256"],
            attempt_index=payload["attempt_index"],
            schema_version=payload.get("schema_version", 1),
        )
        _require_serialized_payload(
            payload,
            value.to_dict(),
            object_name="shard lease",
        )
        return value


def _validate_lease_against_authority(
    lease: PromotedShardLease,
    authority: PromotedFleetAuthority,
) -> None:
    if not isinstance(lease, PromotedShardLease):
        raise TypeError("lease must be PromotedShardLease")
    if lease.fleet_authority_sha256 != authority.sha256:
        raise ValueError("lease belongs to a different fleet authority")
    if lease.execution_plan_sha256 != authority.execution_plan_sha256:
        raise ValueError("lease names a different execution plan")
    expected = authority.shard_map.get(lease.shard_id)
    if expected is None:
        raise ValueError("lease names an unknown fleet shard id")
    if lease.shard_spec_sha256 != expected:
        raise ValueError("lease shard id and shard SHA-256 disagree")
    if lease.attempt_index >= authority.max_attempts_per_shard:
        raise ValueError("lease exceeds frozen infrastructure retry budget")

__all__ = [
    "PromotedFleetAuthority",
    "PromotedShardLease",
    "build_promoted_fleet_authority_from_binding",
]
