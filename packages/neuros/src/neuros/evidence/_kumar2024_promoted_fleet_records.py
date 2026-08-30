"""Attempt and accepted-result records for Kumar2024 FleetAuthority."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from ._kumar2024_promoted_fleet_common import (
    PromotedShardLease,
    _exact_nonnegative_int,
    _failure_code,
    _identity_sha256,
    _require_serialized_payload,
    _sha256,
    _strict_text,
)

AttemptOutcome = Literal["infrastructure_failure", "worker_artifact"]

@dataclass(frozen=True, slots=True)
class PromotedFleetAttemptRecord:
    """Append-only record for one lease outcome."""

    lease: PromotedShardLease
    outcome: AttemptOutcome
    worker_bundle_sha256: str | None = None
    shard_result_sha256: str | None = None
    infrastructure_failure_code: str | None = None
    infrastructure_failure_detail_sha256: str | None = None
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedFleetAttemptRecord schema_version must be 1")
        if not isinstance(self.lease, PromotedShardLease):
            raise TypeError("lease must be PromotedShardLease")
        if self.outcome not in {"infrastructure_failure", "worker_artifact"}:
            raise ValueError("unsupported promoted fleet attempt outcome")

        if self.outcome == "worker_artifact":
            object.__setattr__(
                self,
                "worker_bundle_sha256",
                _sha256("worker_bundle_sha256", self.worker_bundle_sha256),
            )
            object.__setattr__(
                self,
                "shard_result_sha256",
                _sha256("shard_result_sha256", self.shard_result_sha256),
            )
            if self.infrastructure_failure_code is not None:
                raise ValueError("worker artifact record cannot carry failure code")
            if self.infrastructure_failure_detail_sha256 is not None:
                raise ValueError("worker artifact record cannot carry failure detail SHA")
        else:
            code = _failure_code(self.infrastructure_failure_code)
            if self.worker_bundle_sha256 is not None or self.shard_result_sha256 is not None:
                raise ValueError(
                    "infrastructure failure cannot claim worker/result artifact identity"
                )
            detail_sha = self.infrastructure_failure_detail_sha256
            if detail_sha is not None:
                detail_sha = _sha256(
                    "infrastructure_failure_detail_sha256", detail_sha
                )
            object.__setattr__(self, "infrastructure_failure_code", code)
            object.__setattr__(
                self,
                "infrastructure_failure_detail_sha256",
                detail_sha,
            )

    @property
    def terminal(self) -> bool:
        return self.outcome == "worker_artifact"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "lease": self.lease.to_dict(),
            "outcome": self.outcome,
            "worker_bundle_sha256": self.worker_bundle_sha256,
            "shard_result_sha256": self.shard_result_sha256,
            "infrastructure_failure_code": self.infrastructure_failure_code,
            "infrastructure_failure_detail_sha256": (
                self.infrastructure_failure_detail_sha256
            ),
            "terminal": self.terminal,
            "scientific_retry_permitted": False,
            "infrastructure_retry_eligible": not self.terminal,
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_fleet_attempt.v1",
            self.to_dict(),
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PromotedFleetAttemptRecord":
        raw_lease = payload.get("lease")
        if not isinstance(raw_lease, Mapping):
            raise ValueError("serialized attempt record requires lease")
        value = cls(
            lease=PromotedShardLease.from_dict(raw_lease),
            outcome=payload["outcome"],
            worker_bundle_sha256=payload.get("worker_bundle_sha256"),
            shard_result_sha256=payload.get("shard_result_sha256"),
            infrastructure_failure_code=payload.get("infrastructure_failure_code"),
            infrastructure_failure_detail_sha256=payload.get(
                "infrastructure_failure_detail_sha256"
            ),
            schema_version=payload.get("schema_version", 1),
        )
        _require_serialized_payload(
            payload,
            value.to_dict(),
            object_name="attempt record",
            digest_field="attempt_record_sha256",
            digest=value.sha256,
        )
        return value


@dataclass(frozen=True, slots=True)
class PromotedFleetAcceptedResult:
    """Terminal worker artifact selected by authority, never by score."""

    shard_id: str
    shard_spec_sha256: str
    lease_sha256: str
    attempt_index: int
    worker_bundle_sha256: str
    shard_result_sha256: str
    attempt_record_sha256: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedFleetAcceptedResult schema_version must be 1")
        object.__setattr__(self, "shard_id", _strict_text("shard_id", self.shard_id))
        for name in (
            "shard_spec_sha256",
            "lease_sha256",
            "worker_bundle_sha256",
            "shard_result_sha256",
            "attempt_record_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        object.__setattr__(
            self,
            "attempt_index",
            _exact_nonnegative_int("attempt_index", self.attempt_index),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "shard_id": self.shard_id,
            "shard_spec_sha256": self.shard_spec_sha256,
            "lease_sha256": self.lease_sha256,
            "attempt_index": self.attempt_index,
            "worker_bundle_sha256": self.worker_bundle_sha256,
            "shard_result_sha256": self.shard_result_sha256,
            "attempt_record_sha256": self.attempt_record_sha256,
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_fleet_accepted_result.v1",
            self.to_dict(),
        )

__all__ = ["PromotedFleetAttemptRecord", "PromotedFleetAcceptedResult"]
