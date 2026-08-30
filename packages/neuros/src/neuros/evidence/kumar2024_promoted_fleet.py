"""Fail-closed orchestration authority for promoted Kumar2024 execution.

The scientific execution plan already determines *what* may run.  This module
governs only *how* an external scheduler may retry and collect those immutable
worker assignments without acquiring scientific authority of its own.

The central invariant is intentionally strict:

    infrastructure failures may retry; a valid worker artifact may not.

A valid worker artifact is terminal even when its preserved scientific rows
report failure, OOM, non-convergence, or poor performance.  Retrying such a
result would let infrastructure policy become an undeclared model-selection
mechanism.

Nothing in this module dispatches a worker, touches neural data, interprets a
score, or permits ORION comparison.
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
    _nonempty,
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


@dataclass(frozen=True, slots=True)
class PromotedFleetAuthority:
    """Immutable scheduler authority derived from one bound execution plan.

    The fleet authority contains no queue/provider/hostname/timestamp state.
    Those details are operational telemetry, not scientific identity.
    """

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
        retries = _exact_nonnegative_int(
            "max_infrastructure_retries", self.max_infrastructure_retries
        )
        object.__setattr__(self, "max_infrastructure_retries", retries)

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
            "retry_semantics": (
                "retry only when no valid worker artifact was produced; any valid "
                "worker artifact is terminal regardless of scientific row status"
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
        declared = payload.get("fleet_authority_sha256")
        if declared is not None and declared != value.sha256:
            raise ValueError("serialized fleet authority SHA-256 mismatch")
        return value


def build_promoted_fleet_authority(
    execution_plan: PromotedExecutionPlan,
    *,
    binding_bundle_sha256: str,
    max_infrastructure_retries: int = 2,
) -> PromotedFleetAuthority:
    """Freeze scheduler-visible identities without adding scientific choices."""

    if not isinstance(execution_plan, PromotedExecutionPlan):
        raise TypeError("execution_plan must be PromotedExecutionPlan")
    return PromotedFleetAuthority(
        execution_plan_sha256=execution_plan.sha256,
        binding_bundle_sha256=binding_bundle_sha256,
        source_revision=execution_plan.binding.source_revision,
        shard_spec_sha256_by_id=tuple(
            (shard.shard_id, shard.sha256)
            for shard in execution_plan.template.shards
        ),
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
        """Stable logical result location independent of scheduler/provider names."""

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
        if payload.get("lease_sha256") not in (None, value.sha256):
            raise ValueError("serialized shard lease SHA-256 mismatch")
        if payload.get("artifact_key") not in (None, value.artifact_key):
            raise ValueError("serialized shard lease artifact key mismatch")
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


@dataclass(frozen=True, slots=True)
class PromotedFleetAttemptRecord:
    """Append-only record for one lease outcome.

    ``worker_artifact`` is terminal.  It says only that a valid worker artifact
    exists, not that the model succeeded scientifically.
    """

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
            worker_sha = _sha256("worker_bundle_sha256", self.worker_bundle_sha256)
            result_sha = _sha256("shard_result_sha256", self.shard_result_sha256)
            if self.infrastructure_failure_code is not None:
                raise ValueError("worker artifact record cannot carry failure code")
            if self.infrastructure_failure_detail_sha256 is not None:
                raise ValueError("worker artifact record cannot carry failure detail SHA")
            object.__setattr__(self, "worker_bundle_sha256", worker_sha)
            object.__setattr__(self, "shard_result_sha256", result_sha)
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
                self, "infrastructure_failure_detail_sha256", detail_sha
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
        declared = payload.get("attempt_record_sha256")
        if declared is not None and declared != value.sha256:
            raise ValueError("serialized attempt-record SHA-256 mismatch")
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


@dataclass(frozen=True, slots=True)
class PromotedFleetLedger:
    """Canonical append-only attempt history for one fleet authority."""

    authority: PromotedFleetAuthority
    attempts: tuple[PromotedFleetAttemptRecord, ...] = ()
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedFleetLedger schema_version must be 1")
        if not isinstance(self.authority, PromotedFleetAuthority):
            raise TypeError("authority must be PromotedFleetAuthority")
        attempts = tuple(self.attempts)
        if any(not isinstance(item, PromotedFleetAttemptRecord) for item in attempts):
            raise TypeError("attempts must contain PromotedFleetAttemptRecord objects")

        seen_lease_sha: set[str] = set()
        grouped: dict[str, list[PromotedFleetAttemptRecord]] = {}
        for record in attempts:
            _validate_lease_against_authority(record.lease, self.authority)
            if record.lease.sha256 in seen_lease_sha:
                raise ValueError("fleet ledger cannot duplicate a lease outcome")
            seen_lease_sha.add(record.lease.sha256)
            grouped.setdefault(record.lease.shard_spec_sha256, []).append(record)

        canonical: list[PromotedFleetAttemptRecord] = []
        for shard_sha, records in grouped.items():
            ordered = sorted(records, key=lambda item: item.lease.attempt_index)
            indices = [item.lease.attempt_index for item in ordered]
            if indices != list(range(len(indices))):
                raise ValueError(
                    f"fleet attempts for shard {shard_sha} must be contiguous from zero"
                )
            terminal = [index for index, item in enumerate(ordered) if item.terminal]
            if len(terminal) > 1:
                raise ValueError("fleet ledger permits at most one worker artifact per shard")
            if terminal and terminal[0] != len(ordered) - 1:
                raise ValueError(
                    "no fleet attempt may follow a valid worker artifact"
                )
            if len(ordered) > self.authority.max_attempts_per_shard:
                raise ValueError("fleet ledger exceeds frozen retry budget")
            canonical.extend(ordered)

        terminal_records = [item for item in canonical if item.terminal]
        worker_hashes = [item.worker_bundle_sha256 for item in terminal_records]
        result_hashes = [item.shard_result_sha256 for item in terminal_records]
        if len(set(worker_hashes)) != len(worker_hashes):
            raise ValueError("fleet ledger cannot accept one worker bundle for multiple shards")
        if len(set(result_hashes)) != len(result_hashes):
            raise ValueError("fleet ledger cannot accept one shard result for multiple shards")

        canonical.sort(
            key=lambda item: (
                item.lease.shard_id,
                item.lease.attempt_index,
            )
        )
        object.__setattr__(self, "attempts", tuple(canonical))

    @property
    def attempts_by_shard_sha256(
        self,
    ) -> dict[str, tuple[PromotedFleetAttemptRecord, ...]]:
        grouped: dict[str, list[PromotedFleetAttemptRecord]] = {}
        for record in self.attempts:
            grouped.setdefault(record.lease.shard_spec_sha256, []).append(record)
        return {key: tuple(values) for key, values in grouped.items()}

    @property
    def accepted_results(self) -> tuple[PromotedFleetAcceptedResult, ...]:
        accepted: list[PromotedFleetAcceptedResult] = []
        for record in self.attempts:
            if not record.terminal:
                continue
            assert record.worker_bundle_sha256 is not None
            assert record.shard_result_sha256 is not None
            accepted.append(
                PromotedFleetAcceptedResult(
                    shard_id=record.lease.shard_id,
                    shard_spec_sha256=record.lease.shard_spec_sha256,
                    lease_sha256=record.lease.sha256,
                    attempt_index=record.lease.attempt_index,
                    worker_bundle_sha256=record.worker_bundle_sha256,
                    shard_result_sha256=record.shard_result_sha256,
                    attempt_record_sha256=record.sha256,
                )
            )
        accepted.sort(key=lambda item: item.shard_id)
        return tuple(accepted)

    @property
    def accepted_result_map(self) -> dict[str, PromotedFleetAcceptedResult]:
        return {item.shard_spec_sha256: item for item in self.accepted_results}

    @property
    def complete(self) -> bool:
        return len(self.accepted_results) == self.authority.expected_shards

    @property
    def exhausted_without_artifact(self) -> tuple[str, ...]:
        terminal_shas = set(self.accepted_result_map)
        grouped = self.attempts_by_shard_sha256
        exhausted = [
            shard_sha
            for shard_sha, records in grouped.items()
            if shard_sha not in terminal_shas
            and len(records) >= self.authority.max_attempts_per_shard
        ]
        return tuple(sorted(exhausted))

    def next_lease(self, shard_spec_sha256: str) -> PromotedShardLease:
        shard_sha = _sha256("shard_spec_sha256", shard_spec_sha256)
        shard_id = self.authority.shard_id_by_sha256.get(shard_sha)
        if shard_id is None:
            raise ValueError("requested shard is not part of fleet authority")
        records = self.attempts_by_shard_sha256.get(shard_sha, ())
        if any(item.terminal for item in records):
            raise ValueError("valid worker artifact already closes this shard")
        attempt_index = len(records)
        if attempt_index >= self.authority.max_attempts_per_shard:
            raise ValueError("frozen infrastructure retry budget is exhausted")
        return PromotedShardLease(
            fleet_authority_sha256=self.authority.sha256,
            execution_plan_sha256=self.authority.execution_plan_sha256,
            shard_id=shard_id,
            shard_spec_sha256=shard_sha,
            attempt_index=attempt_index,
        )

    def dispatchable_leases(self) -> tuple[PromotedShardLease, ...]:
        """Return the next immutable lease for every unresolved, non-exhausted shard."""

        leases: list[PromotedShardLease] = []
        grouped = self.attempts_by_shard_sha256
        for shard_id, shard_sha in self.authority.shard_spec_sha256_by_id:
            records = grouped.get(shard_sha, ())
            if any(item.terminal for item in records):
                continue
            if len(records) >= self.authority.max_attempts_per_shard:
                continue
            leases.append(self.next_lease(shard_sha))
        return tuple(leases)

    def append(self, record: PromotedFleetAttemptRecord) -> "PromotedFleetLedger":
        """Append one record while revalidating the entire canonical ledger."""

        if not isinstance(record, PromotedFleetAttemptRecord):
            raise TypeError("record must be PromotedFleetAttemptRecord")
        expected = self.next_lease(record.lease.shard_spec_sha256)
        if record.lease != expected:
            raise ValueError("attempt record does not consume the exact next fleet lease")
        return PromotedFleetLedger(
            authority=self.authority,
            attempts=(*self.attempts, record),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fleet_authority_sha256": self.authority.sha256,
            "attempt_records": [
                {**item.to_dict(), "attempt_record_sha256": item.sha256}
                for item in self.attempts
            ],
            "accepted_results": [
                {**item.to_dict(), "accepted_result_sha256": item.sha256}
                for item in self.accepted_results
            ],
            "attempted_records": len(self.attempts),
            "accepted_shards": len(self.accepted_results),
            "expected_shards": self.authority.expected_shards,
            "complete": self.complete,
            "exhausted_without_artifact": list(self.exhausted_without_artifact),
            "claim_boundary": _claim_boundary(),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_fleet_ledger.v1",
            self.to_dict(),
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping[str, Any],
        *,
        authority: PromotedFleetAuthority,
    ) -> "PromotedFleetLedger":
        if payload.get("fleet_authority_sha256") != authority.sha256:
            raise ValueError("serialized ledger names a different fleet authority")
        raw_attempts = payload.get("attempt_records")
        if not isinstance(raw_attempts, list):
            raise ValueError("serialized fleet ledger requires attempt_records")
        value = cls(
            authority=authority,
            attempts=tuple(
                PromotedFleetAttemptRecord.from_dict(item)
                for item in raw_attempts
            ),
            schema_version=payload.get("schema_version", 1),
        )
        declared = payload.get("fleet_ledger_sha256")
        if declared is not None and declared != value.sha256:
            raise ValueError("serialized fleet ledger SHA-256 mismatch")
        return value


def record_infrastructure_failure(
    ledger: PromotedFleetLedger,
    lease: PromotedShardLease,
    *,
    failure_code: str,
    failure_detail: str | None = None,
) -> PromotedFleetLedger:
    """Record a retryable transport/runtime failure that produced no worker artifact."""

    if not isinstance(ledger, PromotedFleetLedger):
        raise TypeError("ledger must be PromotedFleetLedger")
    detail_sha = None if failure_detail is None else _detail_sha256(failure_detail)
    return ledger.append(
        PromotedFleetAttemptRecord(
            lease=lease,
            outcome="infrastructure_failure",
            infrastructure_failure_code=failure_code,
            infrastructure_failure_detail_sha256=detail_sha,
        )
    )


def record_worker_artifact(
    ledger: PromotedFleetLedger,
    lease: PromotedShardLease,
    *,
    worker_bundle_sha256: str,
    shard_result: PromotedShardResult,
    execution_plan: PromotedExecutionPlan,
    comparison_plan: Kumar2024ComparisonPlan,
) -> PromotedFleetLedger:
    """Accept one already-verified worker artifact and permanently close its shard.

    Scientific row status is deliberately ignored for retry policy.  Existing
    promoted-result validation still verifies the complete frontier and every
    bound authority before the artifact is accepted. Persisted worker bundles
    should normally enter through :func:`record_verified_worker_bundle`, which
    cryptographically verifies the complete bundle before calling this lower-
    level ingestion primitive.
    """

    if not isinstance(ledger, PromotedFleetLedger):
        raise TypeError("ledger must be PromotedFleetLedger")
    if not isinstance(execution_plan, PromotedExecutionPlan):
        raise TypeError("execution_plan must be PromotedExecutionPlan")
    if ledger.authority.execution_plan_sha256 != execution_plan.sha256:
        raise ValueError("fleet authority and execution plan differ")
    expected = ledger.next_lease(lease.shard_spec_sha256)
    if lease != expected:
        raise ValueError("worker artifact does not consume the exact next fleet lease")
    if shard_result.shard_spec_sha256 != lease.shard_spec_sha256:
        raise ValueError("worker artifact shard differs from lease")
    validate_promoted_shard_result(
        shard_result,
        execution_plan=execution_plan,
        comparison_plan=comparison_plan,
    )
    return ledger.append(
        PromotedFleetAttemptRecord(
            lease=lease,
            outcome="worker_artifact",
            worker_bundle_sha256=worker_bundle_sha256,
            shard_result_sha256=shard_result.sha256,
        )
    )


def record_verified_worker_bundle(
    ledger: PromotedFleetLedger,
    lease: PromotedShardLease,
    *,
    worker_root: str | Path,
    binding_root: str | Path,
    execution_plan: PromotedExecutionPlan,
    comparison_plan: Kumar2024ComparisonPlan,
) -> PromotedFleetLedger:
    """Verify a persisted worker bundle before accepting it into the fleet ledger."""

    from .kumar2024_promoted_worker import (
        _shard_result_from_payload,
        verify_promoted_worker_bundle,
    )

    root = Path(worker_root).resolve()
    verification = verify_promoted_worker_bundle(root, binding_root=binding_root)
    if verification.get("verified") is not True:
        raise ValueError("worker bundle verifier did not return a verified receipt")
    if verification.get("binding_bundle_sha256") != ledger.authority.binding_bundle_sha256:
        raise ValueError("worker bundle belongs to a different fleet binding bundle")
    if verification.get("execution_plan_sha256") != ledger.authority.execution_plan_sha256:
        raise ValueError("worker bundle belongs to a different fleet execution plan")
    if verification.get("shard_spec_sha256") != lease.shard_spec_sha256:
        raise ValueError("worker bundle verifier receipt names a different lease shard")

    raw = json.loads((root / "shard_result.json").read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("worker shard_result.json must contain a JSON object")
    shard_result = _shard_result_from_payload(raw)
    if verification.get("shard_result_sha256") != shard_result.sha256:
        raise ValueError("worker verifier receipt and shard-result payload differ")

    return record_worker_artifact(
        ledger,
        lease,
        worker_bundle_sha256=verification["worker_bundle_sha256"],
        shard_result=shard_result,
        execution_plan=execution_plan,
        comparison_plan=comparison_plan,
    )


@dataclass(frozen=True, slots=True)
class PromotedFleetAssemblyManifest:
    """Content-addressed proof that a complete fleet was assembled exactly once."""

    fleet_authority_sha256: str
    fleet_ledger_sha256: str
    execution_plan_sha256: str
    comparison_plan_sha256: str
    accepted_results: tuple[PromotedFleetAcceptedResult, ...]
    execution_assembly_sha256: str
    attempted_records: int
    schema_version: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("PromotedFleetAssemblyManifest schema_version must be 1")
        for name in (
            "fleet_authority_sha256",
            "fleet_ledger_sha256",
            "execution_plan_sha256",
            "comparison_plan_sha256",
            "execution_assembly_sha256",
        ):
            object.__setattr__(self, name, _sha256(name, getattr(self, name)))
        object.__setattr__(
            self,
            "attempted_records",
            _exact_nonnegative_int("attempted_records", self.attempted_records),
        )
        accepted = tuple(sorted(self.accepted_results, key=lambda item: item.shard_id))
        if not accepted:
            raise ValueError("fleet assembly manifest requires accepted worker artifacts")
        if any(not isinstance(item, PromotedFleetAcceptedResult) for item in accepted):
            raise TypeError(
                "accepted_results must contain PromotedFleetAcceptedResult objects"
            )
        ids = [item.shard_id for item in accepted]
        shas = [item.shard_spec_sha256 for item in accepted]
        if len(set(ids)) != len(ids) or len(set(shas)) != len(shas):
            raise ValueError("fleet assembly accepted results must be unique")
        object.__setattr__(self, "accepted_results", accepted)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "fleet_authority_sha256": self.fleet_authority_sha256,
            "fleet_ledger_sha256": self.fleet_ledger_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "comparison_plan_sha256": self.comparison_plan_sha256,
            "accepted_results": [
                {**item.to_dict(), "accepted_result_sha256": item.sha256}
                for item in self.accepted_results
            ],
            "expected_shards": len(self.accepted_results),
            "attempted_records": self.attempted_records,
            "execution_assembly_sha256": self.execution_assembly_sha256,
            "global_analysis_performed": True,
            "external_floor_claim_generated": False,
            "orion_comparison_permitted": False,
            "interpretation_boundary": (
                "assembly proves orchestration completeness and delegates statistics "
                "to the preregistered comparison authority; scientific interpretation "
                "and any ORION comparison remain separate claim-authority steps"
            ),
        }

    @property
    def sha256(self) -> str:
        return _identity_sha256(
            "neuros.kumar2024_promoted_fleet_assembly_manifest.v1",
            self.to_dict(),
        )


def assemble_promoted_fleet(
    ledger: PromotedFleetLedger,
    shard_results: Sequence[PromotedShardResult],
    *,
    execution_plan: PromotedExecutionPlan,
    comparison_plan: Kumar2024ComparisonPlan,
) -> dict[str, Any]:
    """Assemble only a complete fleet whose accepted artifact identities match."""

    if not isinstance(ledger, PromotedFleetLedger):
        raise TypeError("ledger must be PromotedFleetLedger")
    if ledger.authority.execution_plan_sha256 != execution_plan.sha256:
        raise ValueError("fleet authority and execution plan differ")
    if execution_plan.template.comparison_plan_sha256 != comparison_plan.sha256:
        raise ValueError("execution plan and comparison plan differ")
    if not ledger.complete:
        raise ValueError(
            "cannot assemble incomplete promoted fleet: "
            f"accepted={len(ledger.accepted_results)}, "
            f"expected={ledger.authority.expected_shards}, "
            f"exhausted={len(ledger.exhausted_without_artifact)}"
        )

    accepted = ledger.accepted_result_map
    observed: dict[str, PromotedShardResult] = {}
    for result in shard_results:
        if not isinstance(result, PromotedShardResult):
            raise TypeError("shard_results must contain PromotedShardResult objects")
        shard_sha = result.shard_spec_sha256
        if shard_sha in observed:
            raise ValueError(f"duplicate fleet shard result for {shard_sha}")
        if shard_sha not in accepted:
            raise ValueError(f"foreign fleet shard result for {shard_sha}")
        validate_promoted_shard_result(
            result,
            execution_plan=execution_plan,
            comparison_plan=comparison_plan,
        )
        if result.sha256 != accepted[shard_sha].shard_result_sha256:
            raise ValueError("fleet shard result content differs from accepted ledger identity")
        observed[shard_sha] = result

    missing = sorted(set(accepted) - set(observed))
    if missing:
        raise ValueError(
            f"fleet assembly missing {len(missing)} accepted shard result(s): {missing[:5]}"
        )

    execution_assembly = assemble_promoted_execution(
        tuple(observed[sha] for sha in sorted(observed)),
        execution_plan=execution_plan,
        comparison_plan=comparison_plan,
    )
    execution_assembly_sha = _identity_sha256(
        "neuros.kumar2024_promoted_execution_assembly.v1",
        execution_assembly,
    )
    manifest = PromotedFleetAssemblyManifest(
        fleet_authority_sha256=ledger.authority.sha256,
        fleet_ledger_sha256=ledger.sha256,
        execution_plan_sha256=execution_plan.sha256,
        comparison_plan_sha256=comparison_plan.sha256,
        accepted_results=ledger.accepted_results,
        execution_assembly_sha256=execution_assembly_sha,
        attempted_records=len(ledger.attempts),
    )
    return {
        "schema_version": 1,
        "fleet_assembly_manifest": {
            **manifest.to_dict(),
            "fleet_assembly_manifest_sha256": manifest.sha256,
        },
        "execution_assembly": execution_assembly,
    }


__all__ = [
    "PromotedFleetAcceptedResult",
    "PromotedFleetAssemblyManifest",
    "PromotedFleetAttemptRecord",
    "PromotedFleetAuthority",
    "PromotedFleetLedger",
    "PromotedShardLease",
    "assemble_promoted_fleet",
    "build_promoted_fleet_authority",
    "record_infrastructure_failure",
    "record_verified_worker_bundle",
    "record_worker_artifact",
]
