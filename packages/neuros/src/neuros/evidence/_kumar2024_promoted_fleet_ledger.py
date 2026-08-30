"""Append-only ledger and verified worker ingestion for Kumar2024 FleetAuthority."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .kumar2024_comparison import Kumar2024ComparisonPlan
from .kumar2024_promoted_execution import PromotedExecutionPlan, PromotedShardResult, validate_promoted_shard_result
from ._kumar2024_promoted_fleet_common import (
    PromotedFleetAuthority,
    PromotedShardLease,
    _claim_boundary,
    _detail_sha256,
    _failure_code,
    _identity_sha256,
    _require_serialized_payload,
    _sha256,
    _validate_lease_against_authority,
)
from ._kumar2024_promoted_fleet_records import PromotedFleetAcceptedResult, PromotedFleetAttemptRecord

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
            if record.outcome == "infrastructure_failure":
                if record.infrastructure_failure_code not in (
                    self.authority.allowed_infrastructure_failure_codes
                ):
                    raise ValueError(
                        "infrastructure failure code is not authorized by fleet authority"
                    )
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
                raise ValueError("no fleet attempt may follow a valid worker artifact")
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

        canonical.sort(key=lambda item: (item.lease.shard_id, item.lease.attempt_index))
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
        return tuple(
            sorted(
                shard_sha
                for shard_sha, records in grouped.items()
                if shard_sha not in terminal_shas
                and len(records) >= self.authority.max_attempts_per_shard
            )
        )

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
        leases: list[PromotedShardLease] = []
        grouped = self.attempts_by_shard_sha256
        for _, shard_sha in self.authority.shard_spec_sha256_by_id:
            records = grouped.get(shard_sha, ())
            if any(item.terminal for item in records):
                continue
            if len(records) >= self.authority.max_attempts_per_shard:
                continue
            leases.append(self.next_lease(shard_sha))
        return tuple(leases)

    def append(self, record: PromotedFleetAttemptRecord) -> "PromotedFleetLedger":
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
                PromotedFleetAttemptRecord.from_dict(item) for item in raw_attempts
            ),
            schema_version=payload.get("schema_version", 1),
        )
        _require_serialized_payload(
            payload,
            value.to_dict(),
            object_name="fleet ledger",
            digest_field="fleet_ledger_sha256",
            digest=value.sha256,
        )
        return value


def record_infrastructure_failure(
    ledger: PromotedFleetLedger,
    lease: PromotedShardLease,
    *,
    failure_code: str,
    failure_detail: str | None = None,
) -> PromotedFleetLedger:
    """Record a retryable failure that produced no valid worker artifact."""

    if not isinstance(ledger, PromotedFleetLedger):
        raise TypeError("ledger must be PromotedFleetLedger")
    code = _failure_code(failure_code)
    if code not in ledger.authority.allowed_infrastructure_failure_codes:
        raise ValueError("infrastructure failure code is not authorized by fleet authority")
    detail_sha = None if failure_detail is None else _detail_sha256(failure_detail)
    return ledger.append(
        PromotedFleetAttemptRecord(
            lease=lease,
            outcome="infrastructure_failure",
            infrastructure_failure_code=code,
            infrastructure_failure_detail_sha256=detail_sha,
        )
    )


def _record_worker_artifact(
    ledger: PromotedFleetLedger,
    lease: PromotedShardLease,
    *,
    worker_bundle_sha256: str,
    shard_result: PromotedShardResult,
    execution_plan: PromotedExecutionPlan,
    comparison_plan: Kumar2024ComparisonPlan,
) -> PromotedFleetLedger:
    """Internal admission primitive used only after complete bundle verification."""

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
    """Verify persisted worker evidence before terminal fleet admission."""

    from .kumar2024_promoted_worker import (
        _shard_result_from_payload,
        verify_promoted_worker_bundle,
    )

    if not isinstance(ledger, PromotedFleetLedger):
        raise TypeError("ledger must be PromotedFleetLedger")
    expected = ledger.next_lease(lease.shard_spec_sha256)
    if lease != expected:
        raise ValueError("worker bundle does not consume the exact next fleet lease")

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

    verified_files = verification.get("files")
    if not isinstance(verified_files, Mapping):
        raise ValueError("worker verifier receipt is missing verified file identities")
    expected_shard_file_sha = verified_files.get("shard_result.json")
    expected_shard_file_sha = _sha256(
        "verified shard_result.json SHA-256", expected_shard_file_sha
    )
    shard_bytes = (root / "shard_result.json").read_bytes()
    observed_file_sha = hashlib.sha256(shard_bytes).hexdigest()
    if observed_file_sha != expected_shard_file_sha:
        raise ValueError("worker shard_result.json changed after bundle verification")
    raw = json.loads(shard_bytes)
    if not isinstance(raw, Mapping):
        raise ValueError("worker shard_result.json must contain a JSON object")
    shard_result = _shard_result_from_payload(raw)
    if verification.get("shard_result_sha256") != shard_result.sha256:
        raise ValueError("worker verifier receipt and shard-result payload differ")

    return _record_worker_artifact(
        ledger,
        lease,
        worker_bundle_sha256=verification["worker_bundle_sha256"],
        shard_result=shard_result,
        execution_plan=execution_plan,
        comparison_plan=comparison_plan,
    )

__all__ = ["PromotedFleetLedger", "record_infrastructure_failure", "record_verified_worker_bundle"]
