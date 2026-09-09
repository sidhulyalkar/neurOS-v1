"""Score-blind fleet lease and settlement authority for Kumar2024 GPU workers.

This provider-neutral, standard-library-only control plane freezes the permitted
EEGNet shard roster, requires claim-before-invocation, admits retries only after
trusted infrastructure failure, and reconstructs completion from cryptographic
artifacts rather than scientific outcomes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

FLEET_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_authority.v1"
PREFLIGHT_SCHEMA = "neuros.nsq_kumar2024_gpu_preflight_admission.v1"
LEASE_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_lease.v1"
CLAIM_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_claim.v1"
FAILURE_SCHEMA = "neuros.nsq_kumar2024_gpu_infrastructure_failure.v1"
SETTLEMENT_SCHEMA = "neuros.nsq_kumar2024_gpu_artifact_settlement.v1"
LEDGER_SCHEMA = "neuros.nsq_kumar2024_gpu_settlement_ledger.v1"

PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION = "07a6c5fa5f212d54aae408237f14bb56f2f6eee9"
EEGNET_METHOD_ID = "braindecode-eegnet"
CALIBRATION_FRONTIER = (0, 1, 2, 5, 10)
DEFAULT_MAX_ATTEMPTS = 3
ALLOWED_INFRASTRUCTURE_FAILURES = (
    "provider_preemption",
    "provider_timeout",
    "node_loss",
    "bootstrap_failure",
    "environment_mismatch",
    "artifact_upload_failure",
)
SYSTEMS_ONLY_PROVIDER_SELECTION_BASIS = (
    "wall_clock_seconds",
    "provider_cost",
    "provider_availability",
    "accelerator_identity",
    "environment_identity",
)
FORBIDDEN_CONTROL_KEYS = frozenset(
    {
        "accuracy",
        "balanced_accuracy",
        "auc",
        "auroc",
        "f1",
        "loss",
        "logits",
        "method_ranking",
        "predictions",
        "probabilities",
        "scientific_score",
        "final_assessment_metric",
    }
)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()


def _identity(schema: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({"schema": schema, "payload": payload})).hexdigest()


def _sha(name: str, value: Any) -> str:
    text = str(value)
    if (
        text != text.strip()
        or text != text.lower()
        or len(text) != 64
        or any(char not in "0123456789abcdef" for char in text)
    ):
        raise ValueError(f"{name} must be a canonical lowercase SHA-256")
    return text


def _revision(name: str, value: Any) -> str:
    text = str(value)
    if (
        text != text.strip()
        or text != text.lower()
        or len(text) != 40
        or any(char not in "0123456789abcdef" for char in text)
    ):
        raise ValueError(f"{name} must be an exact lowercase 40-character git revision")
    return text


def _identifier(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _positive_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _nonnegative_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _elapsed(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("elapsed_seconds must be a finite positive real")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError("elapsed_seconds must be a finite positive real")
    return result


def _frontier(value: Sequence[int]) -> tuple[int, ...]:
    result = tuple(value)
    if result != CALIBRATION_FRONTIER:
        raise ValueError(f"budgets_per_class must equal {CALIBRATION_FRONTIER}")
    return result


def reject_scientific_fields(value: Any, path: str = "$") -> None:
    """Fail if scientific outcome fields cross the fleet control-plane boundary."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string control-plane key at {path}")
            if key.lower() in FORBIDDEN_CONTROL_KEYS:
                raise ValueError(f"scientific field {key!r} is forbidden at {path}")
            reject_scientific_fields(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            reject_scientific_fields(item, f"{path}[{index}]")


@dataclass(frozen=True)
class PreflightAdmission:
    gpu_execution_authority_revision: str
    gpu_binding_sha256: str
    execution_plan_sha256: str
    environment_authority_sha256: str
    gpu_worker_receipt_sha256: str
    accelerator_name: str
    elapsed_seconds: float
    numerical_result_interpretable: bool = False
    global_analysis_performed: bool = False
    external_floor_claim_generated: bool = False
    orion_comparison_permitted: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "gpu_execution_authority_revision",
            _revision(
                "gpu_execution_authority_revision",
                self.gpu_execution_authority_revision,
            ),
        )
        for field in (
            "gpu_binding_sha256",
            "execution_plan_sha256",
            "environment_authority_sha256",
            "gpu_worker_receipt_sha256",
        ):
            object.__setattr__(self, field, _sha(field, getattr(self, field)))
        accelerator = _identifier("accelerator_name", self.accelerator_name)
        if "T4" not in accelerator.upper():
            raise ValueError("preflight admission requires the fixed T4-class accelerator")
        object.__setattr__(self, "accelerator_name", accelerator)
        object.__setattr__(self, "elapsed_seconds", _elapsed(self.elapsed_seconds))
        for field in (
            "numerical_result_interpretable",
            "global_analysis_performed",
            "external_floor_claim_generated",
            "orion_comparison_permitted",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"preflight admission requires {field}=false")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "gpu_execution_authority_revision": self.gpu_execution_authority_revision,
            "gpu_binding_sha256": self.gpu_binding_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "environment_authority_sha256": self.environment_authority_sha256,
            "gpu_worker_receipt_sha256": self.gpu_worker_receipt_sha256,
            "accelerator_name": self.accelerator_name,
            "elapsed_seconds": self.elapsed_seconds,
            "numerical_result_interpretable": False,
            "global_analysis_performed": False,
            "external_floor_claim_generated": False,
            "orion_comparison_permitted": False,
            "admission_inputs": list(SYSTEMS_ONLY_PROVIDER_SELECTION_BASIS),
        }

    @property
    def sha256(self) -> str:
        return _identity(PREFLIGHT_SCHEMA, self.payload())


@dataclass(frozen=True)
class FleetAuthority:
    preflight_admission_sha256: str
    gpu_execution_authority_revision: str
    gpu_binding_sha256: str
    execution_plan_sha256: str
    environment_authority_sha256: str
    expected_shard_count: int
    max_attempts_per_lease: int = DEFAULT_MAX_ATTEMPTS
    method_id: str = EEGNET_METHOD_ID
    budgets_per_class: tuple[int, ...] = CALIBRATION_FRONTIER

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "preflight_admission_sha256",
            _sha("preflight_admission_sha256", self.preflight_admission_sha256),
        )
        object.__setattr__(
            self,
            "gpu_execution_authority_revision",
            _revision(
                "gpu_execution_authority_revision",
                self.gpu_execution_authority_revision,
            ),
        )
        for field in (
            "gpu_binding_sha256",
            "execution_plan_sha256",
            "environment_authority_sha256",
        ):
            object.__setattr__(self, field, _sha(field, getattr(self, field)))
        object.__setattr__(
            self,
            "expected_shard_count",
            _positive_int("expected_shard_count", self.expected_shard_count),
        )
        object.__setattr__(
            self,
            "max_attempts_per_lease",
            _positive_int("max_attempts_per_lease", self.max_attempts_per_lease),
        )
        if self.method_id != EEGNET_METHOD_ID:
            raise ValueError(f"fleet v1 is fixed to {EEGNET_METHOD_ID!r}")
        object.__setattr__(self, "budgets_per_class", _frontier(self.budgets_per_class))

    @classmethod
    def from_preflight(
        cls,
        preflight: PreflightAdmission,
        *,
        expected_shard_count: int,
        max_attempts_per_lease: int = DEFAULT_MAX_ATTEMPTS,
    ) -> "FleetAuthority":
        if (
            preflight.gpu_execution_authority_revision
            != PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION
        ):
            raise ValueError("preflight does not name the promoted GPU execution authority")
        return cls(
            preflight.sha256,
            preflight.gpu_execution_authority_revision,
            preflight.gpu_binding_sha256,
            preflight.execution_plan_sha256,
            preflight.environment_authority_sha256,
            expected_shard_count,
            max_attempts_per_lease,
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "artifact_kind": "score_blind_gpu_fleet_execution_authority",
            "preflight_admission_sha256": self.preflight_admission_sha256,
            "gpu_execution_authority_revision": self.gpu_execution_authority_revision,
            "gpu_binding_sha256": self.gpu_binding_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "environment_authority_sha256": self.environment_authority_sha256,
            "expected_shard_count": self.expected_shard_count,
            "max_attempts_per_lease": self.max_attempts_per_lease,
            "method_id": self.method_id,
            "budgets_per_class": list(self.budgets_per_class),
            "allowed_infrastructure_failures": list(ALLOWED_INFRASTRUCTURE_FAILURES),
            "provider_selection_basis": list(SYSTEMS_ONLY_PROVIDER_SELECTION_BASIS),
            "scientific_outcome_may_control_retry": False,
            "scientific_outcome_may_control_provider_selection": False,
            "orion_comparison_permitted": False,
        }

    @property
    def sha256(self) -> str:
        return _identity(FLEET_SCHEMA, self.payload())


@dataclass(frozen=True)
class LeaseSpec:
    fleet_authority_sha256: str
    gpu_binding_sha256: str
    execution_plan_sha256: str
    environment_authority_sha256: str
    shard_spec_sha256: str
    ordinal: int
    subject: int
    target_session: str
    split_seed: int
    model_seed: int
    max_attempts: int
    method_id: str = EEGNET_METHOD_ID
    budgets_per_class: tuple[int, ...] = CALIBRATION_FRONTIER

    def __post_init__(self) -> None:
        for field in (
            "fleet_authority_sha256",
            "gpu_binding_sha256",
            "execution_plan_sha256",
            "environment_authority_sha256",
            "shard_spec_sha256",
        ):
            object.__setattr__(self, field, _sha(field, getattr(self, field)))
        object.__setattr__(self, "ordinal", _nonnegative_int("ordinal", self.ordinal))
        object.__setattr__(self, "subject", _positive_int("subject", self.subject))
        object.__setattr__(
            self,
            "target_session",
            _identifier("target_session", self.target_session),
        )
        object.__setattr__(
            self,
            "split_seed",
            _nonnegative_int("split_seed", self.split_seed),
        )
        object.__setattr__(
            self,
            "model_seed",
            _nonnegative_int("model_seed", self.model_seed),
        )
        object.__setattr__(
            self,
            "max_attempts",
            _positive_int("max_attempts", self.max_attempts),
        )
        if self.method_id != EEGNET_METHOD_ID:
            raise ValueError("fleet lease is not EEGNet")
        object.__setattr__(self, "budgets_per_class", _frontier(self.budgets_per_class))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "fleet_authority_sha256": self.fleet_authority_sha256,
            "gpu_binding_sha256": self.gpu_binding_sha256,
            "execution_plan_sha256": self.execution_plan_sha256,
            "environment_authority_sha256": self.environment_authority_sha256,
            "shard_spec_sha256": self.shard_spec_sha256,
            "ordinal": self.ordinal,
            "subject": self.subject,
            "target_session": self.target_session,
            "split_seed": self.split_seed,
            "method_id": self.method_id,
            "model_seed": self.model_seed,
            "budgets_per_class": list(self.budgets_per_class),
            "max_attempts": self.max_attempts,
        }

    @property
    def sha256(self) -> str:
        return _identity(LEASE_SCHEMA, self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "lease_sha256": self.sha256}


@dataclass(frozen=True)
class ClaimEvent:
    lease_sha256: str
    shard_spec_sha256: str
    attempt_number: int
    worker_id: str
    provider: str
    provider_run_id: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "lease_sha256",
            _sha("lease_sha256", self.lease_sha256),
        )
        object.__setattr__(
            self,
            "shard_spec_sha256",
            _sha("shard_spec_sha256", self.shard_spec_sha256),
        )
        object.__setattr__(
            self,
            "attempt_number",
            _positive_int("attempt_number", self.attempt_number),
        )
        for field in ("worker_id", "provider", "provider_run_id"):
            object.__setattr__(self, field, _identifier(field, getattr(self, field)))

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_type": "claim",
            "lease_sha256": self.lease_sha256,
            "shard_spec_sha256": self.shard_spec_sha256,
            "attempt_number": self.attempt_number,
            "worker_id": self.worker_id,
            "provider": self.provider,
            "provider_run_id": self.provider_run_id,
        }

    @property
    def sha256(self) -> str:
        return _identity(CLAIM_SCHEMA, self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "claim_sha256": self.sha256}


@dataclass(frozen=True)
class InfrastructureFailureEvent:
    claim_sha256: str
    lease_sha256: str
    shard_spec_sha256: str
    attempt_number: int
    failure_kind: str
    failure_evidence_sha256: str
    valid_worker_artifact_exists: bool = False

    def __post_init__(self) -> None:
        for field in (
            "claim_sha256",
            "lease_sha256",
            "shard_spec_sha256",
            "failure_evidence_sha256",
        ):
            object.__setattr__(self, field, _sha(field, getattr(self, field)))
        object.__setattr__(
            self,
            "attempt_number",
            _positive_int("attempt_number", self.attempt_number),
        )
        if self.failure_kind not in ALLOWED_INFRASTRUCTURE_FAILURES:
            raise ValueError(f"unsupported infrastructure failure kind: {self.failure_kind!r}")
        if self.valid_worker_artifact_exists is not False:
            raise ValueError(
                "infrastructure retry requires proof that no valid artifact exists"
            )

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_type": "infrastructure_failure",
            "claim_sha256": self.claim_sha256,
            "lease_sha256": self.lease_sha256,
            "shard_spec_sha256": self.shard_spec_sha256,
            "attempt_number": self.attempt_number,
            "failure_kind": self.failure_kind,
            "failure_evidence_sha256": self.failure_evidence_sha256,
            "valid_worker_artifact_exists": False,
        }

    @property
    def sha256(self) -> str:
        return _identity(FAILURE_SCHEMA, self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "outcome_sha256": self.sha256}


@dataclass(frozen=True)
class ArtifactSettlementEvent:
    claim_sha256: str
    lease_sha256: str
    shard_spec_sha256: str
    attempt_number: int
    gpu_worker_receipt_sha256: str
    worker_bundle_sha256: str
    learned_state_sha256: str
    environment_authority_sha256: str
    provider_receipt_sha256: str
    accelerator_name: str
    numerical_result_interpretable: bool = False
    global_analysis_performed: bool = False
    external_floor_claim_generated: bool = False
    orion_comparison_permitted: bool = False

    def __post_init__(self) -> None:
        for field in (
            "claim_sha256",
            "lease_sha256",
            "shard_spec_sha256",
            "gpu_worker_receipt_sha256",
            "worker_bundle_sha256",
            "learned_state_sha256",
            "environment_authority_sha256",
            "provider_receipt_sha256",
        ):
            object.__setattr__(self, field, _sha(field, getattr(self, field)))
        object.__setattr__(
            self,
            "attempt_number",
            _positive_int("attempt_number", self.attempt_number),
        )
        accelerator = _identifier("accelerator_name", self.accelerator_name)
        if "T4" not in accelerator.upper():
            raise ValueError("artifact settlement requires the fixed T4-class accelerator")
        object.__setattr__(self, "accelerator_name", accelerator)
        for field in (
            "numerical_result_interpretable",
            "global_analysis_performed",
            "external_floor_claim_generated",
            "orion_comparison_permitted",
        ):
            if getattr(self, field) is not False:
                raise ValueError(f"artifact settlement requires {field}=false")

    def payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "event_type": "artifact_settlement",
            "claim_sha256": self.claim_sha256,
            "lease_sha256": self.lease_sha256,
            "shard_spec_sha256": self.shard_spec_sha256,
            "attempt_number": self.attempt_number,
            "gpu_worker_receipt_sha256": self.gpu_worker_receipt_sha256,
            "worker_bundle_sha256": self.worker_bundle_sha256,
            "learned_state_sha256": self.learned_state_sha256,
            "environment_authority_sha256": self.environment_authority_sha256,
            "provider_receipt_sha256": self.provider_receipt_sha256,
            "accelerator_name": self.accelerator_name,
            "numerical_result_interpretable": False,
            "global_analysis_performed": False,
            "external_floor_claim_generated": False,
            "orion_comparison_permitted": False,
        }

    @property
    def sha256(self) -> str:
        return _identity(SETTLEMENT_SCHEMA, self.payload())

    def to_dict(self) -> dict[str, Any]:
        return {**self.payload(), "outcome_sha256": self.sha256}


OutcomeEvent = InfrastructureFailureEvent | ArtifactSettlementEvent


def build_leases(
    authority: FleetAuthority,
    shards: Iterable[Mapping[str, Any]],
) -> tuple[LeaseSpec, ...]:
    """Freeze an exact EEGNet shard roster into canonically ordered leases."""
    required = {
        "subject",
        "target_session",
        "split_seed",
        "method_id",
        "model_seed",
        "budgets_per_class",
        "shard_spec_sha256",
    }
    normalized = []
    for raw in shards:
        reject_scientific_fields(raw)
        if set(raw) != required:
            raise ValueError(
                f"shard descriptor keys differ: missing={sorted(required - set(raw))}, "
                f"extra={sorted(set(raw) - required)}"
            )
        if raw["method_id"] != authority.method_id:
            raise ValueError("fleet contains a non-EEGNet shard")
        normalized.append(
            (
                _positive_int("subject", raw["subject"]),
                _identifier("target_session", raw["target_session"]),
                _nonnegative_int("split_seed", raw["split_seed"]),
                _nonnegative_int("model_seed", raw["model_seed"]),
                _sha("shard_spec_sha256", raw["shard_spec_sha256"]),
                _frontier(raw["budgets_per_class"]),
            )
        )
    if len(normalized) != authority.expected_shard_count:
        raise ValueError("fleet shard count differs from authority")
    if len({item[4] for item in normalized}) != len(normalized):
        raise ValueError("fleet contains duplicate shard identities")
    normalized.sort(key=lambda item: item[:5])
    return tuple(
        LeaseSpec(
            authority.sha256,
            authority.gpu_binding_sha256,
            authority.execution_plan_sha256,
            authority.environment_authority_sha256,
            shard_sha,
            ordinal,
            subject,
            session,
            split_seed,
            model_seed,
            authority.max_attempts_per_lease,
            authority.method_id,
            frontier,
        )
        for ordinal, (
            subject,
            session,
            split_seed,
            model_seed,
            shard_sha,
            frontier,
        ) in enumerate(normalized)
    )


def reconstruct_settlement(
    authority: FleetAuthority,
    leases: Sequence[LeaseSpec],
    claims: Sequence[ClaimEvent],
    outcomes: Sequence[OutcomeEvent],
) -> dict[str, Any]:
    """Reconstruct score-blind fleet state from immutable lease/attempt events."""
    if len(leases) != authority.expected_shard_count:
        raise ValueError("ledger lease count differs from authority")

    lease_by_sha: dict[str, LeaseSpec] = {}
    shard_ids = set()
    for lease in leases:
        if (
            lease.fleet_authority_sha256 != authority.sha256
            or lease.gpu_binding_sha256 != authority.gpu_binding_sha256
            or lease.execution_plan_sha256 != authority.execution_plan_sha256
            or lease.environment_authority_sha256
            != authority.environment_authority_sha256
            or lease.max_attempts != authority.max_attempts_per_lease
        ):
            raise ValueError("lease authority binding drifted")
        if lease.sha256 in lease_by_sha or lease.shard_spec_sha256 in shard_ids:
            raise ValueError("duplicate lease or shard identity")
        lease_by_sha[lease.sha256] = lease
        shard_ids.add(lease.shard_spec_sha256)
    if {lease.ordinal for lease in leases} != set(range(authority.expected_shard_count)):
        raise ValueError("lease ordinals are not a complete zero-based range")

    claim_by_sha: dict[str, ClaimEvent] = {}
    claim_by_slot: dict[tuple[str, int], ClaimEvent] = {}
    provider_runs = set()
    for claim in claims:
        lease = lease_by_sha.get(claim.lease_sha256)
        if lease is None or claim.shard_spec_sha256 != lease.shard_spec_sha256:
            raise ValueError("claim does not match a known lease")
        if claim.attempt_number > lease.max_attempts:
            raise ValueError("claim exceeds retry ceiling")
        slot = (claim.lease_sha256, claim.attempt_number)
        provider_slot = (claim.provider, claim.provider_run_id)
        if slot in claim_by_slot or provider_slot in provider_runs:
            raise ValueError("duplicate lease attempt or provider run")
        claim_by_slot[slot] = claim
        claim_by_sha[claim.sha256] = claim
        provider_runs.add(provider_slot)

    outcome_by_claim: dict[str, OutcomeEvent] = {}
    for outcome in outcomes:
        claim = claim_by_sha.get(outcome.claim_sha256)
        if claim is None:
            raise ValueError("outcome has no preceding claim")
        if outcome.claim_sha256 in outcome_by_claim:
            raise ValueError("claim has multiple terminal outcomes")
        if (
            outcome.lease_sha256 != claim.lease_sha256
            or outcome.shard_spec_sha256 != claim.shard_spec_sha256
            or outcome.attempt_number != claim.attempt_number
        ):
            raise ValueError("outcome binding differs from claim")
        if (
            isinstance(outcome, ArtifactSettlementEvent)
            and outcome.environment_authority_sha256
            != authority.environment_authority_sha256
        ):
            raise ValueError("artifact settlement environment authority differs from fleet")
        outcome_by_claim[outcome.claim_sha256] = outcome

    accepted: list[ArtifactSettlementEvent] = []
    failures: list[InfrastructureFailureEvent] = []
    pending: set[str] = set()
    for lease in sorted(leases, key=lambda item: item.ordinal):
        lease_claims = sorted(
            (
                claim
                for (lease_sha, _), claim in claim_by_slot.items()
                if lease_sha == lease.sha256
            ),
            key=lambda item: item.attempt_number,
        )
        if not lease_claims:
            pending.add(lease.sha256)
            continue
        if [item.attempt_number for item in lease_claims] != list(
            range(1, len(lease_claims) + 1)
        ):
            raise ValueError("lease attempts are not contiguous from 1")
        artifact_seen = False
        for index, claim in enumerate(lease_claims):
            if artifact_seen:
                raise ValueError("attempt exists after accepted artifact")
            if index:
                prior = outcome_by_claim.get(lease_claims[index - 1].sha256)
                if not isinstance(prior, InfrastructureFailureEvent):
                    raise ValueError("retry requires trusted infrastructure failure")
            outcome = outcome_by_claim.get(claim.sha256)
            if outcome is None:
                if index != len(lease_claims) - 1:
                    raise ValueError("only latest claim may remain unsettled")
                pending.add(lease.sha256)
            elif isinstance(outcome, InfrastructureFailureEvent):
                failures.append(outcome)
            else:
                accepted.append(outcome)
                artifact_seen = True
        if artifact_seen and not isinstance(
            outcome_by_claim.get(lease_claims[-1].sha256),
            ArtifactSettlementEvent,
        ):
            raise ValueError("accepted artifact must terminate the lease")
        if not artifact_seen and lease_claims[-1].sha256 in outcome_by_claim:
            pending.add(lease.sha256)

    if len({item.lease_sha256 for item in accepted}) != len(accepted):
        raise ValueError("multiple accepted artifacts exist for a lease")
    complete = len(accepted) == authority.expected_shard_count and not pending
    payload = {
        "schema_version": 1,
        "fleet_authority_sha256": authority.sha256,
        "environment_authority_sha256": authority.environment_authority_sha256,
        "expected_lease_count": authority.expected_shard_count,
        "lease_sha256s": [
            item.sha256 for item in sorted(leases, key=lambda item: item.ordinal)
        ],
        "claim_sha256s": sorted(item.sha256 for item in claims),
        "outcome_sha256s": sorted(item.sha256 for item in outcomes),
        "accepted_artifact_count": len(accepted),
        "infrastructure_failure_count": len(failures),
        "pending_lease_count": len(pending),
        "complete": complete,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "orion_comparison_permitted": False,
    }
    return {
        **payload,
        "settlement_ledger_sha256": _identity(LEDGER_SCHEMA, payload),
        "accepted_lease_sha256s": sorted(item.lease_sha256 for item in accepted),
        "pending_lease_sha256s": sorted(pending),
    }


def _write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    reject_scientific_fields(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as stream:
            fd = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    return path


def _read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read sealed control-plane record: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError("sealed control-plane record must be a JSON object")
    reject_scientific_fields(value)
    return value


def _read_sealed(
    path: Path,
    *,
    schema: str,
    identity_field: str,
    expected_identity: str | None = None,
) -> dict[str, Any]:
    value = _read_object(path)
    declared = _sha(identity_field, value.get(identity_field, ""))
    payload = dict(value)
    payload.pop(identity_field, None)
    recomputed = _identity(schema, payload)
    if declared != recomputed:
        raise ValueError(f"persisted {identity_field} seal does not match record contents")
    if expected_identity is not None and declared != expected_identity:
        raise ValueError(f"persisted {identity_field} differs from expected authority")
    return value


def _read_outcome(path: Path) -> dict[str, Any]:
    raw = _read_object(path)
    event_type = raw.get("event_type")
    if event_type == "infrastructure_failure":
        schema = FAILURE_SCHEMA
    elif event_type == "artifact_settlement":
        schema = SETTLEMENT_SCHEMA
    else:
        raise ValueError("persisted outcome has unsupported event_type")
    declared = _sha("outcome_sha256", raw.get("outcome_sha256", ""))
    payload = dict(raw)
    payload.pop("outcome_sha256", None)
    if _identity(schema, payload) != declared:
        raise ValueError("persisted outcome seal does not match record contents")
    return raw


def persist_lease(store_root: str | Path, lease: LeaseSpec) -> Path:
    return _write_once(
        Path(store_root) / "leases" / f"{lease.sha256}.json",
        lease.to_dict(),
    )


def _claim_path(root: Path, lease_sha: str, attempt: int) -> Path:
    return root / "claims" / lease_sha / f"attempt-{attempt:04d}.json"


def _outcome_path(root: Path, lease_sha: str, attempt: int) -> Path:
    return root / "outcomes" / lease_sha / f"attempt-{attempt:04d}.json"


def persist_claim(
    store_root: str | Path,
    lease: LeaseSpec,
    claim: ClaimEvent,
) -> Path:
    """Create-if-absent claim after verifying the persisted lease and retry chain."""
    root = Path(store_root)
    lease_path = root / "leases" / f"{lease.sha256}.json"
    if not lease_path.is_file():
        raise FileNotFoundError("lease must be persisted before claim")
    _read_sealed(
        lease_path,
        schema=LEASE_SCHEMA,
        identity_field="lease_sha256",
        expected_identity=lease.sha256,
    )
    if (
        claim.lease_sha256 != lease.sha256
        or claim.shard_spec_sha256 != lease.shard_spec_sha256
    ):
        raise ValueError("claim does not match persisted lease")
    if claim.attempt_number > lease.max_attempts:
        raise ValueError("claim exceeds retry ceiling")

    if claim.attempt_number > 1:
        previous_attempt = claim.attempt_number - 1
        previous_claim_path = _claim_path(root, lease.sha256, previous_attempt)
        previous_path = _outcome_path(root, lease.sha256, previous_attempt)
        if not previous_claim_path.is_file() or not previous_path.is_file():
            raise ValueError("retry requires prior terminal outcome")
        previous_claim = _read_sealed(
            previous_claim_path,
            schema=CLAIM_SCHEMA,
            identity_field="claim_sha256",
        )
        previous = _read_outcome(previous_path)
        if (
            previous.get("event_type") != "infrastructure_failure"
            or previous.get("valid_worker_artifact_exists") is not False
            or previous.get("claim_sha256") != previous_claim.get("claim_sha256")
            or previous.get("lease_sha256") != lease.sha256
            or previous.get("shard_spec_sha256") != lease.shard_spec_sha256
            or previous.get("attempt_number") != previous_attempt
        ):
            raise ValueError(
                "retry requires trusted infrastructure failure with no valid artifact"
            )

    return _write_once(
        _claim_path(root, lease.sha256, claim.attempt_number),
        claim.to_dict(),
    )


def prepare_attempt_namespace(
    store_root: str | Path,
    claim: ClaimEvent,
) -> Path:
    """Create a write-once attempt directory after revalidating the atomic claim."""
    root = Path(store_root)
    claim_path = _claim_path(root, claim.lease_sha256, claim.attempt_number)
    if not claim_path.is_file():
        raise FileNotFoundError("attempt namespace requires persisted claim")
    stored = _read_sealed(
        claim_path,
        schema=CLAIM_SCHEMA,
        identity_field="claim_sha256",
        expected_identity=claim.sha256,
    )
    if (
        stored.get("lease_sha256") != claim.lease_sha256
        or stored.get("shard_spec_sha256") != claim.shard_spec_sha256
        or stored.get("attempt_number") != claim.attempt_number
    ):
        raise ValueError("persisted claim differs from requested attempt")
    path = root / "attempts" / claim.lease_sha256 / (
        f"attempt-{claim.attempt_number:04d}-{claim.sha256[:16]}"
    )
    path.mkdir(parents=True, exist_ok=False)
    return path


def persist_outcome(
    store_root: str | Path,
    outcome: OutcomeEvent,
) -> Path:
    """Persist one terminal outcome only after revalidating its sealed claim."""
    root = Path(store_root)
    claim_path = _claim_path(root, outcome.lease_sha256, outcome.attempt_number)
    if not claim_path.is_file():
        raise FileNotFoundError("outcome requires persisted claim")
    claim = _read_sealed(
        claim_path,
        schema=CLAIM_SCHEMA,
        identity_field="claim_sha256",
        expected_identity=outcome.claim_sha256,
    )
    if (
        claim.get("lease_sha256") != outcome.lease_sha256
        or claim.get("shard_spec_sha256") != outcome.shard_spec_sha256
        or claim.get("attempt_number") != outcome.attempt_number
    ):
        raise ValueError("outcome names a different persisted claim")
    return _write_once(
        _outcome_path(root, outcome.lease_sha256, outcome.attempt_number),
        outcome.to_dict(),
    )


__all__ = [
    "ALLOWED_INFRASTRUCTURE_FAILURES",
    "ArtifactSettlementEvent",
    "CALIBRATION_FRONTIER",
    "ClaimEvent",
    "FleetAuthority",
    "InfrastructureFailureEvent",
    "LeaseSpec",
    "PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION",
    "PreflightAdmission",
    "SYSTEMS_ONLY_PROVIDER_SELECTION_BASIS",
    "build_leases",
    "persist_claim",
    "persist_lease",
    "persist_outcome",
    "prepare_attempt_namespace",
    "reconstruct_settlement",
    "reject_scientific_fields",
]
