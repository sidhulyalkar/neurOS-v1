"""Independent verifier for the provider-free Kumar2024 GPU fleet packet.

This verifier intentionally does not import neurOS, the fleet authority module,
the synthetic runner, model code, provider SDKs, or scientific-result code.
It derives the frozen synthetic protocol independently and verifies the complete
on-disk evidence graph before accepting the top-level receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

FLEET_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_authority.v1"
PREFLIGHT_SCHEMA = "neuros.nsq_kumar2024_gpu_preflight_admission.v1"
LEASE_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_lease.v1"
CLAIM_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_claim.v1"
FAILURE_SCHEMA = "neuros.nsq_kumar2024_gpu_infrastructure_failure.v1"
SETTLEMENT_SCHEMA = "neuros.nsq_kumar2024_gpu_artifact_settlement.v1"
LEDGER_SCHEMA = "neuros.nsq_kumar2024_gpu_settlement_ledger.v1"
SYNTHETIC_RECEIPT_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_synthetic_qualification.v1"

PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION = "07a6c5fa5f212d54aae408237f14bb56f2f6eee9"
EEGNET_METHOD_ID = "braindecode-eegnet"
CALIBRATION_FRONTIER = [0, 1, 2, 5, 10]
MAX_ATTEMPTS = 3
LEASE_COUNT = 4
PROVIDER_A = "synthetic-provider-a"
PROVIDER_B = "synthetic-provider-b"
ALLOWED_INFRASTRUCTURE_FAILURES = [
    "provider_preemption",
    "provider_timeout",
    "node_loss",
    "bootstrap_failure",
    "environment_mismatch",
    "artifact_upload_failure",
]
PROVIDER_SELECTION_BASIS = [
    "wall_clock_seconds",
    "provider_cost",
    "provider_availability",
    "accelerator_identity",
    "environment_identity",
]
FORBIDDEN_SCIENTIFIC_KEYS = frozenset(
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
CLAIM_BOUNDARY = (
    "synthetic systems qualification of lease/claim/retry/artifact settlement only; "
    "no neural data, model execution, prediction, score, efficacy, or ORION evidence"
)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _identity(schema: str, payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({"schema": schema, "payload": payload})).hexdigest()


def _label_sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _bad_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_object_pairs,
            parse_constant=_bad_constant,
        )
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot decode JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"evidence must be a JSON object: {path}")
    _reject_scientific_fields(value)
    return value


def _reject_scientific_fields(value: Any, path: str = "$") -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"non-string key at {path}")
            if key.lower() in FORBIDDEN_SCIENTIFIC_KEYS:
                raise ValueError(f"scientific field {key!r} is forbidden at {path}")
            _reject_scientific_fields(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_scientific_fields(item, f"{path}[{index}]")


def _preflight_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gpu_execution_authority_revision": PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION,
        "gpu_binding_sha256": _label_sha("synthetic-gpu-binding"),
        "execution_plan_sha256": _label_sha("synthetic-execution-plan"),
        "environment_authority_sha256": _label_sha("synthetic-environment-authority"),
        "gpu_worker_receipt_sha256": _label_sha("synthetic-preflight-worker-receipt"),
        "accelerator_name": "Tesla T4 (synthetic)",
        "elapsed_seconds": 1.0,
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "admission_inputs": list(PROVIDER_SELECTION_BASIS),
    }


def _fleet_payload() -> dict[str, Any]:
    preflight_sha = _identity(PREFLIGHT_SCHEMA, _preflight_payload())
    return {
        "schema_version": 1,
        "artifact_kind": "score_blind_gpu_fleet_execution_authority",
        "preflight_admission_sha256": preflight_sha,
        "gpu_execution_authority_revision": PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION,
        "gpu_binding_sha256": _label_sha("synthetic-gpu-binding"),
        "execution_plan_sha256": _label_sha("synthetic-execution-plan"),
        "environment_authority_sha256": _label_sha("synthetic-environment-authority"),
        "expected_shard_count": LEASE_COUNT,
        "max_attempts_per_lease": MAX_ATTEMPTS,
        "method_id": EEGNET_METHOD_ID,
        "budgets_per_class": list(CALIBRATION_FRONTIER),
        "allowed_infrastructure_failures": list(ALLOWED_INFRASTRUCTURE_FAILURES),
        "provider_selection_basis": list(PROVIDER_SELECTION_BASIS),
        "scientific_outcome_may_control_retry": False,
        "scientific_outcome_may_control_provider_selection": False,
        "orion_comparison_permitted": False,
    }


def _expected_leases() -> list[dict[str, Any]]:
    fleet_sha = _identity(FLEET_SCHEMA, _fleet_payload())
    result = []
    for ordinal, subject in enumerate(range(1, LEASE_COUNT + 1)):
        payload = {
            "schema_version": 1,
            "fleet_authority_sha256": fleet_sha,
            "gpu_binding_sha256": _label_sha("synthetic-gpu-binding"),
            "execution_plan_sha256": _label_sha("synthetic-execution-plan"),
            "environment_authority_sha256": _label_sha("synthetic-environment-authority"),
            "shard_spec_sha256": _label_sha(f"synthetic-shard-{subject}"),
            "ordinal": ordinal,
            "subject": subject,
            "target_session": "5",
            "split_seed": 2026,
            "method_id": EEGNET_METHOD_ID,
            "model_seed": 31415 + subject - 1,
            "budgets_per_class": list(CALIBRATION_FRONTIER),
            "max_attempts": MAX_ATTEMPTS,
        }
        result.append({**payload, "lease_sha256": _identity(LEASE_SCHEMA, payload)})
    return result


def _claim_payload(lease: dict[str, Any], attempt: int, provider: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "claim",
        "lease_sha256": lease["lease_sha256"],
        "shard_spec_sha256": lease["shard_spec_sha256"],
        "attempt_number": attempt,
        "worker_id": f"synthetic-worker-{lease['ordinal']}-{attempt}",
        "provider": provider,
        "provider_run_id": f"synthetic-run-{lease['ordinal']}-{attempt}",
    }


def _expected_claims(leases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for lease in leases:
        provider = PROVIDER_A if lease["ordinal"] % 2 == 0 else PROVIDER_B
        payload = _claim_payload(lease, 1, provider)
        result.append({**payload, "claim_sha256": _identity(CLAIM_SCHEMA, payload)})
        if lease["ordinal"] == 1:
            payload = _claim_payload(lease, 2, PROVIDER_A)
            result.append({**payload, "claim_sha256": _identity(CLAIM_SCHEMA, payload)})
    return result


def _failure_payload(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_type": "infrastructure_failure",
        "claim_sha256": claim["claim_sha256"],
        "lease_sha256": claim["lease_sha256"],
        "shard_spec_sha256": claim["shard_spec_sha256"],
        "attempt_number": 1,
        "failure_kind": "provider_preemption",
        "failure_evidence_sha256": _label_sha("synthetic-provider-preemption"),
        "valid_worker_artifact_exists": False,
    }


def _settlement_payload(claim: dict[str, Any], ordinal: int) -> dict[str, Any]:
    attempt = claim["attempt_number"]
    return {
        "schema_version": 1,
        "event_type": "artifact_settlement",
        "claim_sha256": claim["claim_sha256"],
        "lease_sha256": claim["lease_sha256"],
        "shard_spec_sha256": claim["shard_spec_sha256"],
        "attempt_number": attempt,
        "gpu_worker_receipt_sha256": _label_sha(
            f"synthetic-worker-receipt-{ordinal}-{attempt}"
        ),
        "worker_bundle_sha256": _label_sha(
            f"synthetic-worker-bundle-{ordinal}-{attempt}"
        ),
        "learned_state_sha256": _label_sha(
            f"synthetic-learned-state-{ordinal}-{attempt}"
        ),
        "environment_authority_sha256": _label_sha("synthetic-environment-authority"),
        "provider_receipt_sha256": _label_sha(
            f"synthetic-provider-receipt-{ordinal}-{attempt}"
        ),
        "accelerator_name": "Tesla T4 (synthetic)",
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
    }


def _expected_outcomes(
    leases: Sequence[dict[str, Any]], claims: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    claims_by_slot = {
        (claim["lease_sha256"], claim["attempt_number"]): claim for claim in claims
    }
    result = []
    for lease in leases:
        if lease["ordinal"] == 1:
            first = claims_by_slot[(lease["lease_sha256"], 1)]
            failure = _failure_payload(first)
            result.append(
                {**failure, "outcome_sha256": _identity(FAILURE_SCHEMA, failure)}
            )
            terminal = claims_by_slot[(lease["lease_sha256"], 2)]
        else:
            terminal = claims_by_slot[(lease["lease_sha256"], 1)]
        settlement = _settlement_payload(terminal, lease["ordinal"])
        result.append(
            {**settlement, "outcome_sha256": _identity(SETTLEMENT_SCHEMA, settlement)}
        )
    return result


def _expected_invocation(claim: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "provider": claim["provider"],
        "provider_run_id": claim["provider_run_id"],
        "scientific_execution_performed": False,
    }


def _managed_topology(
    leases: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    files = {"synthetic-qualification.json"}
    directories = {"leases", "claims", "outcomes", "attempts"}
    for lease in leases:
        lease_sha = lease["lease_sha256"]
        files.add(f"leases/{lease_sha}.json")
        for prefix in ("claims", "outcomes", "attempts"):
            directories.add(f"{prefix}/{lease_sha}")
    for claim in claims:
        lease_sha = claim["lease_sha256"]
        attempt = claim["attempt_number"]
        files.add(f"claims/{lease_sha}/attempt-{attempt:04d}.json")
        files.add(f"outcomes/{lease_sha}/attempt-{attempt:04d}.json")
        attempt_dir = (
            f"attempts/{lease_sha}/"
            f"attempt-{attempt:04d}-{claim['claim_sha256'][:16]}"
        )
        directories.add(attempt_dir)
        files.add(f"{attempt_dir}/provider-invocation.json")
    return files, directories


def _walk_topology(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(dirnames):
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"symlink directory is forbidden: {path}")
            directories.add(path.relative_to(root).as_posix())
        for name in filenames:
            path = current_path / name
            if path.is_symlink():
                raise ValueError(f"symlink file is forbidden: {path}")
            files.add(path.relative_to(root).as_posix())
    return files, directories


def _require_exact(
    path: Path,
    expected: dict[str, Any],
    identity_field: str,
    schema: str,
) -> None:
    observed = _load_json(path)
    if observed != expected:
        raise ValueError(f"evidence differs from frozen protocol: {path}")
    declared = observed.get(identity_field)
    payload = dict(observed)
    payload.pop(identity_field, None)
    recomputed = _identity(schema, payload)
    if declared != recomputed:
        raise ValueError(f"{identity_field} does not match record contents: {path}")


def _reconstruct_ledger(
    leases: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    accepted = [item for item in outcomes if item["event_type"] == "artifact_settlement"]
    failures = [item for item in outcomes if item["event_type"] == "infrastructure_failure"]
    claim_by_slot = {
        (item["lease_sha256"], item["attempt_number"]): item for item in claims
    }
    outcome_by_claim = {item["claim_sha256"]: item for item in outcomes}

    pending: set[str] = set()
    for lease in leases:
        lease_claims = sorted(
            (
                claim
                for (lease_sha, _), claim in claim_by_slot.items()
                if lease_sha == lease["lease_sha256"]
            ),
            key=lambda item: item["attempt_number"],
        )
        attempts = [item["attempt_number"] for item in lease_claims]
        if attempts != list(range(1, len(attempts) + 1)):
            raise ValueError("attempts are not contiguous")
        artifact_seen = False
        for index, claim in enumerate(lease_claims):
            if artifact_seen:
                raise ValueError("attempt exists after accepted artifact")
            if index:
                prior = outcome_by_claim.get(lease_claims[index - 1]["claim_sha256"])
                if prior is None or prior["event_type"] != "infrastructure_failure":
                    raise ValueError("retry lacks trusted infrastructure failure")
                if prior.get("valid_worker_artifact_exists") is not False:
                    raise ValueError("retry follows a valid worker artifact")
            outcome = outcome_by_claim.get(claim["claim_sha256"])
            if outcome is None:
                pending.add(lease["lease_sha256"])
            elif outcome["event_type"] == "artifact_settlement":
                artifact_seen = True
        if not artifact_seen:
            pending.add(lease["lease_sha256"])

    complete = len(accepted) == LEASE_COUNT and not pending
    payload = {
        "schema_version": 1,
        "fleet_authority_sha256": _identity(FLEET_SCHEMA, _fleet_payload()),
        "environment_authority_sha256": _label_sha("synthetic-environment-authority"),
        "expected_lease_count": LEASE_COUNT,
        "lease_sha256s": [item["lease_sha256"] for item in leases],
        "claim_sha256s": sorted(item["claim_sha256"] for item in claims),
        "outcome_sha256s": sorted(item["outcome_sha256"] for item in outcomes),
        "accepted_artifact_count": len(accepted),
        "infrastructure_failure_count": len(failures),
        "pending_lease_count": len(pending),
        "complete": complete,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "orion_comparison_permitted": False,
    }
    return {**payload, "settlement_ledger_sha256": _identity(LEDGER_SCHEMA, payload)}


def verify_synthetic_store(root: str | Path) -> dict[str, Any]:
    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("synthetic evidence root may not be a symlink")
    root_path = supplied.resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"synthetic evidence root is not a directory: {root_path}")

    leases = _expected_leases()
    claims = _expected_claims(leases)
    outcomes = _expected_outcomes(leases, claims)
    expected_files, expected_directories = _managed_topology(leases, claims)
    files, directories = _walk_topology(root_path)
    if files != expected_files:
        raise ValueError(
            f"synthetic evidence file topology differs: "
            f"missing={sorted(expected_files - files)}, extra={sorted(files - expected_files)}"
        )
    if directories != expected_directories:
        raise ValueError(
            f"synthetic evidence directory topology differs: "
            f"missing={sorted(expected_directories - directories)}, "
            f"extra={sorted(directories - expected_directories)}"
        )

    for lease in leases:
        _require_exact(
            root_path / "leases" / f"{lease['lease_sha256']}.json",
            lease,
            "lease_sha256",
            LEASE_SCHEMA,
        )

    claims_by_slot = {}
    for claim in claims:
        claims_by_slot[(claim["lease_sha256"], claim["attempt_number"])] = claim
        _require_exact(
            root_path
            / "claims"
            / claim["lease_sha256"]
            / f"attempt-{claim['attempt_number']:04d}.json",
            claim,
            "claim_sha256",
            CLAIM_SCHEMA,
        )
        invocation_path = (
            root_path
            / "attempts"
            / claim["lease_sha256"]
            / (
                f"attempt-{claim['attempt_number']:04d}-"
                f"{claim['claim_sha256'][:16]}"
            )
            / "provider-invocation.json"
        )
        if _load_json(invocation_path) != _expected_invocation(claim):
            raise ValueError(f"provider invocation differs from claim: {invocation_path}")

    outcomes_by_slot = {
        (item["lease_sha256"], item["attempt_number"]): item for item in outcomes
    }
    for slot, claim in claims_by_slot.items():
        outcome = outcomes_by_slot.get(slot)
        if outcome is None:
            raise ValueError(f"missing terminal outcome for claim slot {slot}")
        schema = (
            FAILURE_SCHEMA
            if outcome["event_type"] == "infrastructure_failure"
            else SETTLEMENT_SCHEMA
        )
        _require_exact(
            root_path
            / "outcomes"
            / outcome["lease_sha256"]
            / f"attempt-{outcome['attempt_number']:04d}.json",
            outcome,
            "outcome_sha256",
            schema,
        )
        if outcome["claim_sha256"] != claim["claim_sha256"]:
            raise ValueError("outcome does not bind exact claim identity")

    ledger = _reconstruct_ledger(leases, claims, outcomes)
    if not ledger["complete"]:
        raise ValueError("persisted synthetic evidence does not earn complete settlement")

    receipt_payload = {
        "schema_version": 1,
        "artifact_kind": "provider_free_gpu_fleet_transport_qualification",
        "fleet_authority_sha256": _identity(FLEET_SCHEMA, _fleet_payload()),
        "environment_authority_sha256": _label_sha("synthetic-environment-authority"),
        "settlement_ledger_sha256": ledger["settlement_ledger_sha256"],
        "lease_count": LEASE_COUNT,
        "claim_count": len(claims),
        "accepted_artifact_count": ledger["accepted_artifact_count"],
        "infrastructure_failure_count": ledger["infrastructure_failure_count"],
        "pending_lease_count": ledger["pending_lease_count"],
        "providers": sorted({item["provider"] for item in claims}),
        "complete": True,
        "scientific_execution_performed": False,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    expected_receipt = {
        **receipt_payload,
        "synthetic_qualification_sha256": _identity(
            SYNTHETIC_RECEIPT_SCHEMA, receipt_payload
        ),
    }
    receipt_path = root_path / "synthetic-qualification.json"
    observed_receipt = _load_json(receipt_path)
    if observed_receipt != expected_receipt:
        raise ValueError(
            "synthetic qualification receipt differs from independently replayed store"
        )
    payload = dict(observed_receipt)
    declared = payload.pop("synthetic_qualification_sha256")
    if _identity(SYNTHETIC_RECEIPT_SCHEMA, payload) != declared:
        raise ValueError("synthetic qualification receipt SHA does not match contents")

    return {
        "verified": True,
        "synthetic_qualification_sha256": declared,
        "fleet_authority_sha256": receipt_payload["fleet_authority_sha256"],
        "settlement_ledger_sha256": ledger["settlement_ledger_sha256"],
        "lease_count": LEASE_COUNT,
        "claim_count": len(claims),
        "accepted_artifact_count": len(
            [item for item in outcomes if item["event_type"] == "artifact_settlement"]
        ),
        "infrastructure_failure_count": len(
            [item for item in outcomes if item["event_type"] == "infrastructure_failure"]
        ),
        "complete": True,
        "scientific_execution_performed": False,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "orion_comparison_permitted": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify provider-free Kumar2024 GPU fleet evidence"
    )
    parser.add_argument("--root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(verify_synthetic_store(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
