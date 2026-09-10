#!/usr/bin/env python3
"""Independent verifier for the promoted three-lease Kumar2024 synthetic fleet packet.

This verifier is intentionally standard-library only and imports neither neurOS nor
the synthetic generator. It independently derives the fixed synthetic protocol,
checks the complete on-disk packet topology, recomputes every domain-separated
identity, replays retry legality, and reconstructs the final receipt.
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
QUALIFICATION_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_synthetic_qualification.v1"

PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION = "07a6c5fa5f212d54aae408237f14bb56f2f6eee9"
EEGNET_METHOD_ID = "braindecode-eegnet"
CALIBRATION_FRONTIER = [0, 1, 2, 5, 10]
MAX_ATTEMPTS = 2
LEASE_COUNT = 3
PROVIDER = "synthetic"
ACCELERATOR = "Synthetic Tesla T4 systems probe"
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
    "provider-free control-plane systems qualification only; synthetic identities "
    "are not scientific or provider execution evidence"
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


def _digest(label: str) -> str:
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


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"symlink evidence file is forbidden: {path}")
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


def _preflight_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "gpu_execution_authority_revision": PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION,
        "gpu_binding_sha256": _digest("synthetic-gpu-binding"),
        "execution_plan_sha256": _digest("synthetic-execution-plan"),
        "environment_authority_sha256": _digest("synthetic-environment"),
        "gpu_worker_receipt_sha256": _digest("synthetic-preflight-worker"),
        "accelerator_name": ACCELERATOR,
        "elapsed_seconds": 1.0,
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "admission_inputs": list(PROVIDER_SELECTION_BASIS),
    }


def _fleet_payload() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "artifact_kind": "score_blind_gpu_fleet_execution_authority",
        "preflight_admission_sha256": _identity(PREFLIGHT_SCHEMA, _preflight_payload()),
        "gpu_execution_authority_revision": PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION,
        "gpu_binding_sha256": _digest("synthetic-gpu-binding"),
        "execution_plan_sha256": _digest("synthetic-execution-plan"),
        "environment_authority_sha256": _digest("synthetic-environment"),
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


def _shard_rows() -> list[tuple[int, str, int, int, str]]:
    return [
        (1, "5", 2026, 31415, "synthetic-shard-a"),
        (2, "4", 3407, 384165836, "synthetic-shard-b"),
        (3, "3", 9109, 3991196546, "synthetic-shard-c"),
    ]


def _expected_leases() -> list[dict[str, Any]]:
    fleet_sha = _identity(FLEET_SCHEMA, _fleet_payload())
    rows = sorted(
        (
            subject,
            session,
            split_seed,
            model_seed,
            _digest(label),
        )
        for subject, session, split_seed, model_seed, label in _shard_rows()
    )
    result = []
    for ordinal, (subject, session, split_seed, model_seed, shard_sha) in enumerate(rows):
        payload = {
            "schema_version": 1,
            "fleet_authority_sha256": fleet_sha,
            "gpu_binding_sha256": _digest("synthetic-gpu-binding"),
            "execution_plan_sha256": _digest("synthetic-execution-plan"),
            "environment_authority_sha256": _digest("synthetic-environment"),
            "shard_spec_sha256": shard_sha,
            "ordinal": ordinal,
            "subject": subject,
            "target_session": session,
            "split_seed": split_seed,
            "method_id": EEGNET_METHOD_ID,
            "model_seed": model_seed,
            "budgets_per_class": list(CALIBRATION_FRONTIER),
            "max_attempts": MAX_ATTEMPTS,
        }
        result.append({**payload, "lease_sha256": _identity(LEASE_SCHEMA, payload)})
    return result


def _claim(
    lease: Mapping[str, Any],
    attempt: int,
    worker_id: str,
    provider_run_id: str,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "event_type": "claim",
        "lease_sha256": lease["lease_sha256"],
        "shard_spec_sha256": lease["shard_spec_sha256"],
        "attempt_number": attempt,
        "worker_id": worker_id,
        "provider": PROVIDER,
        "provider_run_id": provider_run_id,
    }
    return {**payload, "claim_sha256": _identity(CLAIM_SCHEMA, payload)}


def _expected_claims(leases: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        _claim(leases[0], 1, "synthetic-worker-0", "synthetic-run-0001"),
        _claim(leases[1], 1, "synthetic-worker-1a", "synthetic-run-0002"),
        _claim(leases[1], 2, "synthetic-worker-1b", "synthetic-run-0003"),
        _claim(leases[2], 1, "synthetic-worker-2", "synthetic-run-0004"),
    ]


def _failure(claim: Mapping[str, Any]) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "event_type": "infrastructure_failure",
        "claim_sha256": claim["claim_sha256"],
        "lease_sha256": claim["lease_sha256"],
        "shard_spec_sha256": claim["shard_spec_sha256"],
        "attempt_number": 1,
        "failure_kind": "provider_preemption",
        "failure_evidence_sha256": _digest("synthetic-preemption-evidence"),
        "valid_worker_artifact_exists": False,
    }
    return {**payload, "outcome_sha256": _identity(FAILURE_SCHEMA, payload)}


def _settlement(claim: Mapping[str, Any], label_index: int) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "event_type": "artifact_settlement",
        "claim_sha256": claim["claim_sha256"],
        "lease_sha256": claim["lease_sha256"],
        "shard_spec_sha256": claim["shard_spec_sha256"],
        "attempt_number": claim["attempt_number"],
        "gpu_worker_receipt_sha256": _digest(f"synthetic-worker-receipt-{label_index}"),
        "worker_bundle_sha256": _digest(f"synthetic-worker-bundle-{label_index}"),
        "learned_state_sha256": _digest(f"synthetic-learned-state-{label_index}"),
        "environment_authority_sha256": _digest("synthetic-environment"),
        "provider_receipt_sha256": _digest(f"synthetic-provider-receipt-{label_index}"),
        "accelerator_name": ACCELERATOR,
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
    }
    return {**payload, "outcome_sha256": _identity(SETTLEMENT_SCHEMA, payload)}


def _expected_outcomes(
    leases: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    del leases
    return [
        _settlement(claims[0], 0),
        _failure(claims[1]),
        _settlement(claims[2], 1),
        _settlement(claims[3], 2),
    ]


def _expected_topology(
    leases: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
) -> tuple[set[str], set[str]]:
    files = {"synthetic_qualification_receipt.json"}
    directories = {"store", "store/leases", "store/claims", "store/outcomes", "store/attempts"}
    for lease in leases:
        lease_sha = lease["lease_sha256"]
        files.add(f"store/leases/{lease_sha}.json")
        for prefix in ("claims", "outcomes", "attempts"):
            directories.add(f"store/{prefix}/{lease_sha}")
    for claim in claims:
        lease_sha = claim["lease_sha256"]
        attempt = claim["attempt_number"]
        files.add(f"store/claims/{lease_sha}/attempt-{attempt:04d}.json")
        files.add(f"store/outcomes/{lease_sha}/attempt-{attempt:04d}.json")
        directories.add(
            f"store/attempts/{lease_sha}/"
            f"attempt-{attempt:04d}-{claim['claim_sha256'][:16]}"
        )
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


def _require_exact_record(
    path: Path,
    expected: Mapping[str, Any],
    identity_field: str,
    schema: str,
) -> dict[str, Any]:
    observed = _load_json(path)
    if observed != expected:
        raise ValueError(f"evidence differs from frozen synthetic protocol: {path}")
    declared = observed.get(identity_field)
    payload = dict(observed)
    payload.pop(identity_field, None)
    if declared != _identity(schema, payload):
        raise ValueError(f"{identity_field} does not match record contents: {path}")
    return observed


def _replay_ledger(
    leases: Sequence[dict[str, Any]],
    claims: Sequence[dict[str, Any]],
    outcomes: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    claim_by_slot: dict[tuple[str, int], dict[str, Any]] = {}
    provider_runs: set[tuple[str, str]] = set()
    for claim in claims:
        slot = (claim["lease_sha256"], claim["attempt_number"])
        provider_slot = (claim["provider"], claim["provider_run_id"])
        if slot in claim_by_slot or provider_slot in provider_runs:
            raise ValueError("duplicate claim slot or provider run")
        claim_by_slot[slot] = claim
        provider_runs.add(provider_slot)

    outcome_by_claim: dict[str, dict[str, Any]] = {}
    for outcome in outcomes:
        if outcome["claim_sha256"] in outcome_by_claim:
            raise ValueError("claim has multiple terminal outcomes")
        outcome_by_claim[outcome["claim_sha256"]] = outcome

    accepted = []
    failures = []
    pending: set[str] = set()
    for lease in sorted(leases, key=lambda item: item["ordinal"]):
        lease_claims = sorted(
            (
                claim
                for (lease_sha, _), claim in claim_by_slot.items()
                if lease_sha == lease["lease_sha256"]
            ),
            key=lambda item: item["attempt_number"],
        )
        attempts = [claim["attempt_number"] for claim in lease_claims]
        if attempts != list(range(1, len(attempts) + 1)):
            raise ValueError("lease attempts are not contiguous from 1")
        if not lease_claims:
            pending.add(lease["lease_sha256"])
            continue

        artifact_seen = False
        for index, claim in enumerate(lease_claims):
            if claim["shard_spec_sha256"] != lease["shard_spec_sha256"]:
                raise ValueError("claim shard does not match lease")
            if artifact_seen:
                raise ValueError("attempt exists after accepted artifact")
            if index:
                prior = outcome_by_claim.get(lease_claims[index - 1]["claim_sha256"])
                if prior is None or prior.get("event_type") != "infrastructure_failure":
                    raise ValueError("retry lacks trusted infrastructure failure")
                if prior.get("valid_worker_artifact_exists") is not False:
                    raise ValueError("retry follows a valid worker artifact")
            outcome = outcome_by_claim.get(claim["claim_sha256"])
            if outcome is None:
                if index != len(lease_claims) - 1:
                    raise ValueError("only the latest claim may be unsettled")
                pending.add(lease["lease_sha256"])
                continue
            if (
                outcome["lease_sha256"] != claim["lease_sha256"]
                or outcome["shard_spec_sha256"] != claim["shard_spec_sha256"]
                or outcome["attempt_number"] != claim["attempt_number"]
            ):
                raise ValueError("outcome binding differs from claim")
            if outcome["event_type"] == "infrastructure_failure":
                failures.append(outcome)
            elif outcome["event_type"] == "artifact_settlement":
                if outcome["environment_authority_sha256"] != lease["environment_authority_sha256"]:
                    raise ValueError("artifact environment authority differs from lease")
                accepted.append(outcome)
                artifact_seen = True
            else:
                raise ValueError("unknown terminal outcome type")

        if not artifact_seen and lease_claims[-1]["claim_sha256"] in outcome_by_claim:
            pending.add(lease["lease_sha256"])

    complete = len(accepted) == LEASE_COUNT and not pending
    payload = {
        "schema_version": 1,
        "fleet_authority_sha256": _identity(FLEET_SCHEMA, _fleet_payload()),
        "environment_authority_sha256": _digest("synthetic-environment"),
        "expected_lease_count": LEASE_COUNT,
        "lease_sha256s": [
            item["lease_sha256"] for item in sorted(leases, key=lambda item: item["ordinal"])
        ],
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


def verify_synthetic_packet(root: str | Path) -> dict[str, Any]:
    supplied = Path(root)
    if supplied.is_symlink():
        raise ValueError("synthetic evidence root may not be a symlink")
    root_path = supplied.resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(f"synthetic evidence root is not a directory: {root_path}")

    leases = _expected_leases()
    claims = _expected_claims(leases)
    outcomes = _expected_outcomes(leases, claims)

    expected_files, expected_dirs = _expected_topology(leases, claims)
    files, dirs = _walk_topology(root_path)
    if files != expected_files:
        raise ValueError(
            "synthetic evidence file topology differs: "
            f"missing={sorted(expected_files - files)}, extra={sorted(files - expected_files)}"
        )
    if dirs != expected_dirs:
        raise ValueError(
            "synthetic evidence directory topology differs: "
            f"missing={sorted(expected_dirs - dirs)}, extra={sorted(dirs - expected_dirs)}"
        )

    for lease in leases:
        _require_exact_record(
            root_path / "store" / "leases" / f"{lease['lease_sha256']}.json",
            lease,
            "lease_sha256",
            LEASE_SCHEMA,
        )

    for claim in claims:
        _require_exact_record(
            root_path
            / "store"
            / "claims"
            / claim["lease_sha256"]
            / f"attempt-{claim['attempt_number']:04d}.json",
            claim,
            "claim_sha256",
            CLAIM_SCHEMA,
        )

    outcomes_by_slot = {
        (item["lease_sha256"], item["attempt_number"]): item for item in outcomes
    }
    for claim in claims:
        expected = outcomes_by_slot[(claim["lease_sha256"], claim["attempt_number"])]
        schema = (
            FAILURE_SCHEMA
            if expected["event_type"] == "infrastructure_failure"
            else SETTLEMENT_SCHEMA
        )
        _require_exact_record(
            root_path
            / "store"
            / "outcomes"
            / claim["lease_sha256"]
            / f"attempt-{claim['attempt_number']:04d}.json",
            expected,
            "outcome_sha256",
            schema,
        )

    ledger = _replay_ledger(leases, claims, outcomes)
    if ledger["complete"] is not True:
        raise ValueError("persisted synthetic packet does not earn complete settlement")

    receipt_payload = {
        "schema_version": 1,
        "artifact_kind": "provider_free_gpu_fleet_systems_qualification",
        "promoted_gpu_execution_authority_revision": PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION,
        "preflight_admission_sha256": _identity(PREFLIGHT_SCHEMA, _preflight_payload()),
        "fleet_authority_sha256": _identity(FLEET_SCHEMA, _fleet_payload()),
        "environment_authority_sha256": _digest("synthetic-environment"),
        "lease_sha256s": [lease["lease_sha256"] for lease in leases],
        "settlement_ledger_sha256": ledger["settlement_ledger_sha256"],
        "expected_lease_count": 3,
        "accepted_artifact_count": 3,
        "infrastructure_failure_count": 1,
        "attempt_count": 4,
        "provider": PROVIDER,
        "model_execution_performed": False,
        "neural_data_accessed": False,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "claim_boundary": CLAIM_BOUNDARY,
    }
    expected_receipt = {
        **receipt_payload,
        "synthetic_qualification_sha256": _identity(QUALIFICATION_SCHEMA, receipt_payload),
    }
    receipt_path = root_path / "synthetic_qualification_receipt.json"
    observed_receipt = _load_json(receipt_path)
    if observed_receipt != expected_receipt:
        raise ValueError("synthetic qualification receipt differs from independent replay")
    payload = dict(observed_receipt)
    declared = payload.pop("synthetic_qualification_sha256")
    if declared != _identity(QUALIFICATION_SCHEMA, payload):
        raise ValueError("synthetic qualification receipt identity mismatch")

    return {
        "verified": True,
        "synthetic_qualification_sha256": declared,
        "fleet_authority_sha256": receipt_payload["fleet_authority_sha256"],
        "environment_authority_sha256": receipt_payload["environment_authority_sha256"],
        "settlement_ledger_sha256": ledger["settlement_ledger_sha256"],
        "lease_count": 3,
        "claim_count": 4,
        "accepted_artifact_count": 3,
        "infrastructure_failure_count": 1,
        "complete": True,
        "model_execution_performed": False,
        "neural_data_accessed": False,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "orion_comparison_permitted": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify the promoted Kumar2024 synthetic fleet packet"
    )
    parser.add_argument("--root", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(json.dumps(verify_synthetic_packet(args.root), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
