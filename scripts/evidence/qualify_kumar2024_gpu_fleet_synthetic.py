#!/usr/bin/env python3
"""Provider-free systems qualification for the Kumar2024 GPU fleet control plane.

This script executes no neural model and reads no scientific result. It exercises
only lease, claim, retry, write-once namespace, artifact-settlement, environment
binding, and deterministic ledger semantics using synthetic identities.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping, Sequence

QUALIFICATION_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_synthetic_qualification.v1"


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


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _load_fleet(repo_root: Path) -> ModuleType:
    module_path = (
        repo_root
        / "packages"
        / "neuros"
        / "src"
        / "neuros"
        / "evidence"
        / "kumar2024_gpu_fleet.py"
    )
    if not module_path.is_file():
        raise FileNotFoundError(f"fleet authority module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(
        "neuros_kumar2024_gpu_fleet_synthetic_target",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("unable to load fleet authority module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _synthetic_shards() -> list[dict[str, Any]]:
    seeds = (
        (1, "5", 2026, 31415, "synthetic-shard-a"),
        (2, "4", 3407, 384165836, "synthetic-shard-b"),
        (3, "3", 9109, 3991196546, "synthetic-shard-c"),
    )
    return [
        {
            "subject": subject,
            "target_session": session,
            "split_seed": split_seed,
            "method_id": "braindecode-eegnet",
            "model_seed": model_seed,
            "budgets_per_class": [0, 1, 2, 5, 10],
            "shard_spec_sha256": _digest(label),
        }
        for subject, session, split_seed, model_seed, label in seeds
    ]


def _write_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    path.write_text(encoded, encoding="utf-8")


def run(repo_root: Path, output: Path) -> dict[str, Any]:
    """Run a deterministic provider-free fleet systems qualification."""
    if output.exists():
        raise FileExistsError(f"refusing to reuse qualification output: {output}")
    output.mkdir(parents=True)
    store = output / "store"

    fleet = _load_fleet(repo_root)
    preflight = fleet.PreflightAdmission(
        gpu_execution_authority_revision=fleet.PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION,
        gpu_binding_sha256=_digest("synthetic-gpu-binding"),
        execution_plan_sha256=_digest("synthetic-execution-plan"),
        environment_authority_sha256=_digest("synthetic-environment"),
        gpu_worker_receipt_sha256=_digest("synthetic-preflight-worker"),
        accelerator_name="Synthetic Tesla T4 systems probe",
        elapsed_seconds=1.0,
    )
    authority = fleet.FleetAuthority.from_preflight(
        preflight,
        expected_shard_count=3,
        max_attempts_per_lease=2,
    )
    leases = fleet.build_leases(authority, _synthetic_shards())
    if len(leases) != 3:
        raise RuntimeError("synthetic qualification expected exactly three leases")

    for lease in leases:
        fleet.persist_lease(store, lease)

    claims = []
    outcomes = []

    lease = leases[0]
    claim = fleet.ClaimEvent(
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=1,
        worker_id="synthetic-worker-0",
        provider="synthetic",
        provider_run_id="synthetic-run-0001",
    )
    fleet.persist_claim(store, lease, claim)
    fleet.prepare_attempt_namespace(store, claim)
    outcome = fleet.ArtifactSettlementEvent(
        claim_sha256=claim.sha256,
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=1,
        gpu_worker_receipt_sha256=_digest("synthetic-worker-receipt-0"),
        worker_bundle_sha256=_digest("synthetic-worker-bundle-0"),
        learned_state_sha256=_digest("synthetic-learned-state-0"),
        environment_authority_sha256=authority.environment_authority_sha256,
        provider_receipt_sha256=_digest("synthetic-provider-receipt-0"),
        accelerator_name="Synthetic Tesla T4 systems probe",
    )
    fleet.persist_outcome(store, outcome)
    claims.append(claim)
    outcomes.append(outcome)

    lease = leases[1]
    first = fleet.ClaimEvent(
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=1,
        worker_id="synthetic-worker-1a",
        provider="synthetic",
        provider_run_id="synthetic-run-0002",
    )
    fleet.persist_claim(store, lease, first)
    fleet.prepare_attempt_namespace(store, first)
    failure = fleet.InfrastructureFailureEvent(
        claim_sha256=first.sha256,
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=1,
        failure_kind="provider_preemption",
        failure_evidence_sha256=_digest("synthetic-preemption-evidence"),
    )
    fleet.persist_outcome(store, failure)

    second = fleet.ClaimEvent(
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=2,
        worker_id="synthetic-worker-1b",
        provider="synthetic",
        provider_run_id="synthetic-run-0003",
    )
    fleet.persist_claim(store, lease, second)
    fleet.prepare_attempt_namespace(store, second)
    recovered = fleet.ArtifactSettlementEvent(
        claim_sha256=second.sha256,
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=2,
        gpu_worker_receipt_sha256=_digest("synthetic-worker-receipt-1"),
        worker_bundle_sha256=_digest("synthetic-worker-bundle-1"),
        learned_state_sha256=_digest("synthetic-learned-state-1"),
        environment_authority_sha256=authority.environment_authority_sha256,
        provider_receipt_sha256=_digest("synthetic-provider-receipt-1"),
        accelerator_name="Synthetic Tesla T4 systems probe",
    )
    fleet.persist_outcome(store, recovered)
    claims.extend((first, second))
    outcomes.extend((failure, recovered))

    lease = leases[2]
    claim = fleet.ClaimEvent(
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=1,
        worker_id="synthetic-worker-2",
        provider="synthetic",
        provider_run_id="synthetic-run-0004",
    )
    fleet.persist_claim(store, lease, claim)
    fleet.prepare_attempt_namespace(store, claim)
    outcome = fleet.ArtifactSettlementEvent(
        claim_sha256=claim.sha256,
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=1,
        gpu_worker_receipt_sha256=_digest("synthetic-worker-receipt-2"),
        worker_bundle_sha256=_digest("synthetic-worker-bundle-2"),
        learned_state_sha256=_digest("synthetic-learned-state-2"),
        environment_authority_sha256=authority.environment_authority_sha256,
        provider_receipt_sha256=_digest("synthetic-provider-receipt-2"),
        accelerator_name="Synthetic Tesla T4 systems probe",
    )
    fleet.persist_outcome(store, outcome)
    claims.append(claim)
    outcomes.append(outcome)

    ledger = fleet.reconstruct_settlement(authority, leases, claims, outcomes)
    if ledger["complete"] is not True:
        raise RuntimeError("synthetic fleet did not settle completely")
    if ledger["accepted_artifact_count"] != 3:
        raise RuntimeError("synthetic fleet accepted-artifact count drifted")
    if ledger["infrastructure_failure_count"] != 1:
        raise RuntimeError("synthetic retry path was not exercised exactly once")
    if ledger["pending_lease_count"] != 0:
        raise RuntimeError("synthetic fleet retained a pending lease")
    if ledger["scientific_outcomes_inspected"] is not False:
        raise RuntimeError("synthetic fleet crossed the scientific-outcome boundary")
    if ledger["numerical_result_interpretable"] is not False:
        raise RuntimeError("synthetic fleet promoted numerical interpretation")
    if ledger["orion_comparison_permitted"] is not False:
        raise RuntimeError("synthetic fleet incorrectly permitted ORION comparison")

    receipt_payload = {
        "schema_version": 1,
        "artifact_kind": "provider_free_gpu_fleet_systems_qualification",
        "promoted_gpu_execution_authority_revision": (
            fleet.PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION
        ),
        "preflight_admission_sha256": preflight.sha256,
        "fleet_authority_sha256": authority.sha256,
        "environment_authority_sha256": authority.environment_authority_sha256,
        "lease_sha256s": [lease.sha256 for lease in leases],
        "settlement_ledger_sha256": ledger["settlement_ledger_sha256"],
        "expected_lease_count": 3,
        "accepted_artifact_count": 3,
        "infrastructure_failure_count": 1,
        "attempt_count": 4,
        "provider": "synthetic",
        "model_execution_performed": False,
        "neural_data_accessed": False,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "claim_boundary": (
            "provider-free control-plane systems qualification only; synthetic "
            "identities are not scientific or provider execution evidence"
        ),
    }
    fleet.reject_scientific_fields(receipt_payload)
    qualification_sha256 = _identity(QUALIFICATION_SCHEMA, receipt_payload)
    receipt = {
        **receipt_payload,
        "synthetic_qualification_sha256": qualification_sha256,
    }
    _write_receipt(output / "synthetic_qualification_receipt.json", receipt)
    return receipt


def verify(output: Path) -> dict[str, Any]:
    """Verify the deterministic outer receipt without rerunning the simulation."""
    path = output / "synthetic_qualification_receipt.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("synthetic qualification receipt must be a JSON object")
    declared = raw.get("synthetic_qualification_sha256")
    if not isinstance(declared, str):
        raise ValueError("synthetic qualification receipt lacks its identity")
    payload = dict(raw)
    payload.pop("synthetic_qualification_sha256", None)
    expected = _identity(QUALIFICATION_SCHEMA, payload)
    if declared != expected:
        raise ValueError("synthetic qualification receipt identity mismatch")
    for field in (
        "model_execution_performed",
        "neural_data_accessed",
        "scientific_outcomes_inspected",
        "numerical_result_interpretable",
        "external_floor_claim_generated",
        "orion_comparison_permitted",
    ):
        if payload.get(field) is not False:
            raise ValueError(f"synthetic qualification requires {field}=false")
    if payload.get("accepted_artifact_count") != payload.get("expected_lease_count"):
        raise ValueError("synthetic qualification does not show complete settlement")
    return {
        "verified": True,
        "synthetic_qualification_sha256": expected,
        "settlement_ledger_sha256": payload["settlement_ledger_sha256"],
        "model_execution_performed": False,
        "scientific_outcomes_inspected": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run")
    run_parser.add_argument("--repo-root", default=".")
    run_parser.add_argument("--output", required=True)
    verify_parser = sub.add_parser("verify")
    verify_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run(Path(args.repo_root).resolve(), Path(args.output).resolve())
    else:
        result = verify(Path(args.output).resolve())
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
