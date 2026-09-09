"""Provider-free end-to-end qualification for Kumar2024 GPU fleet control semantics.

No neural data, model execution, predictions, scores, or external provider API is
used. The script exercises the fleet authority as a deterministic systems state
machine with synthetic identities only.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

SYNTHETIC_RECEIPT_SCHEMA = "neuros.nsq_kumar2024_gpu_fleet_synthetic_qualification.v1"
SYNTHETIC_PROVIDER_A = "synthetic-provider-a"
SYNTHETIC_PROVIDER_B = "synthetic-provider-b"
SYNTHETIC_LEASE_COUNT = 4


def _load_fleet():
    module_path = (
        Path(__file__).resolve().parents[2]
        / "packages"
        / "neuros"
        / "src"
        / "neuros"
        / "evidence"
        / "kumar2024_gpu_fleet.py"
    )
    spec = importlib.util.spec_from_file_location(
        "neuros_kumar2024_gpu_fleet_synthetic_contract",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load Kumar2024 GPU fleet authority")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


fleet = _load_fleet()


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _identity(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        {"schema": SYNTHETIC_RECEIPT_SCHEMA, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_once(path: Path, payload: dict[str, Any]) -> None:
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


def _synthetic_roster() -> list[dict[str, Any]]:
    return [
        {
            "subject": subject,
            "target_session": "5",
            "split_seed": 2026,
            "method_id": fleet.EEGNET_METHOD_ID,
            "model_seed": 31415 + subject - 1,
            "budgets_per_class": list(fleet.CALIBRATION_FRONTIER),
            "shard_spec_sha256": _sha(f"synthetic-shard-{subject}"),
        }
        for subject in range(1, SYNTHETIC_LEASE_COUNT + 1)
    ]


def _accepted(claim, *, ordinal: int):
    return fleet.ArtifactSettlementEvent(
        claim_sha256=claim.sha256,
        lease_sha256=claim.lease_sha256,
        shard_spec_sha256=claim.shard_spec_sha256,
        attempt_number=claim.attempt_number,
        gpu_worker_receipt_sha256=_sha(
            f"synthetic-worker-receipt-{ordinal}-{claim.attempt_number}"
        ),
        worker_bundle_sha256=_sha(
            f"synthetic-worker-bundle-{ordinal}-{claim.attempt_number}"
        ),
        learned_state_sha256=_sha(
            f"synthetic-learned-state-{ordinal}-{claim.attempt_number}"
        ),
        environment_authority_sha256=_sha("synthetic-environment-authority"),
        provider_receipt_sha256=_sha(
            f"synthetic-provider-receipt-{ordinal}-{claim.attempt_number}"
        ),
        accelerator_name="Tesla T4 (synthetic)",
    )


def _claim(lease, *, attempt: int, provider: str):
    return fleet.ClaimEvent(
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=attempt,
        worker_id=f"synthetic-worker-{lease.ordinal}-{attempt}",
        provider=provider,
        provider_run_id=f"synthetic-run-{lease.ordinal}-{attempt}",
    )


def run_synthetic_qualification(output: str | Path) -> dict[str, Any]:
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)

    preflight = fleet.PreflightAdmission(
        gpu_execution_authority_revision=fleet.PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION,
        gpu_binding_sha256=_sha("synthetic-gpu-binding"),
        execution_plan_sha256=_sha("synthetic-execution-plan"),
        environment_authority_sha256=_sha("synthetic-environment-authority"),
        gpu_worker_receipt_sha256=_sha("synthetic-preflight-worker-receipt"),
        accelerator_name="Tesla T4 (synthetic)",
        elapsed_seconds=1.0,
    )
    authority = fleet.FleetAuthority.from_preflight(
        preflight,
        expected_shard_count=SYNTHETIC_LEASE_COUNT,
        max_attempts_per_lease=3,
    )
    leases = fleet.build_leases(authority, reversed(_synthetic_roster()))

    for lease in leases:
        fleet.persist_lease(root, lease)

    claims = []
    outcomes = []
    providers = set()

    for lease in leases:
        first_provider = (
            SYNTHETIC_PROVIDER_A if lease.ordinal % 2 == 0 else SYNTHETIC_PROVIDER_B
        )
        first = _claim(lease, attempt=1, provider=first_provider)
        providers.add(first.provider)
        fleet.persist_claim(root, lease, first)
        attempt_root = fleet.prepare_attempt_namespace(root, first)
        _write_once(
            attempt_root / "provider-invocation.json",
            {
                "schema_version": 1,
                "provider": first.provider,
                "provider_run_id": first.provider_run_id,
                "scientific_execution_performed": False,
            },
        )
        claims.append(first)

        if lease.ordinal == 1:
            failure = fleet.InfrastructureFailureEvent(
                claim_sha256=first.sha256,
                lease_sha256=lease.sha256,
                shard_spec_sha256=lease.shard_spec_sha256,
                attempt_number=1,
                failure_kind="provider_preemption",
                failure_evidence_sha256=_sha("synthetic-provider-preemption"),
            )
            fleet.persist_outcome(root, failure)
            outcomes.append(failure)

            second = _claim(lease, attempt=2, provider=SYNTHETIC_PROVIDER_A)
            providers.add(second.provider)
            fleet.persist_claim(root, lease, second)
            second_root = fleet.prepare_attempt_namespace(root, second)
            _write_once(
                second_root / "provider-invocation.json",
                {
                    "schema_version": 1,
                    "provider": second.provider,
                    "provider_run_id": second.provider_run_id,
                    "scientific_execution_performed": False,
                },
            )
            claims.append(second)
            settlement = _accepted(second, ordinal=lease.ordinal)
        else:
            settlement = _accepted(first, ordinal=lease.ordinal)

        fleet.persist_outcome(root, settlement)
        outcomes.append(settlement)

    ledger = fleet.reconstruct_settlement(authority, leases, claims, outcomes)
    if not ledger["complete"]:
        raise RuntimeError("synthetic fleet did not reach complete deterministic settlement")

    payload = {
        "schema_version": 1,
        "artifact_kind": "provider_free_gpu_fleet_transport_qualification",
        "fleet_authority_sha256": authority.sha256,
        "environment_authority_sha256": authority.environment_authority_sha256,
        "settlement_ledger_sha256": ledger["settlement_ledger_sha256"],
        "lease_count": len(leases),
        "claim_count": len(claims),
        "accepted_artifact_count": ledger["accepted_artifact_count"],
        "infrastructure_failure_count": ledger["infrastructure_failure_count"],
        "pending_lease_count": ledger["pending_lease_count"],
        "providers": sorted(providers),
        "complete": True,
        "scientific_execution_performed": False,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "claim_boundary": (
            "synthetic systems qualification of lease/claim/retry/artifact settlement only; "
            "no neural data, model execution, prediction, score, efficacy, or ORION evidence"
        ),
    }
    receipt = {
        **payload,
        "synthetic_qualification_sha256": _identity(payload),
    }
    _write_once(root / "synthetic-qualification.json", receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run provider-free Kumar2024 GPU fleet systems qualification"
    )
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    print(
        json.dumps(
            run_synthetic_qualification(args.output),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
