from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

MODULE_PATH = (
    Path(__file__).parents[1]
    / "packages"
    / "neuros"
    / "src"
    / "neuros"
    / "evidence"
    / "kumar2024_gpu_fleet.py"
)
SPEC = importlib.util.spec_from_file_location("neuros_kumar2024_gpu_fleet_contract", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
fleet = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fleet
SPEC.loader.exec_module(fleet)


def sha(char: str) -> str:
    return char * 64


def preflight(**overrides):
    values = {
        "gpu_execution_authority_revision": fleet.PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION,
        "gpu_binding_sha256": sha("a"),
        "execution_plan_sha256": sha("b"),
        "environment_authority_sha256": sha("c"),
        "gpu_worker_receipt_sha256": sha("d"),
        "accelerator_name": "Tesla T4",
        "elapsed_seconds": 12.5,
    }
    values.update(overrides)
    return fleet.PreflightAdmission(**values)


def authority(count: int = 2, **overrides):
    values = {
        "expected_shard_count": count,
        "max_attempts_per_lease": 3,
    }
    values.update(overrides)
    return fleet.FleetAuthority.from_preflight(preflight(), **values)


def shard(subject: int, digest: str) -> dict:
    return {
        "subject": subject,
        "target_session": "5",
        "split_seed": 2026,
        "method_id": "braindecode-eegnet",
        "model_seed": 31415,
        "budgets_per_class": [0, 1, 2, 5, 10],
        "shard_spec_sha256": sha(digest),
    }


def leases():
    return fleet.build_leases(authority(), [shard(2, "f"), shard(1, "e")])


def claim(lease, attempt: int, run: str):
    return fleet.ClaimEvent(
        lease_sha256=lease.sha256,
        shard_spec_sha256=lease.shard_spec_sha256,
        attempt_number=attempt,
        worker_id=f"worker-{run}",
        provider="kaggle",
        provider_run_id=run,
    )


def accepted(claim_event, *, learned: str = "3", environment: str = "c"):
    return fleet.ArtifactSettlementEvent(
        claim_sha256=claim_event.sha256,
        lease_sha256=claim_event.lease_sha256,
        shard_spec_sha256=claim_event.shard_spec_sha256,
        attempt_number=claim_event.attempt_number,
        gpu_worker_receipt_sha256=sha("1"),
        worker_bundle_sha256=sha("2"),
        learned_state_sha256=sha(learned),
        environment_authority_sha256=sha(environment),
        provider_receipt_sha256=sha("5"),
        accelerator_name="Tesla T4",
    )


def failed(claim_event):
    return fleet.InfrastructureFailureEvent(
        claim_sha256=claim_event.sha256,
        lease_sha256=claim_event.lease_sha256,
        shard_spec_sha256=claim_event.shard_spec_sha256,
        attempt_number=claim_event.attempt_number,
        failure_kind="provider_preemption",
        failure_evidence_sha256=sha("6"),
    )


def test_preflight_is_systems_only_and_t4_bound():
    item = preflight()
    assert item.numerical_result_interpretable is False
    assert item.orion_comparison_permitted is False
    assert item.sha256 == preflight().sha256
    with pytest.raises(ValueError, match="T4-class"):
        preflight(accelerator_name="NVIDIA A100")
    with pytest.raises(ValueError, match="numerical_result_interpretable=false"):
        preflight(numerical_result_interpretable=True)


def test_fleet_authority_requires_promoted_gpu_revision():
    item = authority()
    assert item.gpu_execution_authority_revision == fleet.PROMOTED_GPU_EXECUTION_AUTHORITY_REVISION
    wrong = preflight(gpu_execution_authority_revision="1" * 40)
    with pytest.raises(ValueError, match="promoted GPU execution authority"):
        fleet.FleetAuthority.from_preflight(wrong, expected_shard_count=2)


def test_leases_are_canonical_and_bind_complete_frozen_frontier():
    auth = authority()
    first = fleet.build_leases(auth, [shard(2, "f"), shard(1, "e")])
    second = fleet.build_leases(auth, [shard(1, "e"), shard(2, "f")])
    assert [item.sha256 for item in first] == [item.sha256 for item in second]
    assert [item.subject for item in first] == [1, 2]
    assert all(item.budgets_per_class == (0, 1, 2, 5, 10) for item in first)
    assert all(item.fleet_authority_sha256 == auth.sha256 for item in first)


def test_scientific_outcome_fields_are_rejected_from_lease_inputs():
    auth = authority()
    contaminated = shard(1, "e")
    contaminated["balanced_accuracy"] = 0.99
    with pytest.raises(ValueError, match="scientific field"):
        fleet.build_leases(auth, [contaminated, shard(2, "f")])


def test_full_settlement_accepts_infrastructure_retry_without_scores():
    auth = authority()
    lease_a, lease_b = leases()
    a1 = claim(lease_a, 1, "run-a1")
    b1 = claim(lease_b, 1, "run-b1")
    b2 = claim(lease_b, 2, "run-b2")
    result = fleet.reconstruct_settlement(
        auth,
        [lease_a, lease_b],
        [a1, b1, b2],
        [accepted(a1), failed(b1), accepted(b2, learned="7")],
    )
    assert result["complete"] is True
    assert result["accepted_artifact_count"] == 2
    assert result["infrastructure_failure_count"] == 1
    assert result["pending_lease_count"] == 0
    assert result["scientific_outcomes_inspected"] is False
    assert result["orion_comparison_permitted"] is False


def test_retry_without_trusted_infrastructure_failure_rejects():
    auth = authority()
    lease_a, lease_b = leases()
    a1 = claim(lease_a, 1, "run-a1")
    a2 = claim(lease_a, 2, "run-a2")
    with pytest.raises(ValueError, match="attempt exists after accepted artifact"):
        fleet.reconstruct_settlement(
            auth,
            [lease_a, lease_b],
            [a1, a2],
            [accepted(a1)],
        )


def test_claim_after_accepted_artifact_rejects_even_if_score_is_unknown():
    auth = authority()
    lease_a, lease_b = leases()
    a1 = claim(lease_a, 1, "run-a1")
    a2 = claim(lease_a, 2, "run-a2")
    with pytest.raises(ValueError, match="attempt exists after accepted artifact"):
        fleet.reconstruct_settlement(
            auth,
            [lease_a, lease_b],
            [a1, a2],
            [accepted(a1), accepted(a2)],
        )


def test_settlement_requires_explicit_learned_state_identity():
    lease_a, _ = leases()
    a1 = claim(lease_a, 1, "run-a1")
    with pytest.raises(ValueError, match="learned_state_sha256"):
        accepted(a1, learned="not-a-digest")


def test_infrastructure_failure_cannot_authorize_retry_if_valid_artifact_exists():
    lease_a, _ = leases()
    a1 = claim(lease_a, 1, "run-a1")
    with pytest.raises(ValueError, match="no valid artifact exists"):
        fleet.InfrastructureFailureEvent(
            claim_sha256=a1.sha256,
            lease_sha256=lease_a.sha256,
            shard_spec_sha256=lease_a.shard_spec_sha256,
            attempt_number=1,
            failure_kind="provider_timeout",
            failure_evidence_sha256=sha("6"),
            valid_worker_artifact_exists=True,
        )


def test_write_once_store_claims_before_attempt_and_blocks_duplicate_claim(tmp_path: Path):
    lease_a, _ = leases()
    a1 = claim(lease_a, 1, "run-a1")
    fleet.persist_lease(tmp_path, lease_a)
    fleet.persist_claim(tmp_path, lease_a, a1)
    attempt_root = fleet.prepare_attempt_namespace(tmp_path, a1)
    assert attempt_root.is_dir()
    with pytest.raises(FileExistsError):
        fleet.persist_claim(tmp_path, lease_a, a1)
    with pytest.raises(FileExistsError):
        fleet.prepare_attempt_namespace(tmp_path, a1)


def test_write_once_store_allows_retry_only_after_persisted_infrastructure_failure(
    tmp_path: Path,
):
    lease_a, _ = leases()
    a1 = claim(lease_a, 1, "run-a1")
    a2 = claim(lease_a, 2, "run-a2")
    fleet.persist_lease(tmp_path, lease_a)
    fleet.persist_claim(tmp_path, lease_a, a1)
    with pytest.raises(ValueError, match="prior terminal outcome"):
        fleet.persist_claim(tmp_path, lease_a, a2)
    fleet.persist_outcome(tmp_path, failed(a1))
    fleet.persist_claim(tmp_path, lease_a, a2)
    assert (tmp_path / "claims" / lease_a.sha256 / "attempt-0002.json").is_file()


def test_provider_run_cannot_be_reused_across_leases():
    auth = authority()
    lease_a, lease_b = leases()
    a1 = claim(lease_a, 1, "same-run")
    b1 = claim(lease_b, 1, "same-run")
    with pytest.raises(ValueError, match="duplicate lease attempt or provider run"):
        fleet.reconstruct_settlement(auth, [lease_a, lease_b], [a1, b1], [])


def test_incomplete_fleet_never_claims_completion():
    auth = authority()
    lease_a, lease_b = leases()
    a1 = claim(lease_a, 1, "run-a1")
    result = fleet.reconstruct_settlement(
        auth,
        [lease_a, lease_b],
        [a1],
        [accepted(a1)],
    )
    assert result["complete"] is False
    assert result["accepted_artifact_count"] == 1
    assert result["pending_lease_count"] == 1


def test_environment_authority_propagates_through_fleet_and_settlement():
    auth = authority()
    lease_a, lease_b = leases()
    assert auth.environment_authority_sha256 == sha("c")
    assert lease_a.environment_authority_sha256 == sha("c")
    assert lease_b.environment_authority_sha256 == sha("c")

    a1 = claim(lease_a, 1, "run-a1")
    b1 = claim(lease_b, 1, "run-b1")
    with pytest.raises(ValueError, match="environment authority differs"):
        fleet.reconstruct_settlement(
            auth,
            [lease_a, lease_b],
            [a1, b1],
            [accepted(a1, environment="4"), accepted(b1)],
        )


def test_persisted_seals_are_revalidated_before_attempt_retry_and_outcome(tmp_path: Path):
    lease_a, _ = leases()
    a1 = claim(lease_a, 1, "run-a1")
    a2 = claim(lease_a, 2, "run-a2")

    fleet.persist_lease(tmp_path, lease_a)
    claim_path = fleet.persist_claim(tmp_path, lease_a, a1)

    tampered_claim = json.loads(claim_path.read_text())
    tampered_claim["provider_run_id"] = "forged-run"
    claim_path.write_text(json.dumps(tampered_claim), encoding="utf-8")
    with pytest.raises(ValueError, match="seal does not match"):
        fleet.prepare_attempt_namespace(tmp_path, a1)

    claim_path.write_text(json.dumps(a1.to_dict()), encoding="utf-8")
    fleet.persist_outcome(tmp_path, failed(a1))
    outcome_path = tmp_path / "outcomes" / lease_a.sha256 / "attempt-0001.json"
    tampered_outcome = json.loads(outcome_path.read_text())
    tampered_outcome["failure_kind"] = "provider_timeout"
    outcome_path.write_text(json.dumps(tampered_outcome), encoding="utf-8")
    with pytest.raises(ValueError, match="outcome seal does not match"):
        fleet.persist_claim(tmp_path, lease_a, a2)
