from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from neuros.evidence.kumar2024_comparison import (
    Kumar2024ComparisonPlan,
    MethodOptimizationSeedPolicy,
    promoted_external_floor_plan,
)
from neuros.evidence.kumar2024_promoted_execution import (
    PromotedShardResult,
    bind_promoted_execution_template,
    build_promoted_execution_template,
)
from neuros.evidence.kumar2024_promoted_fleet import (
    PromotedFleetAttemptRecord,
    PromotedFleetAuthority,
    PromotedFleetLedger,
    PromotedShardLease,
    assemble_promoted_fleet,
    build_promoted_fleet_authority,
    record_infrastructure_failure,
    record_worker_artifact,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _small_plan() -> Kumar2024ComparisonPlan:
    return Kumar2024ComparisonPlan(
        plan_id="fixture-promoted-fleet-v1",
        subjects=(1, 10),
        target_sessions=("1", "2"),
        budgets_per_class=(0, 1, 2),
        split_seeds=(2026, 3407),
        method_seed_policies=(
            MethodOptimizationSeedPolicy(
                method_id="mne-csp-lda",
                stochastic=False,
            ),
            MethodOptimizationSeedPolicy(
                method_id="braindecode-eegnet",
                stochastic=True,
                model_seeds=(11, 22),
                seed_source="fixture seeds fixed before synthetic scores",
            ),
        ),
        analysis_seed=77,
        bootstrap_replicates=32,
    )


def _case_key(shard) -> tuple[int, str, int]:
    return shard.subject, shard.target_session, shard.split_seed


def _execution_plan(plan: Kumar2024ComparisonPlan):
    template = build_promoted_execution_template(plan)
    case_map = {
        key: _sha(f"case|{key[0]}|{key[1]}|{key[2]}")
        for key in sorted({_case_key(shard) for shard in template.shards})
    }
    method_map = {
        key: _sha(f"method|{key}")
        for key in template.method_realization_keys
    }
    return bind_promoted_execution_template(
        template,
        study_materialization_sha256=_sha("study-materialization"),
        environment_authority_sha256=_sha("environment"),
        raw_materialization_sha256=_sha("raw"),
        dataset_lineage_sha256=_sha("lineage"),
        protocol_sha256=_sha("protocol"),
        preprocessing_authority_sha256=_sha("preprocessing"),
        source_revision="a" * 40,
        case_authority_sha256_by_case=case_map,
        method_spec_sha256_by_realization=method_map,
    )


def _fleet(plan: Kumar2024ComparisonPlan, *, retries: int = 2):
    execution = _execution_plan(plan)
    authority = build_promoted_fleet_authority(
        execution,
        binding_bundle_sha256=_sha("binding-bundle"),
        max_infrastructure_retries=retries,
    )
    return execution, authority, PromotedFleetLedger(authority=authority)


def _score(shard, budget: int) -> float:
    base = 0.60 if shard.method_id == "mne-csp-lda" else 0.68
    base += 0.01 if shard.subject == 10 else 0.0
    base += 0.005 if shard.target_session == "2" else 0.0
    base += 0.003 if shard.split_seed == 3407 else 0.0
    base += 0.004 if shard.model_seed == 22 else 0.0
    return base + 0.01 * budget


def _rows_for_shard(shard, case_sha: str, *, status: str = "success"):
    cohort = "GR" if shard.subject <= 9 else "PAR"
    return tuple(
        {
            "method_id": shard.method_id,
            "subject": shard.subject,
            "held_out_session": shard.target_session,
            "split_seed": shard.split_seed,
            "model_seed": shard.model_seed,
            "calibration_per_class": budget,
            "case_authority_sha256": case_sha,
            "original_protocol": cohort,
            "status": status,
            "balanced_accuracy": (
                _score(shard, budget) if status == "success" else None
            ),
            "qualification_model_state": {
                "metadata": {"fixture": [shard.shard_id, budget]}
            },
        }
        for budget in shard.budgets_per_class
    )


def _result_for_shard(shard, execution, *, status: str = "success"):
    binding = execution.binding
    case_sha = execution.expected_case_authority_sha256(shard)
    return PromotedShardResult(
        execution_plan_sha256=execution.sha256,
        shard_spec_sha256=shard.sha256,
        comparison_plan_sha256=execution.template.comparison_plan_sha256,
        study_materialization_sha256=binding.study_materialization_sha256,
        environment_authority_sha256=binding.environment_authority_sha256,
        raw_materialization_sha256=binding.raw_materialization_sha256,
        dataset_lineage_sha256=binding.dataset_lineage_sha256,
        protocol_sha256=binding.protocol_sha256,
        preprocessing_authority_sha256=binding.preprocessing_authority_sha256,
        case_authority_sha256=case_sha,
        method_spec_sha256=execution.expected_method_spec_sha256(shard),
        rows=_rows_for_shard(shard, case_sha, status=status),
    )


def test_full_promoted_fleet_authority_is_exact_deterministic_and_non_claiming():
    plan = promoted_external_floor_plan()
    execution = _execution_plan(plan)
    first = build_promoted_fleet_authority(
        execution,
        binding_bundle_sha256=_sha("binding-bundle"),
    )
    second = build_promoted_fleet_authority(
        execution,
        binding_bundle_sha256=_sha("binding-bundle"),
    )

    assert first.sha256 == second.sha256
    assert first.expected_shards == 1350
    assert len(first.shard_map) == 1350
    assert len(first.shard_id_by_sha256) == 1350
    assert first.max_infrastructure_retries == 2
    assert first.max_attempts_per_shard == 3
    assert first.to_dict()["claim_boundary"] == {
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
    }


def test_frozen_first_classical_shard_selection_cannot_drift():
    plan = promoted_external_floor_plan()
    template = build_promoted_execution_template(plan)
    selected = [
        shard
        for shard in template.shards
        if shard.shard_id
        == "subject-01/session-1/split-2026/mne-csp-lda/deterministic"
    ]
    assert len(selected) == 1
    assert (
        selected[0].sha256
        == "b6943a6bd0692fb99c14d3b57b2eea04ea8bf16b79b92a18415912f2b8381ceb"
    )


def test_empty_full_fleet_exposes_exactly_one_initial_lease_per_shard():
    plan = promoted_external_floor_plan()
    _, authority, ledger = _fleet(plan)

    leases = ledger.dispatchable_leases()
    assert len(leases) == authority.expected_shards == 1350
    assert len({lease.sha256 for lease in leases}) == 1350
    assert len({lease.artifact_key for lease in leases}) == 1350
    assert all(lease.attempt_index == 0 for lease in leases)
    assert all(lease.fleet_authority_sha256 == authority.sha256 for lease in leases)


def test_lease_identity_and_artifact_key_are_deterministic_and_scheduler_agnostic():
    _, authority, ledger = _fleet(_small_plan())
    shard_sha = authority.shard_spec_sha256_by_id[0][1]

    first = ledger.next_lease(shard_sha)
    second = ledger.next_lease(shard_sha)

    assert first == second
    assert first.sha256 == second.sha256
    assert first.artifact_key == second.artifact_key
    assert first.shard_spec_sha256 in first.artifact_key
    lowered = first.artifact_key.lower()
    assert "github" not in lowered
    assert "runner" not in lowered
    assert "hostname" not in lowered


def test_lease_must_match_fleet_execution_shard_and_retry_budget():
    _, authority, _ = _fleet(_small_plan(), retries=1)
    shard_id, shard_sha = authority.shard_spec_sha256_by_id[0]
    other_sha = authority.shard_spec_sha256_by_id[1][1]

    with pytest.raises(ValueError, match="shard id and shard SHA"):
        PromotedFleetLedger(
            authority=authority,
            attempts=(
                PromotedFleetAttemptRecord(
                    lease=PromotedShardLease(
                        fleet_authority_sha256=authority.sha256,
                        execution_plan_sha256=authority.execution_plan_sha256,
                        shard_id=shard_id,
                        shard_spec_sha256=other_sha,
                        attempt_index=0,
                    ),
                    outcome="infrastructure_failure",
                    infrastructure_failure_code="runner_bootstrap",
                ),
            ),
        )

    with pytest.raises(ValueError, match="retry budget"):
        PromotedFleetLedger(
            authority=authority,
            attempts=(
                PromotedFleetAttemptRecord(
                    lease=PromotedShardLease(
                        fleet_authority_sha256=authority.sha256,
                        execution_plan_sha256=authority.execution_plan_sha256,
                        shard_id=shard_id,
                        shard_spec_sha256=shard_sha,
                        attempt_index=2,
                    ),
                    outcome="infrastructure_failure",
                    infrastructure_failure_code="runner_bootstrap",
                ),
            ),
        )


def test_attempt_indices_are_contiguous_and_append_requires_exact_next_lease():
    _, authority, ledger = _fleet(_small_plan())
    shard_id, shard_sha = authority.shard_spec_sha256_by_id[0]

    gap_record = PromotedFleetAttemptRecord(
        lease=PromotedShardLease(
            fleet_authority_sha256=authority.sha256,
            execution_plan_sha256=authority.execution_plan_sha256,
            shard_id=shard_id,
            shard_spec_sha256=shard_sha,
            attempt_index=1,
        ),
        outcome="infrastructure_failure",
        infrastructure_failure_code="network",
    )
    with pytest.raises(ValueError, match="contiguous from zero"):
        PromotedFleetLedger(authority=authority, attempts=(gap_record,))

    stale = replace(gap_record.lease, attempt_index=1)
    with pytest.raises(ValueError, match="exact next fleet lease"):
        ledger.append(
            PromotedFleetAttemptRecord(
                lease=stale,
                outcome="infrastructure_failure",
                infrastructure_failure_code="network",
            )
        )


def test_infrastructure_failure_is_retryable_with_new_content_addressed_lease():
    _, authority, ledger = _fleet(_small_plan(), retries=2)
    shard_sha = authority.shard_spec_sha256_by_id[0][1]

    lease0 = ledger.next_lease(shard_sha)
    ledger = record_infrastructure_failure(
        ledger,
        lease0,
        failure_code="artifact_download",
        failure_detail="synthetic fixture detail",
    )
    lease1 = ledger.next_lease(shard_sha)

    assert lease1.attempt_index == 1
    assert lease1.sha256 != lease0.sha256
    assert lease1.artifact_key != lease0.artifact_key
    record = ledger.attempts_by_shard_sha256[shard_sha][0]
    assert record.outcome == "infrastructure_failure"
    assert record.worker_bundle_sha256 is None
    assert record.shard_result_sha256 is None
    assert record.infrastructure_failure_detail_sha256 == hashlib.sha256(
        b"neuros.kumar2024_promoted_infrastructure_failure_detail.v1\0"
        b"synthetic fixture detail"
    ).hexdigest()


def test_retry_budget_exhaustion_is_explicit_and_cannot_loop_forever():
    _, authority, ledger = _fleet(_small_plan(), retries=1)
    shard_sha = authority.shard_spec_sha256_by_id[0][1]

    for attempt in range(2):
        lease = ledger.next_lease(shard_sha)
        assert lease.attempt_index == attempt
        ledger = record_infrastructure_failure(
            ledger,
            lease,
            failure_code="runner_unavailable",
        )

    with pytest.raises(ValueError, match="retry budget is exhausted"):
        ledger.next_lease(shard_sha)
    assert ledger.exhausted_without_artifact == (shard_sha,)
    assert not ledger.complete


def test_valid_worker_artifact_is_terminal_even_when_scientific_rows_fail():
    plan = _small_plan()
    execution, _, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    lease = ledger.next_lease(shard.sha256)
    failed_result = _result_for_shard(shard, execution, status="failed")

    ledger = record_worker_artifact(
        ledger,
        lease,
        worker_bundle_sha256=_sha("valid-worker-bundle-with-failure-rows"),
        shard_result=failed_result,
        execution_plan=execution,
        comparison_plan=plan,
    )

    accepted = ledger.accepted_result_map[shard.sha256]
    assert accepted.shard_result_sha256 == failed_result.sha256
    assert len(ledger.attempts_by_shard_sha256[shard.sha256]) == 1
    with pytest.raises(ValueError, match="already closes this shard"):
        ledger.next_lease(shard.sha256)


def test_no_attempt_may_follow_valid_worker_artifact():
    plan = _small_plan()
    execution, authority, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    lease0 = ledger.next_lease(shard.sha256)
    result = _result_for_shard(shard, execution)
    closed = record_worker_artifact(
        ledger,
        lease0,
        worker_bundle_sha256=_sha("worker-0"),
        shard_result=result,
        execution_plan=execution,
        comparison_plan=plan,
    )

    forged_retry = PromotedFleetAttemptRecord(
        lease=replace(lease0, attempt_index=1),
        outcome="infrastructure_failure",
        infrastructure_failure_code="forged_retry",
    )
    with pytest.raises(ValueError, match="follow a valid worker artifact"):
        PromotedFleetLedger(
            authority=authority,
            attempts=(*closed.attempts, forged_retry),
        )


def test_worker_bundle_and_shard_result_collisions_across_shards_fail_closed():
    _, authority, _ = _fleet(_small_plan())
    (id0, sha0), (id1, sha1) = authority.shard_spec_sha256_by_id[:2]
    common_worker = _sha("collision-worker")
    common_result = _sha("collision-result")

    records = (
        PromotedFleetAttemptRecord(
            lease=PromotedShardLease(
                fleet_authority_sha256=authority.sha256,
                execution_plan_sha256=authority.execution_plan_sha256,
                shard_id=id0,
                shard_spec_sha256=sha0,
                attempt_index=0,
            ),
            outcome="worker_artifact",
            worker_bundle_sha256=common_worker,
            shard_result_sha256=common_result,
        ),
        PromotedFleetAttemptRecord(
            lease=PromotedShardLease(
                fleet_authority_sha256=authority.sha256,
                execution_plan_sha256=authority.execution_plan_sha256,
                shard_id=id1,
                shard_spec_sha256=sha1,
                attempt_index=0,
            ),
            outcome="worker_artifact",
            worker_bundle_sha256=common_worker,
            shard_result_sha256=common_result,
        ),
    )
    with pytest.raises(ValueError, match="worker bundle"):
        PromotedFleetLedger(authority=authority, attempts=records)


def test_record_worker_artifact_rejects_foreign_result_before_closing_shard():
    plan = _small_plan()
    execution, _, ledger = _fleet(plan)
    shard0, shard1 = execution.template.shards[:2]
    lease = ledger.next_lease(shard0.sha256)
    foreign = _result_for_shard(shard1, execution)

    with pytest.raises(ValueError, match="shard differs from lease"):
        record_worker_artifact(
            ledger,
            lease,
            worker_bundle_sha256=_sha("foreign"),
            shard_result=foreign,
            execution_plan=execution,
            comparison_plan=plan,
        )


def test_fleet_authority_lease_attempt_and_ledger_roundtrip_with_hash_checks():
    _, authority, ledger = _fleet(_small_plan())
    authority_payload = {
        **authority.to_dict(),
        "fleet_authority_sha256": authority.sha256,
    }
    restored_authority = PromotedFleetAuthority.from_dict(authority_payload)
    assert restored_authority == authority

    shard_sha = authority.shard_spec_sha256_by_id[0][1]
    lease = ledger.next_lease(shard_sha)
    lease_payload = lease.to_dict()
    assert PromotedShardLease.from_dict(lease_payload) == lease

    ledger = record_infrastructure_failure(
        ledger,
        lease,
        failure_code="network",
    )
    payload = {
        **ledger.to_dict(),
        "fleet_ledger_sha256": ledger.sha256,
    }
    restored = PromotedFleetLedger.from_dict(payload, authority=authority)
    assert restored == ledger
    assert restored.sha256 == ledger.sha256

    tampered = dict(payload)
    tampered["fleet_ledger_sha256"] = _sha("wrong")
    with pytest.raises(ValueError, match="ledger SHA-256 mismatch"):
        PromotedFleetLedger.from_dict(tampered, authority=authority)


def _complete_small_fleet(plan: Kumar2024ComparisonPlan):
    execution, authority, ledger = _fleet(plan)
    results = []
    for index, shard in enumerate(execution.template.shards):
        result = _result_for_shard(shard, execution)
        lease = ledger.next_lease(shard.sha256)
        ledger = record_worker_artifact(
            ledger,
            lease,
            worker_bundle_sha256=_sha(f"worker-bundle-{index}"),
            shard_result=result,
            execution_plan=execution,
            comparison_plan=plan,
        )
        results.append(result)
    return execution, authority, ledger, results


def test_incomplete_fleet_cannot_reach_global_assembly():
    plan = _small_plan()
    execution, _, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    result = _result_for_shard(shard, execution)
    lease = ledger.next_lease(shard.sha256)
    ledger = record_worker_artifact(
        ledger,
        lease,
        worker_bundle_sha256=_sha("one-worker"),
        shard_result=result,
        execution_plan=execution,
        comparison_plan=plan,
    )

    with pytest.raises(ValueError, match="cannot assemble incomplete promoted fleet"):
        assemble_promoted_fleet(
            ledger,
            [result],
            execution_plan=execution,
            comparison_plan=plan,
        )


def test_complete_small_fleet_delegates_to_existing_execution_assembly():
    plan = _small_plan()
    execution, authority, ledger, results = _complete_small_fleet(plan)

    assert ledger.complete
    assert len(ledger.accepted_results) == len(execution.template.shards) == 24
    assembled = assemble_promoted_fleet(
        ledger,
        results,
        execution_plan=execution,
        comparison_plan=plan,
    )

    manifest = assembled["fleet_assembly_manifest"]
    execution_assembly = assembled["execution_assembly"]
    assert manifest["fleet_authority_sha256"] == authority.sha256
    assert manifest["fleet_ledger_sha256"] == ledger.sha256
    assert manifest["expected_shards"] == 24
    assert manifest["attempted_records"] == 24
    assert manifest["global_analysis_performed"] is True
    assert manifest["external_floor_claim_generated"] is False
    assert manifest["orion_comparison_permitted"] is False
    assert execution_assembly["expected_shards"] == 24
    assert execution_assembly["received_shards"] == 24
    assert execution_assembly["attempted_rows"] == 72
    assert execution_assembly["analysis"]["independent_inferential_unit"] == "participant"


def test_complete_fleet_rejects_result_content_different_from_accepted_ledger():
    plan = _small_plan()
    execution, _, ledger, results = _complete_small_fleet(plan)
    original = results[0]
    altered_rows = [dict(row) for row in original.to_dict()["rows"]]
    altered_rows[0]["balanced_accuracy"] = 0.01
    altered = PromotedShardResult(
        execution_plan_sha256=original.execution_plan_sha256,
        shard_spec_sha256=original.shard_spec_sha256,
        comparison_plan_sha256=original.comparison_plan_sha256,
        study_materialization_sha256=original.study_materialization_sha256,
        environment_authority_sha256=original.environment_authority_sha256,
        raw_materialization_sha256=original.raw_materialization_sha256,
        dataset_lineage_sha256=original.dataset_lineage_sha256,
        protocol_sha256=original.protocol_sha256,
        preprocessing_authority_sha256=original.preprocessing_authority_sha256,
        case_authority_sha256=original.case_authority_sha256,
        method_spec_sha256=original.method_spec_sha256,
        rows=tuple(altered_rows),
    )
    assert altered.sha256 != original.sha256

    with pytest.raises(ValueError, match="differs from accepted ledger identity"):
        assemble_promoted_fleet(
            ledger,
            [altered, *results[1:]],
            execution_plan=execution,
            comparison_plan=plan,
        )
