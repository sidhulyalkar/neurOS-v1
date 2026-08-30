from __future__ import annotations

from _kumar2024_fleet_test_helpers import *

def test_no_attempt_may_follow_valid_worker_artifact():
    plan = _small_plan()
    execution, authority, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    lease0 = ledger.next_lease(shard.sha256)
    closed = _record_worker_artifact(
        ledger,
        lease0,
        worker_bundle_sha256=_sha("worker-0"),
        shard_result=_result_for_shard(shard, execution),
        execution_plan=execution,
        comparison_plan=plan,
    )
    forged_retry = PromotedFleetAttemptRecord(
        lease=replace(lease0, attempt_index=1),
        outcome="infrastructure_failure",
        infrastructure_failure_code="runner_unavailable",
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
    records = tuple(
        PromotedFleetAttemptRecord(
            lease=PromotedShardLease(
                fleet_authority_sha256=authority.sha256,
                execution_plan_sha256=authority.execution_plan_sha256,
                shard_id=shard_id,
                shard_spec_sha256=shard_sha,
                attempt_index=0,
            ),
            outcome="worker_artifact",
            worker_bundle_sha256=common_worker,
            shard_result_sha256=common_result,
        )
        for shard_id, shard_sha in ((id0, sha0), (id1, sha1))
    )
    with pytest.raises(ValueError, match="worker bundle"):
        PromotedFleetLedger(authority=authority, attempts=records)


def test_verified_worker_bundle_ingestion_binds_full_receipt_to_ledger(tmp_path, monkeypatch):
    plan = _small_plan()
    execution, authority, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    lease = ledger.next_lease(shard.sha256)
    result = _result_for_shard(shard, execution)
    shard_file_sha = _write_shard_result(tmp_path, result)
    worker_bundle_sha = _mock_verified_receipt(
        monkeypatch,
        authority=authority,
        shard=shard,
        result=result,
        shard_file_sha=shard_file_sha,
    )
    ledger = record_verified_worker_bundle(
        ledger,
        lease,
        worker_root=tmp_path,
        binding_root=tmp_path / "binding",
        execution_plan=execution,
        comparison_plan=plan,
    )
    accepted = ledger.accepted_result_map[shard.sha256]
    assert accepted.worker_bundle_sha256 == worker_bundle_sha
    assert accepted.shard_result_sha256 == result.sha256


def test_verified_worker_bundle_with_failure_rows_is_still_terminal(tmp_path, monkeypatch):
    plan = _small_plan()
    execution, authority, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    lease = ledger.next_lease(shard.sha256)
    result = _result_for_shard(shard, execution, status="failed")
    shard_file_sha = _write_shard_result(tmp_path, result)
    _mock_verified_receipt(
        monkeypatch,
        authority=authority,
        shard=shard,
        result=result,
        shard_file_sha=shard_file_sha,
    )
    ledger = record_verified_worker_bundle(
        ledger,
        lease,
        worker_root=tmp_path,
        binding_root=tmp_path / "binding",
        execution_plan=execution,
        comparison_plan=plan,
    )
    assert shard.sha256 in ledger.accepted_result_map
    with pytest.raises(ValueError, match="already closes this shard"):
        ledger.next_lease(shard.sha256)


def test_verified_worker_bundle_rejects_receipt_from_other_binding(tmp_path, monkeypatch):
    plan = _small_plan()
    execution, authority, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    lease = ledger.next_lease(shard.sha256)
    result = _result_for_shard(shard, execution)
    shard_file_sha = _write_shard_result(tmp_path, result)
    from neuros.evidence import kumar2024_promoted_worker as worker

    monkeypatch.setattr(
        worker,
        "verify_promoted_worker_bundle",
        lambda output, *, binding_root: {
            "verified": True,
            "worker_bundle_sha256": _sha("verified-worker-bundle"),
            "binding_bundle_sha256": _sha("other-binding"),
            "execution_plan_sha256": authority.execution_plan_sha256,
            "shard_spec_sha256": shard.sha256,
            "shard_result_sha256": result.sha256,
            "files": {"shard_result.json": shard_file_sha},
        },
    )
    with pytest.raises(ValueError, match="different fleet binding bundle"):
        record_verified_worker_bundle(
            ledger,
            lease,
            worker_root=tmp_path,
            binding_root=tmp_path / "binding",
            execution_plan=execution,
            comparison_plan=plan,
        )


def test_verified_worker_bundle_detects_shard_file_change_after_verification(tmp_path, monkeypatch):
    plan = _small_plan()
    execution, authority, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    lease = ledger.next_lease(shard.sha256)
    result = _result_for_shard(shard, execution)
    _write_shard_result(tmp_path, result)
    stale_sha = _sha("pre-verification-file")
    _mock_verified_receipt(
        monkeypatch,
        authority=authority,
        shard=shard,
        result=result,
        shard_file_sha=stale_sha,
    )
    with pytest.raises(ValueError, match="changed after bundle verification"):
        record_verified_worker_bundle(
            ledger,
            lease,
            worker_root=tmp_path,
            binding_root=tmp_path / "binding",
            execution_plan=execution,
            comparison_plan=plan,
        )


def test_serialized_authority_rejects_tampered_reviewer_facing_claim_fields():
    _, authority, _ = _fleet(_small_plan())
    payload = {**authority.to_dict(), "fleet_authority_sha256": authority.sha256}
    restored = PromotedFleetAuthority.from_dict(payload)
    assert restored == authority

    tampered = json.loads(json.dumps(payload))
    tampered["claim_boundary"]["orion_comparison_permitted"] = True
    with pytest.raises(ValueError, match="claim_boundary"):
        PromotedFleetAuthority.from_dict(tampered)

    tampered_codes = json.loads(json.dumps(payload))
    tampered_codes["allowed_infrastructure_failure_codes"].append("low_accuracy")
    with pytest.raises(ValueError, match="allowed_infrastructure_failure_codes"):
        PromotedFleetAuthority.from_dict(tampered_codes)


def test_serialized_attempt_rejects_tampered_retry_semantics():
    _, authority, ledger = _fleet(_small_plan())
    lease = ledger.next_lease(authority.shard_spec_sha256_by_id[0][1])
    ledger = record_infrastructure_failure(
        ledger,
        lease,
        failure_code="runner_unavailable",
    )
    record = ledger.attempts[0]
    payload = {**record.to_dict(), "attempt_record_sha256": record.sha256}
    restored = PromotedFleetAttemptRecord.from_dict(payload)
    assert restored == record
    tampered = json.loads(json.dumps(payload))
    tampered["scientific_retry_permitted"] = True
    with pytest.raises(ValueError, match="scientific_retry_permitted"):
        PromotedFleetAttemptRecord.from_dict(tampered)


def test_serialized_ledger_rejects_tampered_derived_summary_and_claims():
    _, authority, ledger = _fleet(_small_plan())
    lease = ledger.next_lease(authority.shard_spec_sha256_by_id[0][1])
    ledger = record_infrastructure_failure(
        ledger,
        lease,
        failure_code="runner_unavailable",
    )
    payload = {**ledger.to_dict(), "fleet_ledger_sha256": ledger.sha256}
    restored = PromotedFleetLedger.from_dict(payload, authority=authority)
    assert restored == ledger
    assert restored.sha256 == ledger.sha256

    tampered_count = json.loads(json.dumps(payload))
    tampered_count["accepted_shards"] = 999
    with pytest.raises(ValueError, match="accepted_shards"):
        PromotedFleetLedger.from_dict(tampered_count, authority=authority)

    tampered_claim = json.loads(json.dumps(payload))
    tampered_claim["claim_boundary"]["numerical_result_interpretable"] = True
    with pytest.raises(ValueError, match="claim_boundary"):
        PromotedFleetLedger.from_dict(tampered_claim, authority=authority)


def _complete_small_fleet(plan: Kumar2024ComparisonPlan):
    execution, authority, ledger = _fleet(plan)
    results = []
    for index, shard in enumerate(execution.template.shards):
        result = _result_for_shard(shard, execution)
        lease = ledger.next_lease(shard.sha256)
        ledger = _record_worker_artifact(
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
    ledger = _record_worker_artifact(
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
    assert manifest["numerical_result_interpretable"] is False
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
