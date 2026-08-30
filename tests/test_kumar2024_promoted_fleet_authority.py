from __future__ import annotations

from _kumar2024_fleet_test_helpers import *

def test_full_promoted_fleet_authority_is_exact_deterministic_and_non_claiming():
    plan = promoted_external_floor_plan()
    execution = _execution_plan(plan)
    first = _build_promoted_fleet_authority_from_plan(
        execution,
        binding_bundle_sha256=_sha("binding-bundle"),
    )
    second = _build_promoted_fleet_authority_from_plan(
        execution,
        binding_bundle_sha256=_sha("binding-bundle"),
    )

    assert first.sha256 == second.sha256
    assert first.expected_shards == 1350
    assert len(first.shard_map) == 1350
    assert len(first.shard_id_by_sha256) == 1350
    assert first.max_infrastructure_retries == 2
    assert first.max_attempts_per_shard == 3
    assert "low_accuracy" not in first.allowed_infrastructure_failure_codes
    assert "runner_unavailable" in first.allowed_infrastructure_failure_codes
    assert first.to_dict()["claim_boundary"] == {
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
    }


def test_public_builder_requires_verified_binding_and_binds_receipt(tmp_path, monkeypatch):
    execution = _execution_plan(_small_plan())
    (tmp_path / "execution_plan.json").write_text(
        json.dumps(_execution_artifact_payload(execution)),
        encoding="utf-8",
    )
    from neuros.evidence import kumar2024_promoted_binding as binding

    bundle_sha = _sha("verified-binding-bundle")
    monkeypatch.setattr(
        binding,
        "verify_promoted_binding_bundle",
        lambda root: {
            "verified": True,
            "bundle_sha256": bundle_sha,
            "execution_plan_sha256": execution.sha256,
        },
    )
    authority = build_promoted_fleet_authority_from_binding(tmp_path)
    assert authority.binding_bundle_sha256 == bundle_sha
    assert authority.execution_plan_sha256 == execution.sha256
    assert authority.source_revision == execution.binding.source_revision


def test_public_builder_rejects_binding_receipt_execution_mismatch(tmp_path, monkeypatch):
    execution = _execution_plan(_small_plan())
    (tmp_path / "execution_plan.json").write_text(
        json.dumps(_execution_artifact_payload(execution)),
        encoding="utf-8",
    )
    from neuros.evidence import kumar2024_promoted_binding as binding

    monkeypatch.setattr(
        binding,
        "verify_promoted_binding_bundle",
        lambda root: {
            "verified": True,
            "bundle_sha256": _sha("verified-binding-bundle"),
            "execution_plan_sha256": _sha("foreign-plan"),
        },
    )
    with pytest.raises(ValueError, match="receipt and execution plan differ"):
        build_promoted_fleet_authority_from_binding(tmp_path)


def test_frozen_first_classical_shard_selection_cannot_drift():
    template = build_promoted_execution_template(promoted_external_floor_plan())
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


def test_empty_full_fleet_exposes_one_initial_lease_per_shard():
    _, authority, ledger = _fleet(promoted_external_floor_plan())
    leases = ledger.dispatchable_leases()
    assert len(leases) == authority.expected_shards == 1350
    assert len({lease.sha256 for lease in leases}) == 1350
    assert len({lease.artifact_key for lease in leases}) == 1350
    assert all(lease.attempt_index == 0 for lease in leases)
    assert all(lease.fleet_authority_sha256 == authority.sha256 for lease in leases)


def test_lease_identity_and_artifact_key_are_scheduler_agnostic():
    _, authority, ledger = _fleet(_small_plan())
    shard_sha = authority.shard_spec_sha256_by_id[0][1]
    first = ledger.next_lease(shard_sha)
    second = ledger.next_lease(shard_sha)
    assert first == second
    assert first.sha256 == second.sha256
    assert first.artifact_key == second.artifact_key
    assert shard_sha in first.artifact_key
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
                    infrastructure_failure_code="runner_unavailable",
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
                    infrastructure_failure_code="runner_unavailable",
                ),
            ),
        )


def test_attempt_indices_are_contiguous_and_append_requires_exact_next_lease():
    _, authority, ledger = _fleet(_small_plan())
    shard_id, shard_sha = authority.shard_spec_sha256_by_id[0]
    gap = PromotedFleetAttemptRecord(
        lease=PromotedShardLease(
            fleet_authority_sha256=authority.sha256,
            execution_plan_sha256=authority.execution_plan_sha256,
            shard_id=shard_id,
            shard_spec_sha256=shard_sha,
            attempt_index=1,
        ),
        outcome="infrastructure_failure",
        infrastructure_failure_code="runner_unavailable",
    )
    with pytest.raises(ValueError, match="contiguous from zero"):
        PromotedFleetLedger(authority=authority, attempts=(gap,))
    with pytest.raises(ValueError, match="exact next fleet lease"):
        ledger.append(gap)


def test_infrastructure_failure_is_retryable_with_new_content_addressed_lease():
    _, authority, ledger = _fleet(_small_plan(), retries=2)
    shard_sha = authority.shard_spec_sha256_by_id[0][1]
    lease0 = ledger.next_lease(shard_sha)
    ledger = record_infrastructure_failure(
        ledger,
        lease0,
        failure_code="binding_artifact_download_failed",
        failure_detail="synthetic fixture detail",
    )
    lease1 = ledger.next_lease(shard_sha)
    assert lease1.attempt_index == 1
    assert lease1.sha256 != lease0.sha256
    assert lease1.artifact_key != lease0.artifact_key
    record = ledger.attempts_by_shard_sha256[shard_sha][0]
    assert record.worker_bundle_sha256 is None
    assert record.shard_result_sha256 is None
    assert record.infrastructure_failure_detail_sha256 == hashlib.sha256(
        b"neuros.kumar2024_promoted_infrastructure_failure_detail.v1\0"
        b"synthetic fixture detail"
    ).hexdigest()


def test_infrastructure_failure_code_rejects_free_text_and_scientific_retry_signal():
    _, authority, ledger = _fleet(_small_plan())
    lease = ledger.next_lease(authority.shard_spec_sha256_by_id[0][1])
    with pytest.raises(ValueError, match="stable lowercase machine code"):
        record_infrastructure_failure(
            ledger,
            lease,
            failure_code="Runner crashed: timeout after 10m",
        )
    with pytest.raises(ValueError, match="not authorized by fleet authority"):
        record_infrastructure_failure(
            ledger,
            lease,
            failure_code="low_accuracy",
        )
    with pytest.raises(ValueError, match="not authorized by fleet authority"):
        record_infrastructure_failure(
            ledger,
            lease,
            failure_code="model_nonconvergence",
        )


def test_retry_budget_exhaustion_is_explicit_and_finite():
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


def test_internal_worker_admission_is_terminal_even_when_scientific_rows_fail():
    plan = _small_plan()
    execution, _, ledger = _fleet(plan)
    shard = execution.template.shards[0]
    lease = ledger.next_lease(shard.sha256)
    failed_result = _result_for_shard(shard, execution, status="failed")
    ledger = _record_worker_artifact(
        ledger,
        lease,
        worker_bundle_sha256=_sha("valid-worker-bundle-with-failure-rows"),
        shard_result=failed_result,
        execution_plan=execution,
        comparison_plan=plan,
    )
    assert ledger.accepted_result_map[shard.sha256].shard_result_sha256 == failed_result.sha256
    with pytest.raises(ValueError, match="already closes this shard"):
        ledger.next_lease(shard.sha256)


def test_public_api_does_not_export_unverified_worker_admission():
    from neuros.evidence import kumar2024_promoted_fleet as fleet

    assert "record_worker_artifact" not in fleet.__all__
    assert "_record_worker_artifact" not in fleet.__all__
    assert "record_verified_worker_bundle" in fleet.__all__
    assert "build_promoted_fleet_authority_from_binding" in fleet.__all__


# Keep the full integrity/assembly adversarial surface owned by this CI entrypoint.
from test_kumar2024_promoted_fleet_integrity import *  # noqa: E402,F401,F403
