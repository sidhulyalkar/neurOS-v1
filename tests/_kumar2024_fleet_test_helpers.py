"""Shared fixtures for Kumar2024 FleetAuthority adversarial tests."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from neuros.evidence.kumar2024_comparison import Kumar2024ComparisonPlan, MethodOptimizationSeedPolicy, promoted_external_floor_plan
from neuros.evidence.kumar2024_promoted_execution import PromotedShardResult, bind_promoted_execution_template, build_promoted_execution_template
from neuros.evidence.kumar2024_promoted_fleet import (
    PromotedFleetAttemptRecord, PromotedFleetAuthority, PromotedFleetLedger, PromotedShardLease,
    assemble_promoted_fleet, build_promoted_fleet_authority_from_binding,
    record_infrastructure_failure, record_verified_worker_bundle,
)
from neuros.evidence._kumar2024_promoted_fleet_common import _build_promoted_fleet_authority_from_plan
from neuros.evidence._kumar2024_promoted_fleet_ledger import _record_worker_artifact

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
    authority = _build_promoted_fleet_authority_from_plan(
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
            "balanced_accuracy": _score(shard, budget) if status == "success" else None,
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


def _execution_artifact_payload(execution):
    return {
        "schema_version": 1,
        "template": {
            **execution.template.to_dict(),
            "template_sha256": execution.template.sha256,
        },
        "binding": {
            **execution.binding.to_dict(),
            "binding_sha256": execution.binding.sha256,
        },
        "execution_plan_sha256": execution.sha256,
    }


def _write_shard_result(root, result: PromotedShardResult) -> str:
    path = root / "shard_result.json"
    path.write_text(
        json.dumps({**result.to_dict(), "shard_result_sha256": result.sha256}),
        encoding="utf-8",
    )
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mock_verified_receipt(monkeypatch, *, authority, shard, result, shard_file_sha):
    from neuros.evidence import kumar2024_promoted_worker as worker

    worker_bundle_sha = _sha(f"verified-worker-bundle|{shard.sha256}|{result.sha256}")
    monkeypatch.setattr(
        worker,
        "verify_promoted_worker_bundle",
        lambda output, *, binding_root: {
            "verified": True,
            "worker_bundle_sha256": worker_bundle_sha,
            "binding_bundle_sha256": authority.binding_bundle_sha256,
            "execution_plan_sha256": authority.execution_plan_sha256,
            "shard_spec_sha256": shard.sha256,
            "shard_result_sha256": result.sha256,
            "files": {"shard_result.json": shard_file_sha},
        },
    )
    return worker_bundle_sha

__all__ = [name for name in globals() if not name.startswith("__")]
