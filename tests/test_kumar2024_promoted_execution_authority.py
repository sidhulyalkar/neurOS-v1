from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from neuros.evidence.kumar2024_comparison import (
    Kumar2024ComparisonPlan,
    MethodOptimizationSeedPolicy,
    promoted_external_floor_plan,
)
from neuros.evidence.kumar2024_promoted_execution import (
    PromotedShardResult,
    assemble_promoted_execution,
    bind_promoted_execution_template,
    build_promoted_execution_template,
    validate_promoted_shard_result,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _small_plan() -> Kumar2024ComparisonPlan:
    return Kumar2024ComparisonPlan(
        plan_id="fixture-promoted-execution-v1",
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


def _case_sha(key: tuple[int, str, int]) -> str:
    return _sha(f"case|{key[0]}|{key[1]}|{key[2]}")


def _binding_inputs(template):
    case_map = {
        key: _case_sha(key)
        for key in sorted({_case_key(shard) for shard in template.shards})
    }
    method_map = {
        key: _sha(f"method|{key}")
        for key in template.method_realization_keys
    }
    return case_map, method_map


def _execution_plan(plan: Kumar2024ComparisonPlan):
    template = build_promoted_execution_template(plan)
    case_map, method_map = _binding_inputs(template)
    execution = bind_promoted_execution_template(
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
    return execution


def _score(shard, budget: int) -> float:
    base = 0.60 if shard.method_id == "mne-csp-lda" else 0.68
    base += 0.01 if shard.subject == 10 else 0.0
    base += 0.005 if shard.target_session == "2" else 0.0
    base += 0.003 if shard.split_seed == 3407 else 0.0
    base += 0.004 if shard.model_seed == 22 else 0.0
    return base + 0.01 * budget


def _rows_for_shard(shard, case_sha: str):
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
            "status": "success",
            "balanced_accuracy": _score(shard, budget),
            "qualification_model_state": {
                "metadata": {"fixture": [shard.shard_id, budget]}
            },
        }
        for budget in shard.budgets_per_class
    )


def _result_for_shard(shard, execution, *, case_sha: str | None = None, rows=None):
    binding = execution.binding
    expected_case = execution.expected_case_authority_sha256(shard)
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
        case_authority_sha256=case_sha or expected_case,
        method_spec_sha256=execution.expected_method_spec_sha256(shard),
        rows=_rows_for_shard(shard, case_sha or expected_case) if rows is None else rows,
    )


def test_promoted_external_floor_expands_to_exact_atomic_shard_graph():
    plan = promoted_external_floor_plan()
    template = build_promoted_execution_template(plan)

    assert len(template.shards) == 1350
    assert template.expected_fit_attempts == 6750
    assert len({_case_key(shard) for shard in template.shards}) == 270
    assert Counter(shard.method_id for shard in template.shards) == {
        "mne-csp-lda": 270,
        "pyriemann-rg-lr": 270,
        "braindecode-eegnet": 810,
    }
    assert len(template.method_realization_keys) == 5
    assert all(shard.budgets_per_class == plan.budgets_per_class for shard in template.shards)
    assert template.sha256 == build_promoted_execution_template(plan).sha256


def test_binding_requires_every_case_and_method_realization_authority():
    plan = _small_plan()
    template = build_promoted_execution_template(plan)
    case_map, method_map = _binding_inputs(template)

    missing_case = dict(case_map)
    missing_case.pop(next(iter(missing_case)))
    with pytest.raises(ValueError, match="case-authority map does not match template cases"):
        bind_promoted_execution_template(
            template,
            study_materialization_sha256=_sha("study"),
            environment_authority_sha256=_sha("environment"),
            raw_materialization_sha256=_sha("raw"),
            dataset_lineage_sha256=_sha("lineage"),
            protocol_sha256=_sha("protocol"),
            preprocessing_authority_sha256=_sha("preprocessing"),
            source_revision="a" * 40,
            case_authority_sha256_by_case=missing_case,
            method_spec_sha256_by_realization=method_map,
        )

    missing_method = dict(method_map)
    missing_method.pop(next(iter(missing_method)))
    with pytest.raises(ValueError, match="method-spec authority map does not match template realizations"):
        bind_promoted_execution_template(
            template,
            study_materialization_sha256=_sha("study"),
            environment_authority_sha256=_sha("environment"),
            raw_materialization_sha256=_sha("raw"),
            dataset_lineage_sha256=_sha("lineage"),
            protocol_sha256=_sha("protocol"),
            preprocessing_authority_sha256=_sha("preprocessing"),
            source_revision="a" * 40,
            case_authority_sha256_by_case=case_map,
            method_spec_sha256_by_realization=missing_method,
        )


def test_shard_result_is_deeply_immutable_and_serialization_is_detached():
    plan = _small_plan()
    execution = _execution_plan(plan)
    shard = execution.template.shards[0]
    result = _result_for_shard(shard, execution)
    original_sha = result.sha256

    with pytest.raises(TypeError):
        result.rows[0]["balanced_accuracy"] = 0.99
    with pytest.raises(TypeError):
        result.rows[0]["qualification_model_state"]["metadata"]["x"] = 1

    payload = result.to_dict()
    payload["rows"][0]["balanced_accuracy"] = 0.01
    payload["rows"][0]["qualification_model_state"]["metadata"]["fixture"][0] = "changed"
    assert result.sha256 == original_sha


def test_shard_validation_requires_atomic_frontier_and_exact_bound_case_authority():
    plan = _small_plan()
    execution = _execution_plan(plan)
    shard = execution.template.shards[0]

    result = _result_for_shard(shard, execution)
    validated = validate_promoted_shard_result(
        result, execution_plan=execution, comparison_plan=plan
    )
    assert [row["calibration_per_class"] for row in validated] == list(shard.budgets_per_class)

    short_rows = _rows_for_shard(shard, execution.expected_case_authority_sha256(shard))[:-1]
    short = _result_for_shard(shard, execution, rows=short_rows)
    with pytest.raises(ValueError, match="exactly one attempted row for every calibration budget"):
        validate_promoted_shard_result(
            short, execution_plan=execution, comparison_plan=plan
        )

    wrong_case = _result_for_shard(shard, execution, case_sha=_sha("wrong-case"))
    with pytest.raises(ValueError, match="case-authority SHA differs"):
        validate_promoted_shard_result(
            wrong_case, execution_plan=execution, comparison_plan=plan
        )


def test_assembly_rejects_missing_duplicate_and_foreign_shards():
    plan = _small_plan()
    execution = _execution_plan(plan)
    results = [_result_for_shard(shard, execution) for shard in execution.template.shards]

    with pytest.raises(ValueError, match="missing 1 expected shard"):
        assemble_promoted_execution(
            results[:-1], execution_plan=execution, comparison_plan=plan
        )

    with pytest.raises(ValueError, match="duplicate promoted shard result"):
        assemble_promoted_execution(
            [*results, results[0]], execution_plan=execution, comparison_plan=plan
        )

    shard = execution.template.shards[0]
    foreign = PromotedShardResult(
        **{
            **_result_for_shard(shard, execution).to_dict(),
            "shard_spec_sha256": _sha("foreign-shard"),
        }
    )
    with pytest.raises(ValueError, match="unknown promoted shard result"):
        assemble_promoted_execution(
            [foreign, *results[1:]], execution_plan=execution, comparison_plan=plan
        )


def test_complete_small_execution_delegates_to_participant_level_comparison_authority():
    plan = _small_plan()
    execution = _execution_plan(plan)
    results = [_result_for_shard(shard, execution) for shard in execution.template.shards]

    assembled = assemble_promoted_execution(
        results, execution_plan=execution, comparison_plan=plan
    )

    assert assembled["expected_shards"] == 24
    assert assembled["received_shards"] == 24
    assert assembled["attempted_rows"] == 72
    assert len(assembled["shard_result_sha256s"]) == 24
    analysis = assembled["analysis"]
    assert analysis["comparison_plan_sha256"] == plan.sha256
    assert analysis["independent_inferential_unit"] == "participant"
    assert analysis["primary_study_endpoint"] == (
        "paired_normalized_balanced_accuracy_frontier_auc"
    )
    assert all(
        item["participant_frontier_auc"]["n_participants"] == 2
        for item in analysis["method_frontier_auc"]
    )
