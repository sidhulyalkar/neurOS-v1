from __future__ import annotations

import hashlib

import pytest

from neuros.evidence.kumar2024_comparison import (
    KUMAR2024_EEGNET_MODEL_SEEDS,
    KUMAR2024_PRIMARY_ENDPOINT,
    KUMAR2024_PROMOTED_ANALYSIS_SEED,
    KUMAR2024_PROMOTED_SPLIT_SEEDS,
    Kumar2024ComparisonPlan,
    MethodOptimizationSeedPolicy,
    promoted_external_floor_plan,
    summarize_promoted_rows,
    validate_promoted_rows,
)


def _derived_seed(namespace: str) -> int:
    return int.from_bytes(hashlib.sha256(namespace.encode("utf-8")).digest()[:4], "big")


def _small_plan() -> Kumar2024ComparisonPlan:
    return Kumar2024ComparisonPlan(
        plan_id="fixture-comparison-v1",
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
                method_id="pyriemann-rg-lr",
                stochastic=False,
            ),
            MethodOptimizationSeedPolicy(
                method_id="braindecode-eegnet",
                stochastic=True,
                model_seeds=(11, 22),
                seed_source="fixture predeclared before results",
            ),
        ),
        analysis_seed=77,
        bootstrap_replicates=64,
    )


def _authority_sha(subject: int, session: str, split_seed: int) -> str:
    return hashlib.sha256(
        f"case|{subject}|{session}|{split_seed}".encode("utf-8")
    ).hexdigest()


def _score(
    *,
    method: str,
    subject: int,
    session: str,
    split_seed: int,
    model_seed: int | None,
    budget: int,
) -> float:
    base = {
        "mne-csp-lda": 0.58,
        "pyriemann-rg-lr": 0.63,
        "braindecode-eegnet": 0.68,
    }[method]
    value = base
    value += 0.02 if subject == 10 else 0.0
    value += 0.01 if session == "2" else 0.0
    value += 0.005 if split_seed == 3407 else 0.0
    value += 0.01 if model_seed == 22 else 0.0
    value += {0: 0.0, 1: 0.015, 2: 0.03}[budget]
    return value


def _complete_rows(plan: Kumar2024ComparisonPlan) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method in plan.methods:
        policy = plan.policy_for(method)
        for subject in plan.subjects:
            cohort = "GR" if subject <= 9 else "PAR"
            for session in plan.target_sessions:
                for split_seed in plan.split_seeds:
                    authority_sha = _authority_sha(subject, session, split_seed)
                    for model_seed in policy.realization_model_seeds:
                        for budget in plan.budgets_per_class:
                            rows.append(
                                {
                                    "method_id": method,
                                    "subject": subject,
                                    "held_out_session": session,
                                    "split_seed": split_seed,
                                    "model_seed": model_seed,
                                    "calibration_per_class": budget,
                                    "case_authority_sha256": authority_sha,
                                    "original_protocol": cohort,
                                    "status": "success",
                                    "balanced_accuracy": _score(
                                        method=method,
                                        subject=subject,
                                        session=session,
                                        split_seed=split_seed,
                                        model_seed=model_seed,
                                        budget=budget,
                                    ),
                                }
                            )
    return rows


def _find_summary_method(analysis: dict, method: str) -> dict:
    return next(
        item for item in analysis["method_frontier_auc"] if item["method_id"] == method
    )


def _find_pair(analysis: dict, left: str, right: str) -> dict:
    return next(
        item
        for item in analysis["paired_calibration_efficiency"]
        if item["left_method"] == left and item["right_method"] == right
    )


def _find_failure_summary(analysis: dict, method: str) -> dict:
    return next(
        item for item in analysis["failure_summary"] if item["method_id"] == method
    )


def test_promoted_plan_constants_are_preregistered_and_hash_derived():
    plan = promoted_external_floor_plan()

    assert plan.subjects == tuple(range(1, 19))
    assert plan.target_sessions == ("1", "2", "3", "4", "5")
    assert plan.budgets_per_class == (0, 1, 2, 5, 10)
    assert plan.split_seeds == (2026, 3407, 9109)
    assert plan.primary_endpoint == KUMAR2024_PRIMARY_ENDPOINT
    assert plan.independent_unit == "participant"
    assert plan.aggregation_hierarchy == (
        "participant",
        "target_session",
        "split_seed",
        "model_seed",
        "calibration_budget",
    )
    assert KUMAR2024_PROMOTED_SPLIT_SEEDS == (2026, 3407, 9109)
    assert KUMAR2024_EEGNET_MODEL_SEEDS == tuple(sorted((
        31415,
        _derived_seed("neuros.kumar2024.eegnet.optimization_seed.v1|1"),
        _derived_seed("neuros.kumar2024.eegnet.optimization_seed.v1|2"),
    )))
    assert KUMAR2024_PROMOTED_ANALYSIS_SEED == _derived_seed(
        "neuros.kumar2024.promoted.analysis.bootstrap.v1"
    )
    assert KUMAR2024_PROMOTED_ANALYSIS_SEED not in KUMAR2024_PROMOTED_SPLIT_SEEDS
    assert KUMAR2024_PROMOTED_ANALYSIS_SEED not in KUMAR2024_EEGNET_MODEL_SEEDS
    assert len(plan.sha256) == 64


def test_seed_policy_and_plan_order_fail_closed():
    with pytest.raises(ValueError, match="cannot invent a model-seed axis"):
        MethodOptimizationSeedPolicy(
            method_id="deterministic",
            stochastic=False,
            model_seeds=(1,),
        )
    with pytest.raises(ValueError, match="require predeclared model seeds"):
        MethodOptimizationSeedPolicy(
            method_id="stochastic",
            stochastic=True,
            seed_source="fixed",
        )
    with pytest.raises(ValueError, match="split_seeds must be in increasing"):
        Kumar2024ComparisonPlan(
            plan_id="bad-order",
            subjects=(1, 10),
            target_sessions=("1", "2"),
            budgets_per_class=(0, 1, 2),
            split_seeds=(3407, 2026),
            method_seed_policies=(
                MethodOptimizationSeedPolicy(
                    method_id="mne-csp-lda",
                    stochastic=False,
                ),
            ),
            analysis_seed=77,
        )


def test_validation_rejects_unplanned_axes_bad_scores_duplicates_and_cohort_drift():
    plan = _small_plan()
    rows = _complete_rows(plan)

    bad_model_seed = [dict(row) for row in rows]
    target = next(
        row
        for row in bad_model_seed
        if row["method_id"] == "braindecode-eegnet" and row["model_seed"] == 22
    )
    target["model_seed"] = 33
    with pytest.raises(ValueError, match="unplanned model seed"):
        validate_promoted_rows(bad_model_seed, plan=plan)

    deterministic_seed = [dict(row) for row in rows]
    target = next(
        row for row in deterministic_seed if row["method_id"] == "mne-csp-lda"
    )
    target["model_seed"] = 9
    with pytest.raises(ValueError, match="cannot carry model_seed"):
        validate_promoted_rows(deterministic_seed, plan=plan)

    out_of_range = [dict(row) for row in rows]
    out_of_range[0]["balanced_accuracy"] = 1.2
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_promoted_rows(out_of_range, plan=plan)

    duplicate = [dict(row) for row in rows]
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(ValueError, match="duplicate promoted result row"):
        validate_promoted_rows(duplicate, plan=plan)

    cohort_drift = [dict(row) for row in rows]
    target = next(row for row in cohort_drift if row["subject"] == 10)
    target["original_protocol"] = "GR"
    with pytest.raises(ValueError, match="cohort mismatch"):
        validate_promoted_rows(cohort_drift, plan=plan)


def test_competing_methods_and_model_seeds_must_share_exact_case_authority():
    plan = _small_plan()
    rows = _complete_rows(plan)
    target = next(
        row
        for row in rows
        if row["method_id"] == "braindecode-eegnet"
        and row["subject"] == 1
        and row["held_out_session"] == "1"
        and row["split_seed"] == 2026
        and row["model_seed"] == 11
        and row["calibration_per_class"] == 0
    )
    target["case_authority_sha256"] = "f" * 64

    with pytest.raises(ValueError, match="do not share exact case authority"):
        validate_promoted_rows(rows, plan=plan)


def test_complete_comparison_collapses_repeated_axes_inside_participant():
    plan = _small_plan()
    analysis = summarize_promoted_rows(_complete_rows(plan), plan=plan)

    assert analysis["independent_inferential_unit"] == "participant"
    assert analysis["primary_study_endpoint"] == (
        "paired_normalized_balanced_accuracy_frontier_auc"
    )
    assert analysis["repeated_measure_axes"] == [
        "target_session",
        "split_seed",
        "model_seed",
        "calibration_budget",
    ]

    for method in plan.methods:
        summary = _find_summary_method(analysis, method)
        assert summary["complete_frontier_participants"] == [1, 10]
        assert summary["participant_frontier_auc"]["n_participants"] == 2

    pair = _find_pair(analysis, "mne-csp-lda", "braindecode-eegnet")
    assert pair["matched_complete_frontier_participants"] == [1, 10]
    assert pair[
        "left_minus_right_normalized_balanced_accuracy_frontier_auc"
    ]["n_participants"] == 2

    # The synthetic input contains many rows per person, but inferential N remains 2.
    assert len(_complete_rows(plan)) > 2
    assert all(
        item["participant_balanced_accuracy"]["n_participants"] == 2
        for item in analysis["secondary_pointwise_performance"]
    )

    frontier_diagnostics = analysis["frontier_diagnostics"]
    assert len(frontier_diagnostics["model_realization_frontier_auc"]) == 32
    assert len(frontier_diagnostics["model_seed_averaged_split_frontier_auc"]) == 24
    assert len(frontier_diagnostics["split_seed_averaged_session_frontier_auc"]) == 12
    assert "descriptive traceability only" in analysis["frontier_diagnostics_policy"]
    assert all(
        "n_participants" not in record
        for layer in frontier_diagnostics.values()
        for record in layer
    )

    gr_eegnet = next(
        item
        for item in analysis["cohort_descriptive"]
        if item["method_id"] == "braindecode-eegnet"
        and item["original_protocol"] == "GR"
    )
    par_eegnet = next(
        item
        for item in analysis["cohort_descriptive"]
        if item["method_id"] == "braindecode-eegnet"
        and item["original_protocol"] == "PAR"
    )
    assert gr_eegnet["complete_frontier_participants"] == [1]
    assert par_eegnet["complete_frontier_participants"] == [10]


def test_one_failed_eegnet_budget_invalidates_subject_primary_frontier_without_dropping_failure():
    plan = _small_plan()
    rows = _complete_rows(plan)
    target = next(
        row
        for row in rows
        if row["method_id"] == "braindecode-eegnet"
        and row["subject"] == 1
        and row["held_out_session"] == "1"
        and row["split_seed"] == 2026
        and row["model_seed"] == 11
        and row["calibration_per_class"] == 2
    )
    target["status"] = "oom"
    target["balanced_accuracy"] = None

    analysis = summarize_promoted_rows(rows, plan=plan)
    eegnet = _find_summary_method(analysis, "braindecode-eegnet")
    csp = _find_summary_method(analysis, "mne-csp-lda")
    pair = _find_pair(analysis, "mne-csp-lda", "braindecode-eegnet")
    failure = _find_failure_summary(analysis, "braindecode-eegnet")

    assert csp["complete_frontier_participants"] == [1, 10]
    assert eegnet["complete_frontier_participants"] == [10]
    assert eegnet["participant_frontier_auc"]["n_participants"] == 1
    assert pair["matched_complete_frontier_participants"] == [10]
    assert failure["failure_status_counts"] == {"oom": 1}

    incomplete = next(
        item
        for item in analysis["incomplete_realization_frontiers"]
        if item["method_id"] == "braindecode-eegnet"
        and item["subject"] == 1
        and item["held_out_session"] == "1"
        and item["split_seed"] == 2026
        and item["model_seed"] == 11
    )
    assert incomplete["missing_or_failed_budgets"] == [2]

    split_diagnostics = analysis["frontier_diagnostics"][
        "model_seed_averaged_split_frontier_auc"
    ]
    assert not any(
        item["method_id"] == "braindecode-eegnet"
        and item["subject"] == 1
        and item["held_out_session"] == "1"
        and item["split_seed"] == 2026
        for item in split_diagnostics
    )


def test_completely_missing_model_seed_realization_is_explicitly_incomplete():
    plan = _small_plan()
    rows = [
        row
        for row in _complete_rows(plan)
        if not (
            row["method_id"] == "braindecode-eegnet"
            and row["subject"] == 1
            and row["held_out_session"] == "1"
            and row["split_seed"] == 2026
            and row["model_seed"] == 22
        )
    ]

    analysis = summarize_promoted_rows(rows, plan=plan)
    eegnet = _find_summary_method(analysis, "braindecode-eegnet")
    assert eegnet["complete_frontier_participants"] == [10]

    missing = next(
        item
        for item in analysis["incomplete_realization_frontiers"]
        if item["method_id"] == "braindecode-eegnet"
        and item["subject"] == 1
        and item["held_out_session"] == "1"
        and item["split_seed"] == 2026
        and item["model_seed"] == 22
    )
    assert missing["missing_or_failed_budgets"] == [0, 1, 2]
