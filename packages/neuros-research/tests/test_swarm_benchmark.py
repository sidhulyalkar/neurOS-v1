from __future__ import annotations

import inspect

import pytest
from neuros.research import swarm_benchmark_cases
from neuros.research.swarm import AgentReview, CouncilRun, Finding
from neuros.research.swarm_benchmark import (
    _GROUND_TRUTH,
    compare_benchmark_reports,
    score_benchmark,
)
from neuros.research.swarm_benchmark import (
    BENCHMARK_CORPUS_SHA256 as SCORER_CORPUS_SHA256,
)
from neuros.research.swarm_benchmark_cases import (
    BENCHMARK_CORPUS_SHA256,
    DEFECT_TAXONOMY,
    benchmark_case,
    benchmark_cases,
    benchmark_public_manifest,
    build_benchmark_task,
)

REV = "4" * 64
REPO = "sidhulyalkar/neurOS-v1"


def _finding(defect_id: str) -> Finding:
    return Finding(
        finding_id=defect_id,
        severity="high",
        category="reproducibility",
        claim="classified defect",
        evidence="benchmark stimulus",
        falsification_test="apply deterministic regression",
        proposed_repair="repair the defect",
        confidence=0.9,
        requires_human_judgment=False,
        reference="benchmark",
    )


def _run(case_id: str, by_member: dict[str, set[str]], failed=()) -> CouncilRun:
    task = build_benchmark_task(case_id, repository=REPO, source_revision=REV)
    reviews = []
    for index, (member, ids) in enumerate(sorted(by_member.items())):
        reviews.append(
            AgentReview(
                task_sha256=task.sha256,
                member_id=member,
                role="reviewer",
                model=f"model-{member}",
                prompt_sha256=f"{index + 1:064x}",
                response_sha256=f"{index + 101:064x}",
                findings=tuple(_finding(defect_id) for defect_id in sorted(ids)),
            )
        )
    return CouncilRun(task.sha256, tuple(reviews), tuple(failed))


def _perfect_runs(member="solo"):
    return {
        case.case_id: _run(case.case_id, {member: set(_GROUND_TRUTH[case.case_id])})
        for case in benchmark_cases()
    }


def test_taxonomy_has_exactly_twenty_unique_closed_labels():
    ids = [item.defect_id for item in DEFECT_TAXONOMY]
    assert len(ids) == 20
    assert len(set(ids)) == 20
    assert ids == [f"D{index:02d}" for index in range(1, 21)]


def test_corpus_has_29_opaque_unique_cases():
    cases = benchmark_cases()
    assert len(cases) == 29
    assert len({case.case_id for case in cases}) == 29
    assert [case.case_id for case in cases] == [f"case-{index:03d}" for index in range(1, 30)]


def test_case_ids_do_not_leak_taxonomy_or_diagnostic_words():
    forbidden = {
        "leak", "seed", "retry", "duplicate", "stale", "shadow", "environment",
        "provider", "participant", "artifact", "claim", "preprocess", "shard",
    }
    for case in benchmark_cases():
        lowered = case.case_id.lower()
        assert not any(token in lowered for token in forbidden)


def test_corpus_contains_five_clean_controls_and_three_compound_cases():
    labels = list(_GROUND_TRUTH.values())
    assert sum(not value for value in labels) == 5
    assert sum(len(value) > 1 for value in labels) == 3


def test_every_defect_class_is_represented_in_ground_truth():
    represented = frozenset().union(*_GROUND_TRUTH.values())
    assert represented == {item.defect_id for item in DEFECT_TAXONOMY}


def test_public_corpus_commitment_matches_scorer_answer_key():
    assert BENCHMARK_CORPUS_SHA256 == SCORER_CORPUS_SHA256


def test_public_module_contains_no_answer_key_or_scorer_import():
    source = inspect.getsource(swarm_benchmark_cases)
    assert "_GROUND_TRUTH" not in source
    assert "from .swarm_benchmark import" not in source


def test_public_manifest_contains_no_ground_truth_fields():
    manifest = benchmark_public_manifest()
    assert manifest["ground_truth_included"] is False
    text = repr(manifest).lower()
    assert "expected_defect_ids" not in text
    assert manifest["corpus_sha256"] == BENCHMARK_CORPUS_SHA256


def test_public_task_contains_taxonomy_but_not_case_labels():
    task = build_benchmark_task("case-001", repository=REPO, source_revision=REV)
    payload = task.to_dict()
    context = payload["public_context"]
    assert context["case"]["case_id"] == "case-001"
    assert len(context["defect_taxonomy"]) == 20
    assert "expected_defect_ids" not in repr(context)
    assert "D01" not in context["case"]["case_id"]


def test_task_identity_binds_case_corpus_and_revision():
    one = build_benchmark_task("case-001", repository=REPO, source_revision=REV)
    two = build_benchmark_task("case-002", repository=REPO, source_revision=REV)
    other_rev = build_benchmark_task("case-001", repository=REPO, source_revision="5" * 64)
    assert one.sha256 != two.sha256
    assert one.sha256 != other_rev.sha256
    assert one.to_dict()["public_context"]["corpus_sha256"] == BENCHMARK_CORPUS_SHA256


def test_case_payload_substitution_is_rejected():
    from neuros.research.swarm_benchmark_cases import BenchmarkCase

    forged = BenchmarkCase("case-001", "clean and unrelated")
    with pytest.raises(ValueError, match="frozen corpus"):
        build_benchmark_task(forged, repository=REPO, source_revision=REV)


def test_perfect_predictions_score_one_everywhere():
    report = score_benchmark(_perfect_runs(), repository=REPO, source_revision=REV)
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0
    assert report.exact_case_accuracy == 1.0
    assert report.clean_control_false_positive_rate == 0.0
    assert report.false_positives == 0
    assert report.false_negatives == 0


def test_unknown_finding_id_invalidates_score():
    runs = {"case-001": _run("case-001", {"a": {"NOT_A_LABEL"}})}
    with pytest.raises(ValueError, match="unknown benchmark defect IDs"):
        score_benchmark(runs, repository=REPO, source_revision=REV)


def test_swapped_run_to_case_binding_is_rejected():
    run = _run("case-002", {"a": {"D02"}})
    with pytest.raises(ValueError, match="not bound"):
        score_benchmark({"case-001": run}, repository=REPO, source_revision=REV)


def test_false_alarm_on_clean_control_reduces_precision_and_clean_score():
    runs = {"case-021": _run("case-021", {"a": {"D01"}})}
    report = score_benchmark(runs, repository=REPO, source_revision=REV)
    assert report.true_positives == 0
    assert report.false_positives == 1
    assert report.precision == 0.0
    assert report.clean_control_false_positive_rate == 1.0
    assert report.exact_case_accuracy == 0.0


def test_missing_defect_reduces_recall():
    runs = {"case-027": _run("case-027", {"a": {"D01"}})}
    report = score_benchmark(runs, repository=REPO, source_revision=REV)
    assert report.true_positives == 1
    assert report.false_negatives == 1
    assert report.recall == 0.5


def test_member_scores_show_when_one_member_carries_the_council():
    runs = {
        "case-027": _run("case-027", {"ultra": {"D01", "D05"}, "light": set()}),
        "case-021": _run("case-021", {"ultra": set(), "light": {"D03"}}),
    }
    report = score_benchmark(runs, repository=REPO, source_revision=REV)
    scores = {score.member_id: score for score in report.member_scores}
    assert scores["ultra"].true_positives == 2
    assert scores["ultra"].false_positives == 0
    assert scores["light"].true_positives == 0
    assert scores["light"].false_positives == 1


def test_unique_valid_findings_are_attributed_to_the_only_supporter():
    runs = {
        "case-027": _run("case-027", {"a": {"D01"}, "b": {"D05"}, "c": set()})
    }
    report = score_benchmark(runs, repository=REPO, source_revision=REV)
    scores = {score.member_id: score.unique_valid_findings for score in report.member_scores}
    assert scores == {"a": 1, "b": 1, "c": 0}


def test_pairwise_disagreement_is_zero_for_identical_reviewers():
    run = _run("case-001", {"a": {"D01"}, "b": {"D01"}})
    report = score_benchmark({"case-001": run}, repository=REPO, source_revision=REV)
    assert report.mean_pairwise_disagreement == 0.0


def test_pairwise_disagreement_is_one_for_disjoint_nonempty_reviews():
    run = _run("case-027", {"a": {"D01"}, "b": {"D05"}})
    report = score_benchmark({"case-027": run}, repository=REPO, source_revision=REV)
    assert report.mean_pairwise_disagreement == 1.0


def test_failed_member_calls_are_counted_without_provider_diagnostics():
    run = _run("case-001", {"a": {"D01"}}, failed=("b:TimeoutError",))
    report = score_benchmark({"case-001": run}, repository=REPO, source_revision=REV)
    scores = {score.member_id: score for score in report.member_scores}
    assert report.failed_member_calls == 1
    assert scores["b"].failed_cases == 1
    assert scores["b"].reviewed_cases == 0


def test_report_identity_is_order_independent_over_mapping_input():
    first = {
        "case-002": _run("case-002", {"a": {"D02"}}),
        "case-001": _run("case-001", {"a": {"D01"}}),
    }
    second = dict(reversed(list(first.items())))
    one = score_benchmark(first, repository=REPO, source_revision=REV)
    two = score_benchmark(second, repository=REPO, source_revision=REV)
    assert one.sha256 == two.sha256


def test_comparison_reports_metric_deltas_and_denies_authority():
    baseline = score_benchmark(
        {"case-027": _run("case-027", {"a": {"D01"}})},
        repository=REPO,
        source_revision=REV,
    )
    candidate = score_benchmark(
        {"case-027": _run("case-027", {"a": {"D01", "D05"}})},
        repository=REPO,
        source_revision=REV,
    )
    comparison = compare_benchmark_reports(baseline, candidate)
    assert comparison["recall_delta"] == 0.5
    assert comparison["f1_delta"] > 0
    assert comparison["comparison_is_scientific_authority"] is False
    assert len(comparison["comparison_sha256"]) == 64


def test_comparison_rejects_different_case_slices():
    baseline = score_benchmark(
        {"case-001": _run("case-001", {"a": {"D01"}})},
        repository=REPO,
        source_revision=REV,
    )
    candidate = score_benchmark(
        {"case-002": _run("case-002", {"a": {"D02"}})},
        repository=REPO,
        source_revision=REV,
    )
    with pytest.raises(ValueError, match="different case sets"):
        compare_benchmark_reports(baseline, candidate)


def test_report_manifest_explicitly_denies_authority():
    report = score_benchmark(
        {"case-001": _run("case-001", {"a": {"D01"}})},
        repository=REPO,
        source_revision=REV,
    ).to_dict()
    assert report["benchmark_output_is_scientific_authority"] is False
    assert report["merge_authority"] is False
    assert report["provider_execution_authority"] is False
    assert report["scientific_promotion_authority"] is False


def test_unknown_case_fails_closed():
    with pytest.raises(ValueError, match="unknown benchmark case"):
        benchmark_case("case-999")


def test_scorer_rejects_successful_member_also_marked_failed():
    run = _run("case-001", {"a": {"D01"}}, failed=("a:TimeoutError",))
    with pytest.raises(ValueError, match="successful reviewer as failed"):
        score_benchmark({"case-001": run}, repository=REPO, source_revision=REV)


def test_scorer_rejects_repeated_failed_member_identity():
    run = _run(
        "case-001",
        {},
        failed=("a:TimeoutError", "a:RuntimeError"),
    )
    with pytest.raises(ValueError, match="repeats a failed member identity"):
        score_benchmark({"case-001": run}, repository=REPO, source_revision=REV)
