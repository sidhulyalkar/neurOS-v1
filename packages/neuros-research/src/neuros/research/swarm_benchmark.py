"""Scorer-side deterministic evaluation for the neurOS scientific-engineering swarm."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from itertools import combinations
from typing import Any

from ._canonical import canonical_sha256, require_nonempty, require_sha256
from .swarm import CouncilRun, SealedSwarmTask
from .swarm_benchmark_cases import (
    BENCHMARK_SCHEMA,
    DEFECT_IDS,
    DEFECT_TAXONOMY,
    BenchmarkCase,
    benchmark_case,
    benchmark_cases,
)

REPORT_SCHEMA = "neuros.scientific_engineering_swarm_benchmark_report.v1"

# Scorer-side answer key. This mapping is never included in reviewer-visible task payloads.
_GROUND_TRUTH: dict[str, frozenset[str]] = {
    "case-001": frozenset({"D01"}),
    "case-002": frozenset({"D02"}),
    "case-003": frozenset({"D03"}),
    "case-004": frozenset({"D04"}),
    "case-005": frozenset({"D05"}),
    "case-006": frozenset({"D06"}),
    "case-007": frozenset({"D07"}),
    "case-008": frozenset({"D08"}),
    "case-009": frozenset({"D09"}),
    "case-010": frozenset({"D10"}),
    "case-011": frozenset({"D11"}),
    "case-012": frozenset({"D12"}),
    "case-013": frozenset({"D13"}),
    "case-014": frozenset({"D14"}),
    "case-015": frozenset({"D15"}),
    "case-016": frozenset({"D16"}),
    "case-017": frozenset({"D17"}),
    "case-018": frozenset({"D18"}),
    "case-019": frozenset({"D19"}),
    "case-020": frozenset({"D20"}),
    "case-021": frozenset(),
    "case-022": frozenset(),
    "case-023": frozenset(),
    "case-024": frozenset(),
    "case-025": frozenset(),
    "case-026": frozenset({"D07"}),
    "case-027": frozenset({"D01", "D05"}),
    "case-028": frozenset({"D09", "D10"}),
    "case-029": frozenset({"D13", "D14"}),
}


def _validate_corpus() -> None:
    cases = benchmark_cases()
    case_ids = tuple(case.case_id for case in cases)
    expected_case_ids = tuple(f"case-{index:03d}" for index in range(1, 30))
    if case_ids != expected_case_ids:
        raise RuntimeError("benchmark cases must use the frozen opaque case-001..case-029 sequence")
    if set(_GROUND_TRUTH) != set(case_ids):
        raise RuntimeError("benchmark ground truth must cover every case exactly once")
    if len(DEFECT_TAXONOMY) != 20 or len(DEFECT_IDS) != 20:
        raise RuntimeError("benchmark taxonomy must contain exactly 20 unique defect IDs")
    unknown_labels = sorted(set().union(*_GROUND_TRUTH.values()) - DEFECT_IDS)
    if unknown_labels:
        raise RuntimeError(f"benchmark ground truth contains unknown labels: {unknown_labels}")
    represented = set().union(*_GROUND_TRUTH.values())
    if represented != DEFECT_IDS:
        raise RuntimeError("every benchmark defect class must be represented by at least one case")
    if sum(not labels for labels in _GROUND_TRUTH.values()) != 5:
        raise RuntimeError("benchmark must contain exactly five clean controls")
    if sum(len(labels) > 1 for labels in _GROUND_TRUTH.values()) != 3:
        raise RuntimeError("benchmark must contain exactly three compound cases")


_validate_corpus()


def _corpus_payload() -> dict[str, Any]:
    return {
        "schema": BENCHMARK_SCHEMA,
        "taxonomy": [defect.to_dict() for defect in DEFECT_TAXONOMY],
        "cases": [
            {
                **case.to_public_dict(),
                "expected_defect_ids": sorted(_GROUND_TRUTH[case.case_id]),
            }
            for case in benchmark_cases()
        ],
    }


BENCHMARK_CORPUS_SHA256 = canonical_sha256(_corpus_payload())


def benchmark_public_manifest() -> dict[str, Any]:
    """Return public corpus metadata without any expected labels."""
    cases = benchmark_cases()
    return {
        "schema": BENCHMARK_SCHEMA,
        "corpus_sha256": BENCHMARK_CORPUS_SHA256,
        "case_count": len(cases),
        "taxonomy": [defect.to_dict() for defect in DEFECT_TAXONOMY],
        "cases": [case.to_public_dict() for case in cases],
        "ground_truth_included": False,
    }


def build_benchmark_task(
    case: BenchmarkCase | str,
    *,
    repository: str,
    source_revision: str,
) -> SealedSwarmTask:
    """Build the exact reviewer-visible task for one benchmark case."""
    if isinstance(case, str):
        selected = benchmark_case(case)
    else:
        selected = benchmark_case(case.case_id)
        if selected != case:
            raise ValueError("benchmark case payload does not match the frozen corpus")
    revision = require_sha256(source_revision, name="source_revision")
    repo = require_nonempty(repository, name="repository")
    return SealedSwarmTask(
        repository=repo,
        source_revision=revision,
        objective=(
            "Identify every defect present in the stimulus. Emit zero findings for a clean case. "
            "Each finding_id must be one exact defect_id from the supplied taxonomy."
        ),
        claim_boundary=(
            "Benchmark classification only. Model output cannot authorize code, provider execution, "
            "scientific promotion, Kumar2024 execution, or ORION comparison."
        ),
        forbidden_actions=(
            "merge code",
            "authorize provider execution",
            "promote scientific claims",
            "change benchmark ground truth",
        ),
        public_context={
            "benchmark_schema": BENCHMARK_SCHEMA,
            "corpus_sha256": BENCHMARK_CORPUS_SHA256,
            "case": selected.to_public_dict(),
            "defect_taxonomy": [defect.to_dict() for defect in DEFECT_TAXONOMY],
            "scoring_rule": (
                "Report only present defects. Unknown finding IDs are invalid. False positives on "
                "clean controls reduce precision."
            ),
        },
    )


@dataclass(frozen=True, slots=True)
class MemberBenchmarkScore:
    member_id: str
    true_positives: int
    false_positives: int
    false_negatives: int
    reviewed_cases: int
    failed_cases: int
    unique_valid_findings: int

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "member_id": self.member_id,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "reviewed_cases": self.reviewed_cases,
            "failed_cases": self.failed_cases,
            "unique_valid_findings": self.unique_valid_findings,
        }


@dataclass(frozen=True, slots=True)
class BenchmarkReport:
    corpus_sha256: str
    repository: str
    source_revision: str
    evaluated_case_ids: tuple[str, ...]
    true_positives: int
    false_positives: int
    false_negatives: int
    exact_case_matches: int
    clean_cases: int
    clean_cases_with_false_positive: int
    failed_member_calls: int
    mean_pairwise_disagreement: float
    member_scores: tuple[MemberBenchmarkScore, ...]
    run_sha256s: tuple[tuple[str, str], ...]

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else 1.0

    @property
    def f1(self) -> float:
        denominator = self.precision + self.recall
        return 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def exact_case_accuracy(self) -> float:
        return self.exact_case_matches / len(self.evaluated_case_ids)

    @property
    def clean_control_false_positive_rate(self) -> float:
        if not self.clean_cases:
            return 0.0
        return self.clean_cases_with_false_positive / self.clean_cases

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": REPORT_SCHEMA,
            "corpus_sha256": self.corpus_sha256,
            "repository": self.repository,
            "source_revision": self.source_revision,
            "evaluated_case_ids": list(self.evaluated_case_ids),
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "exact_case_matches": self.exact_case_matches,
            "exact_case_accuracy": self.exact_case_accuracy,
            "clean_cases": self.clean_cases,
            "clean_cases_with_false_positive": self.clean_cases_with_false_positive,
            "clean_control_false_positive_rate": self.clean_control_false_positive_rate,
            "failed_member_calls": self.failed_member_calls,
            "mean_pairwise_disagreement": self.mean_pairwise_disagreement,
            "member_scores": [score.to_dict() for score in self.member_scores],
            "run_sha256s": [list(item) for item in self.run_sha256s],
            "benchmark_output_is_scientific_authority": False,
            "merge_authority": False,
            "provider_execution_authority": False,
            "scientific_promotion_authority": False,
        }
        payload["report_sha256"] = canonical_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return self.to_dict()["report_sha256"]


def _finding_ids(run: CouncilRun) -> dict[str, frozenset[str]]:
    by_member: dict[str, frozenset[str]] = {}
    for review in run.reviews:
        ids = frozenset(finding.finding_id for finding in review.findings)
        unknown = sorted(ids - DEFECT_IDS)
        if unknown:
            raise ValueError(f"review contains unknown benchmark defect IDs: {unknown}")
        by_member[review.member_id] = ids
    return by_member


def _jaccard_distance(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    if not union:
        return 0.0
    return 1.0 - len(left & right) / len(union)


def score_benchmark(
    runs: Mapping[str, CouncilRun],
    *,
    repository: str,
    source_revision: str,
) -> BenchmarkReport:
    """Score one run per opaque case against scorer-side deterministic ground truth."""
    if not runs:
        raise ValueError("benchmark requires at least one case run")
    repo = require_nonempty(repository, name="repository")
    revision = require_sha256(source_revision, name="source_revision")
    case_ids = tuple(sorted(require_nonempty(case_id, name="case_id") for case_id in runs))
    known_case_ids = {case.case_id for case in benchmark_cases()}
    unknown_cases = sorted(set(case_ids) - known_case_ids)
    if unknown_cases:
        raise ValueError(f"unknown benchmark cases: {unknown_cases}")

    tp = fp = fn = exact = clean_cases = clean_fp = failed_calls = 0
    member_ids: set[str] = set()
    predictions_by_case: dict[str, dict[str, frozenset[str]]] = {}
    failed_by_case: dict[str, set[str]] = {}
    run_ids: list[tuple[str, str]] = []

    for case_id in case_ids:
        run = runs[case_id]
        expected_task = build_benchmark_task(
            case_id,
            repository=repo,
            source_revision=revision,
        )
        if run.task_sha256 != expected_task.sha256:
            raise ValueError(f"run for {case_id} is not bound to the expected benchmark task")
        per_member = _finding_ids(run)
        predictions_by_case[case_id] = per_member
        member_ids.update(per_member)
        failed_members = {value.split(":", 1)[0] for value in run.failed_members}
        if len(failed_members) != len(run.failed_members):
            raise ValueError(f"run for {case_id} repeats a failed member identity")
        if failed_members & set(per_member):
            raise ValueError(f"run for {case_id} marks a successful reviewer as failed")
        failed_by_case[case_id] = failed_members
        member_ids.update(failed_members)
        failed_calls += len(run.failed_members)
        run_ids.append((case_id, run.sha256))

        predicted = frozenset().union(*per_member.values()) if per_member else frozenset()
        expected = _GROUND_TRUTH[case_id]
        tp += len(predicted & expected)
        fp += len(predicted - expected)
        fn += len(expected - predicted)
        exact += int(predicted == expected)
        if not expected:
            clean_cases += 1
            clean_fp += int(bool(predicted))

    member_scores: list[MemberBenchmarkScore] = []
    unique_counts = {member_id: 0 for member_id in member_ids}
    disagreement_values: list[float] = []

    for case_id in case_ids:
        per_member = predictions_by_case[case_id]
        expected = _GROUND_TRUTH[case_id]
        for defect_id in expected:
            supporters = [
                member_id
                for member_id, predicted in per_member.items()
                if defect_id in predicted
            ]
            if len(supporters) == 1:
                unique_counts[supporters[0]] += 1
        for left, right in combinations(sorted(per_member), 2):
            disagreement_values.append(_jaccard_distance(per_member[left], per_member[right]))

    for member_id in sorted(member_ids):
        member_tp = member_fp = member_fn = reviewed = failed = 0
        for case_id in case_ids:
            predicted = predictions_by_case[case_id].get(member_id, frozenset())
            expected = _GROUND_TRUTH[case_id]
            member_tp += len(predicted & expected)
            member_fp += len(predicted - expected)
            member_fn += len(expected - predicted)
            reviewed += int(member_id in predictions_by_case[case_id])
            failed += int(member_id in failed_by_case[case_id])
        member_scores.append(
            MemberBenchmarkScore(
                member_id=member_id,
                true_positives=member_tp,
                false_positives=member_fp,
                false_negatives=member_fn,
                reviewed_cases=reviewed,
                failed_cases=failed,
                unique_valid_findings=unique_counts[member_id],
            )
        )

    mean_disagreement = (
        sum(disagreement_values) / len(disagreement_values)
        if disagreement_values
        else 0.0
    )
    return BenchmarkReport(
        corpus_sha256=BENCHMARK_CORPUS_SHA256,
        repository=repo,
        source_revision=revision,
        evaluated_case_ids=case_ids,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        exact_case_matches=exact,
        clean_cases=clean_cases,
        clean_cases_with_false_positive=clean_fp,
        failed_member_calls=failed_calls,
        mean_pairwise_disagreement=mean_disagreement,
        member_scores=tuple(member_scores),
        run_sha256s=tuple(run_ids),
    )


def compare_benchmark_reports(
    baseline: BenchmarkReport,
    candidate: BenchmarkReport,
) -> dict[str, Any]:
    """Return deterministic deltas for an identical benchmark slice."""
    if baseline.corpus_sha256 != candidate.corpus_sha256:
        raise ValueError("benchmark reports bind different corpora")
    if baseline.evaluated_case_ids != candidate.evaluated_case_ids:
        raise ValueError("benchmark reports evaluate different case sets")
    if (
        baseline.repository != candidate.repository
        or baseline.source_revision != candidate.source_revision
    ):
        raise ValueError("benchmark reports evaluate different source identities")
    payload = {
        "schema": "neuros.scientific_engineering_swarm_benchmark_comparison.v1",
        "corpus_sha256": baseline.corpus_sha256,
        "baseline_report_sha256": baseline.sha256,
        "candidate_report_sha256": candidate.sha256,
        "precision_delta": candidate.precision - baseline.precision,
        "recall_delta": candidate.recall - baseline.recall,
        "f1_delta": candidate.f1 - baseline.f1,
        "exact_case_accuracy_delta": candidate.exact_case_accuracy - baseline.exact_case_accuracy,
        "clean_control_false_positive_rate_delta": (
            candidate.clean_control_false_positive_rate
            - baseline.clean_control_false_positive_rate
        ),
        "comparison_is_scientific_authority": False,
    }
    payload["comparison_sha256"] = canonical_sha256(payload)
    return payload
