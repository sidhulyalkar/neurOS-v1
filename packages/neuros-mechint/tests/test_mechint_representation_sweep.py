from __future__ import annotations

import numpy as np
import pytest

from neuros_mechint.representations.contracts import (
    FitRegime,
    MethodOutcome,
    MethodStatus,
    RepresentationBenchmarkResult,
    RepresentationEmbedding,
)
from neuros_mechint.representations.sweep import (
    CaseMethodEvidence,
    RepresentationSweepResult,
    SweepCase,
    build_representation_sweep,
)


def _ok_outcome(method_id: str, value: float) -> MethodOutcome:
    embedding = RepresentationEmbedding(
        method_id=method_id,
        sequences=(np.column_stack((np.arange(5.0), np.arange(5.0) ** 2)),),
        sequence_ids=("eval",),
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
    )
    return MethodOutcome(
        method_id=method_id,
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
        status=MethodStatus.OK,
        embedding=embedding,
        metrics={"reference_pairwise_distance_rank": value},
    )


def _unavailable_outcome(method_id: str) -> MethodOutcome:
    return MethodOutcome(
        method_id=method_id,
        fit_regime=FitRegime.TRANSDUCTIVE_TARGET_OBSERVED,
        status=MethodStatus.UNAVAILABLE,
        error_type="OptionalDependencyUnavailable",
        error_message="optional dependency unavailable",
    )


def _result(pca_value: float) -> RepresentationBenchmarkResult:
    return RepresentationBenchmarkResult(
        train_sequence_ids=("train",),
        evaluation_sequence_ids=("eval",),
        outcomes=(
            _ok_outcome("pca", pca_value),
            _unavailable_outcome("tphate"),
        ),
    )


def test_sweep_preserves_cartesian_case_evidence_and_denominators() -> None:
    sweep = build_representation_sweep(
        (
            (SweepCase(0.0, 0), _result(0.8)),
            (SweepCase(0.5, 0), _result(0.6)),
            (SweepCase(0.0, 1), _result(0.7)),
            (SweepCase(0.5, 1), _result(0.5)),
        )
    )
    assert len(sweep.cases) == 4
    assert len(sweep.evidence) == 8
    summaries = {summary.method_id: summary for summary in sweep.summaries}
    assert summaries["pca"].declared_cases == 4
    assert summaries["pca"].ok_cases == 4
    assert summaries["pca"].failure_rate == 0.0
    assert summaries["pca"].metric_summaries["reference_pairwise_distance_rank"]["n"] == 4
    assert summaries["tphate"].declared_cases == 4
    assert summaries["tphate"].unavailable_cases == 4
    assert summaries["tphate"].failure_rate == 1.0
    assert summaries["tphate"].metric_summaries == {}
    assert "winner" not in sweep.to_dict()
    assert sweep.metadata["ranking_policy"] == "none"


def test_failed_case_evidence_cannot_carry_scientific_metrics() -> None:
    with pytest.raises(ValueError, match="cannot carry scientific metrics"):
        CaseMethodEvidence(
            case=SweepCase(0.5, 7),
            method_id="broken",
            fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
            status=MethodStatus.FAILED,
            metrics={"reference_pairwise_distance_rank": 0.9},
            error_type="RuntimeError",
            error_message="failed",
        )


def test_sweep_rejects_duplicate_case_ids() -> None:
    case = SweepCase(0.35, 7)
    with pytest.raises(ValueError, match="duplicate sweep cases"):
        build_representation_sweep(((case, _result(0.8)), (case, _result(0.7))))


def test_sweep_rejects_missing_method_in_one_case() -> None:
    partial = RepresentationBenchmarkResult(
        train_sequence_ids=("train",),
        evaluation_sequence_ids=("eval",),
        outcomes=(_ok_outcome("pca", 0.7),),
    )
    with pytest.raises(ValueError, match="same method ID set"):
        build_representation_sweep(
            ((SweepCase(0.0, 0), _result(0.8)), (SweepCase(0.5, 0), partial))
        )


def test_sweep_result_rejects_incomplete_cartesian_evidence() -> None:
    complete = build_representation_sweep(
        ((SweepCase(0.0, 0), _result(0.8)), (SweepCase(0.5, 0), _result(0.6)))
    )
    with pytest.raises(ValueError, match="Cartesian complete"):
        RepresentationSweepResult(
            cases=complete.cases,
            evidence=complete.evidence[:-1],
            summaries=complete.summaries,
        )


def test_sweep_case_validation_is_fail_closed() -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        SweepCase(float("nan"), 0)
    with pytest.raises(TypeError, match="seed must be an integer"):
        SweepCase(0.1, True)
