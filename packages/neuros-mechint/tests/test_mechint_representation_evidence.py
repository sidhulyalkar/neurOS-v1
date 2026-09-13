from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from neuros_mechint.representations import (
    CaseStatus,
    EvaluationScope,
    FitRegime,
    MethodEvidenceSummary,
    RepresentationCaseEvidence,
    RepresentationEvidenceGrid,
)
from neuros_mechint.representations.contracts import (
    FitRegime as ExecutionFitRegime,
    MethodStatus,
    RepresentationEmbedding,
)
from neuros_mechint.representations.sequence_authority import (
    SequenceMethodOutcome,
    SequenceRepresentationBenchmarkResult,
)


def _case(
    method: str,
    sequence: str,
    *,
    status: CaseStatus = CaseStatus.OK,
    metrics: dict[str, float | None] | None = None,
    error_type: str | None = None,
    error_message: str | None = None,
    metadata: dict[str, object] | None = None,
    operational: dict[str, float] | None = None,
) -> RepresentationCaseEvidence:
    return RepresentationCaseEvidence(
        method_id=method,
        sequence_id=sequence,
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
        evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
        status=status,
        metrics=metrics,
        error_type=error_type,
        error_message=error_message,
        metadata=metadata,
        operational=operational,
    )


def _grid(cases: tuple[RepresentationCaseEvidence, ...]) -> RepresentationEvidenceGrid:
    return RepresentationEvidenceGrid(
        train_sequence_ids=("train-b", "train-a"),
        evaluation_sequence_ids=("eval-1", "eval-2"),
        method_ids=("pca", "tphate"),
        cases=cases,
        metadata={"study": {"phase": 1, "tags": ["geometry", "controlled"]}},
    )


def _sequence_result(runtime: float) -> SequenceRepresentationBenchmarkResult:
    embedding = RepresentationEmbedding(
        method_id="pca",
        sequences=(np.asarray([[0.0], [0.5], [1.0]], dtype=float),),
        sequence_ids=("eval-1",),
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
    )
    return SequenceRepresentationBenchmarkResult(
        method_ids=("pca",),
        train_sequence_ids=("train-1",),
        evaluation_sequence_ids=("eval-1", "eval-2"),
        outcomes=(
            SequenceMethodOutcome(
                method_id="pca",
                sequence_id="eval-1",
                fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
                status=MethodStatus.OK,
                embedding=embedding,
                metrics={"trustworthiness": 0.8},
                metadata={
                    "execution_scope": "native_per_sequence",
                    "runtime_seconds": runtime,
                    "runtime_domain": "operational",
                },
            ),
            SequenceMethodOutcome(
                method_id="pca",
                sequence_id="eval-2",
                fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
                status=MethodStatus.FAILED,
                error_type="RuntimeError",
                error_message="controlled failure",
                metadata={
                    "execution_scope": "native_per_sequence",
                    "runtime_seconds": runtime + 1.0,
                    "runtime_domain": "operational",
                },
            ),
        ),
        metadata={
            "ranking_policy": "none",
            "claim_scope": "representation_geometry_sequence_level",
        },
    )


def test_evidence_reuses_execution_fit_regime_contract() -> None:
    assert FitRegime is ExecutionFitRegime


def test_grid_preserves_failures_and_metric_denominators() -> None:
    grid = _grid(
        (
            _case("pca", "eval-1", metrics={"trustworthiness": 0.8, "continuity": 0.7}),
            _case("pca", "eval-2", metrics={"trustworthiness": None, "continuity": 0.5}),
            _case("tphate", "eval-1", metrics={"trustworthiness": 0.9}),
            _case(
                "tphate",
                "eval-2",
                status=CaseStatus.NONCONVERGED,
                error_type="RepresentationNonconvergenceError",
                error_message="optimizer did not converge",
            ),
        )
    )

    pca = grid.summary_for_method("pca")
    assert pca.total_cases == 2
    assert pca.ok_cases == 2
    assert pca.metric_n == {"continuity": 2, "trustworthiness": 1}
    assert pca.metric_mean["continuity"] == pytest.approx(0.6)
    assert pca.metric_mean["trustworthiness"] == pytest.approx(0.8)

    tphate = grid.summary_for_method("tphate")
    assert tphate.ok_cases == 1
    assert tphate.nonconverged_cases == 1
    assert tphate.non_ok_rate == pytest.approx(0.5)
    assert tphate.nonconverged_rate == pytest.approx(0.5)
    assert not hasattr(tphate, "failure_rate")


def test_metric_schema_can_be_sparse_but_denominator_stays_explicit() -> None:
    grid = _grid(
        (
            _case("pca", "eval-1", metrics={"a": 1.0}),
            _case("pca", "eval-2", metrics={"b": 3.0}),
            _case("tphate", "eval-1", metrics={}),
            _case("tphate", "eval-2", metrics={}),
        )
    )
    summary = grid.summary_for_method("pca")
    assert summary.metric_mean == {"a": 1.0, "b": 3.0}
    assert summary.metric_n == {"a": 1, "b": 1}


def test_grid_requires_exact_cartesian_evidence() -> None:
    with pytest.raises(ValueError, match="exact declared"):
        _grid(
            (
                _case("pca", "eval-1", metrics={}),
                _case("pca", "eval-2", metrics={}),
                _case("tphate", "eval-1", metrics={}),
            )
        )


def test_grid_rejects_duplicate_case() -> None:
    duplicate = _case("pca", "eval-1", metrics={})
    with pytest.raises(ValueError, match="duplicate representation case"):
        _grid(
            (
                duplicate,
                duplicate,
                _case("pca", "eval-2", metrics={}),
                _case("tphate", "eval-1", metrics={}),
                _case("tphate", "eval-2", metrics={}),
            )
        )


def test_non_success_case_requires_error_and_forbids_metrics() -> None:
    with pytest.raises(ValueError, match="error_type"):
        _case("pca", "eval-1", status=CaseStatus.FAILED)

    with pytest.raises(ValueError, match="cannot carry scientific metric"):
        _case(
            "pca",
            "eval-1",
            status=CaseStatus.UNAVAILABLE,
            metrics={"trustworthiness": 0.1},
            error_type="MissingBackend",
            error_message="optional backend unavailable",
        )


def test_method_fit_regime_and_scope_cannot_drift_across_cases() -> None:
    changed = RepresentationCaseEvidence(
        method_id="pca",
        sequence_id="eval-2",
        fit_regime=FitRegime.TRANSDUCTIVE_TARGET_OBSERVED,
        evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
        status=CaseStatus.OK,
        metrics={},
    )
    with pytest.raises(ValueError, match="fit_regime"):
        _grid(
            (
                _case("pca", "eval-1", metrics={}),
                changed,
                _case("tphate", "eval-1", metrics={}),
                _case("tphate", "eval-2", metrics={}),
            )
        )


def test_identity_is_canonical_but_status_sensitive() -> None:
    cases = (
        _case("pca", "eval-1", metrics={"m": -0.0}, metadata={"b": 2, "a": 1}),
        _case("pca", "eval-2", metrics={"m": 0.5}),
        _case("tphate", "eval-1", metrics={"m": 0.4}),
        _case("tphate", "eval-2", metrics={"m": 0.3}),
    )
    first = _grid(cases)
    second = RepresentationEvidenceGrid(
        train_sequence_ids=("train-a", "train-b"),
        evaluation_sequence_ids=("eval-2", "eval-1"),
        method_ids=("tphate", "pca"),
        cases=tuple(reversed(cases)),
        metadata={"study": {"tags": ["geometry", "controlled"], "phase": 1}},
    )
    assert first.evidence_sha256 == second.evidence_sha256

    changed_cases = cases[:-1] + (
        _case(
            "tphate",
            "eval-2",
            status=CaseStatus.FAILED,
            error_type="RuntimeError",
            error_message="controlled failure",
        ),
    )
    assert first.evidence_sha256 != _grid(changed_cases).evidence_sha256


def test_operational_telemetry_is_preserved_but_not_scientific_identity() -> None:
    first = RepresentationEvidenceGrid.from_sequence_benchmark(_sequence_result(0.1))
    second = RepresentationEvidenceGrid.from_sequence_benchmark(_sequence_result(9.9))

    assert first.evidence_sha256 == second.evidence_sha256
    assert first.cases[0].operational["runtime_seconds"] == pytest.approx(0.1)
    assert second.cases[0].operational["runtime_seconds"] == pytest.approx(9.9)
    assert "operational" not in first.cases[0].to_manifest()
    assert first.to_record()["cases"][0]["operational"] == {"runtime_seconds": 0.1}


def test_sequence_adapter_preserves_status_scope_and_fails_on_unknown_metadata() -> None:
    grid = RepresentationEvidenceGrid.from_sequence_benchmark(_sequence_result(0.1))
    assert [case.status for case in grid.cases] == [CaseStatus.OK, CaseStatus.FAILED]
    assert {case.evaluation_scope for case in grid.cases} == {EvaluationScope.SEQUENCE_LOCAL}
    assert grid.summary_for_method("pca").failed_rate == pytest.approx(0.5)

    result = _sequence_result(0.1)
    bad = SequenceMethodOutcome(
        method_id="pca",
        sequence_id="eval-2",
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
        status=MethodStatus.FAILED,
        error_type="RuntimeError",
        error_message="controlled failure",
        metadata={"execution_scope": "native_per_sequence", "mystery": 1},
    )
    mutated = SequenceRepresentationBenchmarkResult(
        method_ids=result.method_ids,
        train_sequence_ids=result.train_sequence_ids,
        evaluation_sequence_ids=result.evaluation_sequence_ids,
        outcomes=(result.outcomes[0], bad),
        metadata=result.metadata,
    )
    with pytest.raises(ValueError, match="unclassified"):
        RepresentationEvidenceGrid.from_sequence_benchmark(mutated)


def test_metadata_is_deeply_frozen_and_unordered_sets_fail_closed() -> None:
    evidence = _case(
        "pca",
        "eval-1",
        metrics={},
        metadata={"nested": {"values": [1, 2]}},
    )
    with pytest.raises(TypeError):
        evidence.metadata["new"] = 3  # type: ignore[index]
    nested = evidence.metadata["nested"]
    with pytest.raises(TypeError):
        nested["new"] = 4  # type: ignore[index]
    assert nested["values"] == (1, 2)

    with pytest.raises(TypeError, match="unordered sets"):
        _case("pca", "eval-1", metrics={}, metadata={"bad": {1, 2}})


def test_summary_direct_construction_fails_closed() -> None:
    kwargs = dict(
        method_id="pca",
        fit_regime=FitRegime.TRAIN_ONLY_INDUCTIVE,
        evaluation_scope=EvaluationScope.SEQUENCE_LOCAL,
        total_cases=2,
        ok_cases=1,
        failed_cases=1,
        unavailable_cases=0,
        nonconverged_cases=0,
        metric_mean={"m": 0.5},
        metric_n={"m": 1},
    )
    summary = MethodEvidenceSummary(**kwargs)
    assert summary.metric_mean == {"m": 0.5}

    with pytest.raises(ValueError, match="sum exactly"):
        MethodEvidenceSummary(**{**kwargs, "failed_cases": 0})
    with pytest.raises(ValueError, match="exactly match"):
        MethodEvidenceSummary(**{**kwargs, "metric_n": {}})
    with pytest.raises(ValueError, match="metric_n=0"):
        MethodEvidenceSummary(**{**kwargs, "metric_n": {"m": 0}})


def test_contracts_are_frozen() -> None:
    evidence = _case("pca", "eval-1", metrics={})
    with pytest.raises(FrozenInstanceError):
        evidence.status = CaseStatus.FAILED  # type: ignore[misc]


def test_metric_values_fail_closed_on_non_finite_boolean_and_negative_telemetry() -> None:
    with pytest.raises(TypeError, match="finite real"):
        _case("pca", "eval-1", metrics={"m": True})
    with pytest.raises(ValueError, match="finite real"):
        _case("pca", "eval-1", metrics={"m": float("nan")})
    with pytest.raises(ValueError, match="finite real"):
        _case("pca", "eval-1", metrics={"m": float("inf")})
    with pytest.raises(ValueError, match="nonnegative"):
        _case("pca", "eval-1", metrics={}, operational={"runtime_seconds": -1.0})
