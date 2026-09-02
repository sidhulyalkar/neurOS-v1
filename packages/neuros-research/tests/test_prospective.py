from __future__ import annotations

import copy

import pytest
from neuros.research._canonical import canonical_sha256
from neuros.research.prospective import (
    ProspectiveGeometryCandidate,
    ProspectiveGeometryPlan,
    ProspectiveGeometryReveal,
    ProspectiveOutcome,
    evaluate_prospective_geometry_gain,
)


def candidate(candidate_id: str, score: float, evidence_digit: str) -> ProspectiveGeometryCandidate:
    return ProspectiveGeometryCandidate(
        candidate_id=candidate_id,
        geometry_score=score,
        geometry_evidence_sha256=evidence_digit * 64,
    )


def plan(*, reverse: bool = False) -> ProspectiveGeometryPlan:
    rows = [
        candidate("actions", 0.10, "1"),
        candidate("vjepa2", 0.30, "2"),
        candidate("vjepa21-dense", 0.50, "3"),
        candidate("world-model", 0.70, "4"),
        candidate("temporal-fusion", 0.90, "5"),
    ]
    if reverse:
        rows.reverse()
    return ProspectiveGeometryPlan(
        plan_id="algonauts-g1-geometry-screen",
        candidates=tuple(rows),
        geometry_metric="development_neural_geometry_score",
        evaluation_fingerprint="a" * 64,
        split_fingerprint="b" * 64,
        preprocessing_fingerprint="c" * 64,
        source_revision="d" * 40,
    )


def reveal(
    frozen_plan: ProspectiveGeometryPlan,
    values: dict[str, float] | None = None,
    *,
    reverse: bool = False,
) -> ProspectiveGeometryReveal:
    values = values or {
        "actions": 0.001,
        "vjepa2": 0.010,
        "vjepa21-dense": 0.020,
        "world-model": 0.030,
        "temporal-fusion": 0.040,
    }
    evidence_digits = ("6", "7", "8", "9", "a")
    rows = [
        ProspectiveOutcome(
            candidate_id=candidate_id,
            validation_pearson_delta=value,
            outcome_evidence_sha256=evidence_digits[index] * 64,
        )
        for index, (candidate_id, value) in enumerate(values.items())
    ]
    if reverse:
        rows.reverse()
    return ProspectiveGeometryReveal(
        plan_fingerprint=frozen_plan.fingerprint,
        outcomes=tuple(rows),
        source_revision="e" * 40,
    )


def test_plan_identity_is_independent_of_input_candidate_order() -> None:
    forward = plan()
    backward = plan(reverse=True)
    assert forward.fingerprint == backward.fingerprint
    assert forward.candidate_set_sha256 == backward.candidate_set_sha256
    assert forward.geometry_projection_sha256 == backward.geometry_projection_sha256


def test_plan_requires_at_least_five_candidates() -> None:
    frozen = plan()
    with pytest.raises(ValueError, match="at least 5"):
        ProspectiveGeometryPlan(
            plan_id=frozen.plan_id,
            candidates=frozen.candidates[:4],
            geometry_metric=frozen.geometry_metric,
            evaluation_fingerprint=frozen.evaluation_fingerprint,
            split_fingerprint=frozen.split_fingerprint,
            preprocessing_fingerprint=frozen.preprocessing_fingerprint,
            source_revision=frozen.source_revision,
        )


def test_plan_rejects_constant_geometry_scores() -> None:
    rows = tuple(candidate(f"candidate-{index}", 0.5, str(index + 1)) for index in range(5))
    with pytest.raises(ValueError, match="must not be constant"):
        ProspectiveGeometryPlan(
            plan_id="constant-screen",
            candidates=rows,
            geometry_metric="rsa",
            evaluation_fingerprint="a" * 64,
            split_fingerprint="b" * 64,
            preprocessing_fingerprint="c" * 64,
            source_revision="d" * 40,
        )


def test_plan_artifact_round_trip_and_unsigned_tamper_detection() -> None:
    frozen = plan()
    artifact = frozen.to_artifact()
    assert ProspectiveGeometryPlan.from_artifact(artifact) == frozen
    tampered = copy.deepcopy(artifact)
    tampered["candidates"][0]["geometry_score"] = 0.99
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ProspectiveGeometryPlan.from_artifact(tampered)


def test_plan_artifact_rejects_unknown_fields_even_if_refingerprinted() -> None:
    artifact = plan().to_artifact()
    artifact["post_hoc_note"] = "added after geometry freeze"
    artifact.pop("fingerprint")
    artifact["fingerprint"] = canonical_sha256(artifact)
    with pytest.raises(ValueError, match="non-canonical or altered fields"):
        ProspectiveGeometryPlan.from_artifact(artifact)


def test_reveal_must_reference_exact_frozen_plan() -> None:
    frozen = plan()
    disclosed = reveal(frozen)
    altered = ProspectiveGeometryPlan(
        plan_id=frozen.plan_id,
        candidates=(
            candidate("actions", 0.11, "1"),
            *frozen.candidates[1:],
        ),
        geometry_metric=frozen.geometry_metric,
        evaluation_fingerprint=frozen.evaluation_fingerprint,
        split_fingerprint=frozen.split_fingerprint,
        preprocessing_fingerprint=frozen.preprocessing_fingerprint,
        source_revision=frozen.source_revision,
    )
    with pytest.raises(ValueError, match="exact frozen plan fingerprint"):
        evaluate_prospective_geometry_gain(altered, disclosed)


def test_reveal_requires_exact_candidate_set() -> None:
    frozen = plan()
    rows = tuple(
        ProspectiveOutcome(
            candidate_id=row.candidate_id,
            validation_pearson_delta=float(index),
            outcome_evidence_sha256="f" * 64,
        )
        for index, row in enumerate(frozen.candidates[:-1])
    )
    disclosed = ProspectiveGeometryReveal(
        plan_fingerprint=frozen.fingerprint,
        outcomes=rows,
        source_revision="e" * 40,
    )
    with pytest.raises(ValueError, match="candidate set mismatch"):
        evaluate_prospective_geometry_gain(frozen, disclosed)


def test_reveal_artifact_round_trip_and_tamper_detection() -> None:
    frozen = plan()
    disclosed = reveal(frozen)
    artifact = disclosed.to_artifact()
    assert ProspectiveGeometryReveal.from_artifact(artifact) == disclosed
    artifact["outcomes"][0]["validation_pearson_delta"] = 10.0
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        ProspectiveGeometryReveal.from_artifact(artifact)


def test_perfect_positive_screen_has_rho_one_and_exact_permutation_evidence() -> None:
    frozen = plan()
    result = evaluate_prospective_geometry_gain(frozen, reveal(frozen))
    assert result["metric_name"] == "prospective_geometry_gain_spearman"
    assert result["metric_value"] == pytest.approx(1.0)
    assert result["candidate_count"] == 5
    permutation = result["permutation_evidence"]
    assert permutation["mode"] == "exact_label_permutation"
    assert permutation["permutations"] == 120
    assert permutation["extreme_count"] == 1
    assert permutation["p_value"] == pytest.approx(1 / 120)
    stability = result["leave_one_out_stability"]
    assert stability["defined_count"] == 5
    assert stability["all_positive"] is True
    assert stability["minimum_rho"] == pytest.approx(1.0)
    assert len(result["fingerprint"]) == 64


def test_reverse_association_is_not_positive_evidence() -> None:
    frozen = plan()
    values = {
        "actions": 0.040,
        "vjepa2": 0.030,
        "vjepa21-dense": 0.020,
        "world-model": 0.010,
        "temporal-fusion": 0.001,
    }
    result = evaluate_prospective_geometry_gain(frozen, reveal(frozen, values))
    assert result["metric_value"] == pytest.approx(-1.0)
    assert result["permutation_evidence"]["p_value"] == pytest.approx(1.0)
    assert result["leave_one_out_stability"]["all_positive"] is False


def test_constant_revealed_outcome_fails_closed() -> None:
    frozen = plan()
    values = {row.candidate_id: 0.01 for row in frozen.candidates}
    with pytest.raises(ValueError, match="undefined for a constant vector"):
        evaluate_prospective_geometry_gain(frozen, reveal(frozen, values))


def test_reveal_identity_and_result_are_independent_of_input_row_order() -> None:
    frozen = plan()
    forward = reveal(frozen)
    backward = reveal(frozen, reverse=True)
    assert forward.fingerprint == backward.fingerprint
    assert evaluate_prospective_geometry_gain(frozen, forward) == evaluate_prospective_geometry_gain(
        frozen, backward
    )


def test_invalid_evidence_hash_and_revision_fail_closed() -> None:
    with pytest.raises(ValueError, match="geometry_evidence_sha256"):
        ProspectiveGeometryCandidate("bad", 0.2, "not-a-hash")
    frozen = plan()
    with pytest.raises(ValueError, match="source_revision"):
        ProspectiveGeometryReveal(
            plan_fingerprint=frozen.fingerprint,
            outcomes=reveal(frozen).outcomes,
            source_revision="not-a-revision",
        )
