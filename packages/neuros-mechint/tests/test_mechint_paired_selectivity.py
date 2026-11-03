"""Adversarial synthetic tests for held-out paired selectivity."""

from dataclasses import replace

import pytest

from neuros_mechint.benchmarks.paired_selectivity import (
    PairedSelectivityDesign,
    PairedSelectivityObservation,
    evaluate_paired_selectivity,
)


def _fixture(*, candidate_drop=5.0, control_drop=0.0, magnitude_drop=0.0, n=8):
    # Two-target circuits with 24 distinct, same-cardinality controls.
    controls = tuple(
        (f"random-{i}", (f"r{i}", f"s{i}")) for i in range(24)
    )
    design = PairedSelectivityDesign(
        design_id="toy-repetition-v3",
        model_revision="synthetic-pinned-model-v1",
        discovery_commitment="d" * 64,
        candidate_id="candidate",
        candidate_targets=("a", "b"),
        magnitude_id="magnitude",
        magnitude_targets=("c", "d"),
        random_controls=controls,
        validation_pair_hashes=tuple((f"heldout-{i}", f"{i + 1:064x}") for i in range(n)),
        bootstrap_samples=200,
    )
    observations = []
    for pair_id, digest in design.validation_pair_hashes:
        for circuit_id in ("candidate", "magnitude", *(name for name, _ in controls)):
            effect = (
                candidate_drop if circuit_id == "candidate"
                else magnitude_drop if circuit_id == "magnitude"
                else 0.0
            )
            observations.append(PairedSelectivityObservation(
                pair_id=pair_id,
                pair_content_hash=digest,
                circuit_id=circuit_id,
                intact_repeated=5.0,
                intact_control=0.0,
                intervened_repeated=5.0 - effect - control_drop,
                intervened_control=0.0 - control_drop,
            ))
    return design, tuple(observations)


def test_planted_selective_path_supports_diagnostic_only():
    design, observations = _fixture()
    report = evaluate_paired_selectivity(design, observations)
    assert report["status"] == "SUPPORTED_DIAGNOSTIC_ONLY"
    assert report["supports_selective_effect"]
    assert report["mean_candidate_difference_in_differences"] == 5.0
    assert report["random_control_plus_one_p"] == pytest.approx(1 / 25)
    assert report["preregistration_independently_verified"] is False
    assert report["real_model_evidence"] is False
    assert len(report["result_hash"]) == 64
    assert report == evaluate_paired_selectivity(design, observations[::-1])


def test_generic_degradation_cannot_masquerade_as_selective():
    design, observations = _fixture(candidate_drop=0, control_drop=8)
    report = evaluate_paired_selectivity(design, observations)
    assert report["status"] == "REJECTED_NO_SELECTIVE_EFFECT"
    assert report["mean_candidate_difference_in_differences"] == 0.0


def test_magnitude_comparator_can_defeat_candidate():
    design, observations = _fixture(candidate_drop=3, magnitude_drop=4)
    report = evaluate_paired_selectivity(design, observations)
    assert report["status"] == "REJECTED_CONTROLS"
    assert report["ci_candidate_advantage_vs_magnitude"][1] < 0


def test_token_prior_and_weak_behavior_fail_competence():
    design, observations = _fixture()
    observations = tuple(
        replace(row, intact_control=5.0, intervened_control=-5.0)
        for row in observations
    )
    result = evaluate_paired_selectivity(design, observations)
    assert result["status"] == "REJECTED_BEHAVIORAL_COMPETENCE"


def test_missing_rows_are_nonestimable_not_zero_effect():
    design, observations = _fixture()
    result = evaluate_paired_selectivity(design, observations[:-1])
    assert result["status"] == "NONESTIMABLE_MISSING_OBSERVATIONS"
    assert result["supports_selective_effect"] is False
    assert len(result["missing"]) == 1
    assert "mean_candidate_difference_in_differences" not in result


def test_duplicate_and_mismatched_pair_and_baselines_fail_closed():
    design, observations = _fixture()
    with pytest.raises(ValueError, match="duplicate matched"):
        evaluate_paired_selectivity(design, observations + (observations[0],))
    with pytest.raises(ValueError, match="content hash mismatch"):
        evaluate_paired_selectivity(
            design, (replace(observations[0], pair_content_hash="f" * 64), *observations[1:])
        )
    with pytest.raises(ValueError, match="intact baseline drift"):
        evaluate_paired_selectivity(
            design, (observations[0], replace(observations[1], intact_repeated=8), *observations[2:])
        )


def test_discovery_or_nonfinite_in_validation_fails_closed():
    design, observations = _fixture()
    with pytest.raises(ValueError, match="discovery observations"):
        evaluate_paired_selectivity(
            design, (replace(observations[0], split="discovery"), *observations[1:])
        )
    with pytest.raises(ValueError, match="nonfinite"):
        evaluate_paired_selectivity(
            design, (replace(observations[0], intervened_repeated=float("nan")), *observations[1:])
        )


def test_invalid_or_underpowered_design_never_promotes():
    design, observations = _fixture(n=4)
    assert evaluate_paired_selectivity(design, observations)["status"] == "INSUFFICIENT_COVERAGE"
    with pytest.raises(ValueError, match="same-size"):
        replace(design, magnitude_targets=("c",))
    with pytest.raises(ValueError, match="duplicate validation pair content"):
        replace(design, validation_pair_hashes=(("a", "a" * 64), ("b", "a" * 64)))
    with pytest.raises(ValueError, match="must be distinct"):
        replace(design, magnitude_targets=("b", "a"))


def test_novel_suffix_comparison_is_sensitivity_only():
    design, observations = _fixture()
    report = evaluate_paired_selectivity(
        replace(design, control_condition="novel"), observations
    )
    assert report["status"] == "SENSITIVITY_ONLY_NOT_PRIMARY"
    assert report["supports_selective_effect"] is False
