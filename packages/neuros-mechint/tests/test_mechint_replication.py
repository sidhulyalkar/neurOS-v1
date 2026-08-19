from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from neuros_mechint.benchmarks.dose_response import (
    DoseResponseObservation,
    DoseResponsePolicy,
    DoseResponseSpec,
    InterventionManifoldAssumption,
    InterventionManifoldKind,
    analyze_dose_response,
)
from neuros_mechint.benchmarks.replication import (
    HierarchicalReplicationPolicy,
    HierarchicalReplicationSpec,
    ReplicationAxis,
    ReplicationCoordinates,
    ReplicationObservation,
    analyze_hierarchical_replication,
    observation_from_correspondence,
    observation_from_factorial_contrast,
    read_replication_artifact,
    write_replication_artifact,
)
from neuros_mechint.benchmarks.replication_ground_truth import (
    run_replication_ground_truth_benchmark,
)


def _coords(seed: int, trial: int, *, subject: int = 0) -> ReplicationCoordinates:
    return ReplicationCoordinates(
        model_seed=seed,
        subject_id=f"subject:{subject}",
        session_id=f"subject:{subject}/session:0",
        trial_id=f"seed:{seed}/subject:{subject}/trial:{trial}",
    )


def _spec(**policy_overrides) -> HierarchicalReplicationSpec:
    policy = HierarchicalReplicationPolicy(
        min_independent_units=policy_overrides.pop("min_independent_units", 3),
        bootstrap_samples=policy_overrides.pop("bootstrap_samples", 200),
        confidence_level=0.95,
        min_sign_agreement=policy_overrides.pop("min_sign_agreement", 0.75),
        min_estimable_fraction=policy_overrides.pop("min_estimable_fraction", 0.8),
        require_ci_excludes_null=policy_overrides.pop("require_ci_excludes_null", True),
        min_absolute_effect=policy_overrides.pop("min_absolute_effect", 0.1),
        **policy_overrides,
    )
    return HierarchicalReplicationSpec(
        study_id="replication-test",
        family_id="family",
        claim_axis=ReplicationAxis.MODEL_SEED,
        primary_metric="effect",
        hierarchy=(ReplicationAxis.MODEL_SEED, ReplicationAxis.SUBJECT, ReplicationAxis.TRIAL),
        expected_direction=1,
        seed=7,
        policy=policy,
    )


def test_model_seed_claim_counts_seeds_not_trials() -> None:
    observations = []
    for seed, effect, trials in ((0, 0.8, 100), (1, 0.7, 3), (2, 0.6, 3)):
        observations.extend(
            ReplicationObservation(
                observation_id=f"{seed}:{trial}",
                family_id="family",
                coordinates=_coords(seed, trial),
                metrics={"effect": effect},
            )
            for trial in range(trials)
        )
    result = analyze_hierarchical_replication(_spec(), observations)
    assert result.decision.estimable
    assert result.decision.replicated
    assert result.primary_estimate is not None
    assert result.primary_estimate.independent_units == 3
    assert result.primary_estimate.estimate == pytest.approx(0.7)


def test_many_trials_from_one_seed_are_not_replication() -> None:
    observations = [
        ReplicationObservation(
            observation_id=f"trial:{trial}",
            family_id="family",
            coordinates=_coords(0, trial),
            metrics={"effect": 0.9},
        )
        for trial in range(500)
    ]
    result = analyze_hierarchical_replication(_spec(), observations)
    assert not result.decision.estimable
    assert not result.decision.replicated
    assert any("1 independent unit" in reason for reason in result.decision.reasons)


def test_estimable_but_sign_inconsistent_replication_is_rejected() -> None:
    observations = []
    for seed, effect in enumerate((0.8, 0.7, -0.8, -0.7)):
        observations.extend(
            ReplicationObservation(
                observation_id=f"{seed}:{trial}",
                family_id="family",
                coordinates=_coords(seed, trial),
                metrics={"effect": effect},
            )
            for trial in range(5)
        )
    result = analyze_hierarchical_replication(_spec(min_independent_units=4), observations)
    assert result.decision.estimable
    assert not result.decision.replicated
    assert result.primary_estimate is not None
    assert result.primary_estimate.sign_agreement == pytest.approx(0.5)


def test_missing_claim_hierarchy_coordinate_is_nonestimable() -> None:
    observations = [
        ReplicationObservation(
            observation_id=f"{seed}",
            family_id="family",
            coordinates=ReplicationCoordinates(model_seed=seed, trial_id=f"trial:{seed}"),
            metrics={"effect": 0.5},
        )
        for seed in range(3)
    ]
    result = analyze_hierarchical_replication(_spec(), observations)
    assert not result.decision.estimable
    assert any("subject" in reason for reason in result.decision.reasons)


def test_family_mismatch_is_nonestimable() -> None:
    observations = [
        ReplicationObservation(
            observation_id=f"{seed}",
            family_id="other-family" if seed == 2 else "family",
            coordinates=_coords(seed, 0),
            metrics={"effect": 0.5},
        )
        for seed in range(3)
    ]
    result = analyze_hierarchical_replication(_spec(), observations)
    assert not result.decision.estimable
    assert any("family" in reason for reason in result.decision.reasons)


def test_hierarchical_bootstrap_is_deterministic() -> None:
    observations = [
        ReplicationObservation(
            observation_id=f"{seed}:{trial}",
            family_id="family",
            coordinates=_coords(seed, trial),
            metrics={"effect": 0.4 + 0.1 * seed + 0.01 * trial},
        )
        for seed in range(4)
        for trial in range(4)
    ]
    first = analyze_hierarchical_replication(_spec(min_independent_units=4), observations)
    second = analyze_hierarchical_replication(_spec(min_independent_units=4), observations)
    assert first.to_dict() == second.to_dict()


def test_replication_artifact_round_trip_and_tamper_detection(tmp_path) -> None:
    observations = [
        ReplicationObservation(
            observation_id=f"{seed}:{trial}",
            family_id="family",
            coordinates=_coords(seed, trial),
            metrics={"effect": 0.6 + 0.05 * seed},
        )
        for seed in range(3)
        for trial in range(4)
    ]
    result = analyze_hierarchical_replication(_spec(), observations)
    path = write_replication_artifact(result, tmp_path / "replication.json")
    payload = read_replication_artifact(path)
    assert payload["study_fingerprint"] == result.study_fingerprint

    envelope = json.loads(path.read_text())
    envelope["result"]["decision"]["replicated"] = False
    path.write_text(json.dumps(envelope))
    with pytest.raises(ValueError, match="integrity hash"):
        read_replication_artifact(path)


def test_correspondence_bridge_preserves_causal_metrics() -> None:
    validation = SimpleNamespace(
        median_causal_recovery=0.9,
        median_causal_score=0.8,
        predictive_r2=0.7,
        random_margin=0.5,
        shuffled_margin=0.4,
        median_source_effect=0.3,
        median_target_effect=0.2,
    )
    result = SimpleNamespace(
        validation_metrics=validation,
        promotion=SimpleNamespace(passed=True),
        study_fingerprint="correspondence-fingerprint",
    )
    observation = observation_from_correspondence(
        result,
        observation_id="replica:0",
        family_id="family",
        coordinates=_coords(0, 0),
    )
    assert observation.metrics["causal_recovery"] == pytest.approx(0.9)
    assert observation.metadata["promotion_passed"] is True


def test_factorial_bridge_preserves_nonestimable_result() -> None:
    contrast = SimpleNamespace(
        contrast_id="interaction",
        outcome_effects={"validation_joint_faithfulness": -0.4},
        estimable=False,
        reasons=("token budget mismatch",),
    )
    report = SimpleNamespace(contrasts=(contrast,), study_fingerprint="factorial-fingerprint")
    observation = observation_from_factorial_contrast(
        report,
        contrast_id="interaction",
        observation_id="replica:0",
        family_id="family",
        coordinates=_coords(0, 0),
    )
    assert not observation.estimable
    assert observation.rejection_reasons == ("token budget mismatch",)


def _manifold() -> InterventionManifoldAssumption:
    return InterventionManifoldAssumption(
        kind=InterventionManifoldKind.EMPIRICAL_DONOR,
        description="held-out activation donor",
        donor_pool_id="validation-pool",
        expected_in_manifold=True,
    )


def test_monotonic_dose_response_passes() -> None:
    spec = DoseResponseSpec(
        study_id="dose",
        intervention_id="substitution",
        expected_direction=1,
        manifold=_manifold(),
        policy=DoseResponsePolicy(min_units=3, min_endpoint_effect=0.5),
    )
    observations = [
        DoseResponseObservation(unit_id=f"u:{unit}", dose=dose, metric=0.8 * dose + unit)
        for unit in range(3)
        for dose in (0.0, 0.25, 0.5, 0.75, 1.0)
    ]
    result = analyze_dose_response(spec, observations)
    assert result.passed
    assert result.endpoint_effect == pytest.approx(0.8)
    assert result.mean_monotonic_fraction == pytest.approx(1.0)


def test_nonmonotonic_dose_response_is_rejected() -> None:
    spec = DoseResponseSpec(
        study_id="dose",
        intervention_id="substitution",
        expected_direction=1,
        manifold=_manifold(),
        policy=DoseResponsePolicy(min_units=3, min_monotonic_fraction=0.9),
    )
    curve = (0.0, 1.0, 0.2, 0.9, 0.3)
    observations = [
        DoseResponseObservation(unit_id=f"u:{unit}", dose=dose, metric=value)
        for unit in range(3)
        for dose, value in zip((0.0, 0.25, 0.5, 0.75, 1.0), curve, strict=True)
    ]
    result = analyze_dose_response(spec, observations)
    assert not result.passed
    assert any("monotonic" in reason for reason in result.reasons)


def test_fitted_manifold_donor_requires_discovery_partition() -> None:
    with pytest.raises(ValueError, match="fitted_on_partition_id"):
        InterventionManifoldAssumption(
            kind=InterventionManifoldKind.GENERATIVE,
            description="latent generator",
            donor_pool_id="generator:0",
            expected_in_manifold=True,
        )


def test_v09_ground_truth_recovers_replication_and_rejects_pseudoreplication() -> None:
    report = run_replication_ground_truth_benchmark(seed=0)
    assert report.passed
    assert report.true_replication_recovered
    assert report.pseudoreplication_rejected
    assert report.heterogeneous_replication_rejected
    assert report.independent_seed_count_correct
    assert report.dose_response_recovered
