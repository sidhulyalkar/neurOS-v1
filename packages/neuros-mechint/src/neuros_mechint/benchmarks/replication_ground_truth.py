"""Known-ground-truth benchmark for v0.9 hierarchical replication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dose_response import (
    DoseResponseObservation,
    DoseResponsePolicy,
    DoseResponseSpec,
    InterventionManifoldAssumption,
    InterventionManifoldKind,
    analyze_dose_response,
)
from .replication import (
    HierarchicalReplicationPolicy,
    HierarchicalReplicationSpec,
    ReplicationAxis,
    ReplicationCoordinates,
    ReplicationObservation,
    analyze_hierarchical_replication,
)


@dataclass(frozen=True, slots=True)
class ReplicationGroundTruthReport:
    true_replication_recovered: bool
    pseudoreplication_rejected: bool
    heterogeneous_replication_rejected: bool
    independent_seed_count_correct: bool
    dose_response_recovered: bool
    true_primary_estimate: float
    true_ci_low: float
    true_ci_high: float
    true_sign_agreement: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "dose_response_recovered": self.dose_response_recovered,
            "heterogeneous_replication_rejected": self.heterogeneous_replication_rejected,
            "independent_seed_count_correct": self.independent_seed_count_correct,
            "passed": self.passed,
            "pseudoreplication_rejected": self.pseudoreplication_rejected,
            "true_ci_high": self.true_ci_high,
            "true_ci_low": self.true_ci_low,
            "true_primary_estimate": self.true_primary_estimate,
            "true_replication_recovered": self.true_replication_recovered,
            "true_sign_agreement": self.true_sign_agreement,
        }


def _coordinates(seed: int, subject: int, session: int, trial: int) -> ReplicationCoordinates:
    return ReplicationCoordinates(
        dataset_id="synthetic-neural-dataset",
        model_seed=seed,
        checkpoint="step:10000",
        dictionary_id="dictionary:0",
        projector_id="event-preserving-v1",
        subject_id=f"subject:{subject}",
        session_id=f"subject:{subject}/session:{session}",
        trial_id=f"seed:{seed}/subject:{subject}/session:{session}/trial:{trial}",
    )


def _replicated_observations(seed: int) -> tuple[ReplicationObservation, ...]:
    rng = np.random.default_rng(seed)
    seed_effects = (0.72, 0.61, 0.68, 0.57)
    observations = []
    for model_seed, effect in enumerate(seed_effects):
        for subject in range(2):
            for session in range(2):
                trial_count = 4 + ((model_seed + subject + session) % 3)
                for trial in range(trial_count):
                    value = effect + 0.03 * subject - 0.02 * session + rng.normal(0.0, 0.025)
                    observations.append(
                        ReplicationObservation(
                            observation_id=f"true:{model_seed}:{subject}:{session}:{trial}",
                            family_id="shared-causal-correspondence",
                            coordinates=_coordinates(model_seed, subject, session, trial),
                            metrics={"causal_recovery_margin": float(value)},
                        )
                    )
    return tuple(observations)


def _pseudoreplicated_observations(seed: int) -> tuple[ReplicationObservation, ...]:
    rng = np.random.default_rng(seed + 1)
    return tuple(
        ReplicationObservation(
            observation_id=f"pseudo:{trial}",
            family_id="shared-causal-correspondence",
            coordinates=_coordinates(0, 0, 0, trial),
            metrics={"causal_recovery_margin": float(0.8 + rng.normal(0.0, 0.02))},
        )
        for trial in range(300)
    )


def _heterogeneous_observations(seed: int) -> tuple[ReplicationObservation, ...]:
    rng = np.random.default_rng(seed + 2)
    seed_effects = (0.75, 0.68, -0.72, -0.66)
    observations = []
    for model_seed, effect in enumerate(seed_effects):
        for trial in range(10):
            observations.append(
                ReplicationObservation(
                    observation_id=f"heterogeneous:{model_seed}:{trial}",
                    family_id="shared-causal-correspondence",
                    coordinates=_coordinates(model_seed, model_seed, 0, trial),
                    metrics={
                        "causal_recovery_margin": float(effect + rng.normal(0.0, 0.02))
                    },
                )
            )
    return tuple(observations)


def _spec(study_id: str, *, min_units: int = 3) -> HierarchicalReplicationSpec:
    return HierarchicalReplicationSpec(
        study_id=study_id,
        family_id="shared-causal-correspondence",
        claim_axis=ReplicationAxis.MODEL_SEED,
        primary_metric="causal_recovery_margin",
        hierarchy=(
            ReplicationAxis.MODEL_SEED,
            ReplicationAxis.SUBJECT,
            ReplicationAxis.SESSION,
            ReplicationAxis.TRIAL,
        ),
        null_value=0.0,
        expected_direction=1,
        seed=17,
        policy=HierarchicalReplicationPolicy(
            min_independent_units=min_units,
            bootstrap_samples=500,
            confidence_level=0.95,
            min_sign_agreement=0.75,
            min_estimable_fraction=0.90,
            require_ci_excludes_null=True,
            min_absolute_effect=0.20,
        ),
    )


def _dose_response() -> bool:
    manifold = InterventionManifoldAssumption(
        kind=InterventionManifoldKind.EMPIRICAL_DONOR,
        description="held-out empirical activation donor",
        donor_pool_id="validation-donor-pool",
        expected_in_manifold=True,
    )
    spec = DoseResponseSpec(
        study_id="dose-response-ground-truth",
        intervention_id="mapped-substitution",
        expected_direction=1,
        manifold=manifold,
        policy=DoseResponsePolicy(
            min_doses=5,
            min_units=3,
            min_monotonic_fraction=0.9,
            min_endpoint_effect=0.5,
        ),
    )
    observations = []
    for unit in range(4):
        for dose in (0.0, 0.25, 0.5, 0.75, 1.0):
            observations.append(
                DoseResponseObservation(
                    unit_id=f"unit:{unit}",
                    dose=dose,
                    metric=0.1 * unit + 0.8 * dose,
                )
            )
    return analyze_dose_response(spec, observations).passed


def run_replication_ground_truth_benchmark(seed: int = 0) -> ReplicationGroundTruthReport:
    """Verify replication, pseudoreplication rejection, heterogeneity, and dose response."""

    true_result = analyze_hierarchical_replication(
        _spec("true-replication"),
        _replicated_observations(seed),
    )
    pseudo_result = analyze_hierarchical_replication(
        _spec("pseudoreplication"),
        _pseudoreplicated_observations(seed),
    )
    heterogeneous_result = analyze_hierarchical_replication(
        _spec("heterogeneous-replication"),
        _heterogeneous_observations(seed),
    )
    primary = true_result.primary_estimate
    if primary is None:
        raise AssertionError("ground-truth replicated study unexpectedly produced no estimate")
    true_recovered = true_result.decision.estimable and true_result.decision.replicated
    pseudo_rejected = not pseudo_result.decision.estimable
    heterogeneous_rejected = (
        heterogeneous_result.decision.estimable and not heterogeneous_result.decision.replicated
    )
    seed_count_correct = primary.independent_units == 4
    dose_recovered = _dose_response()
    passed = all(
        (
            true_recovered,
            pseudo_rejected,
            heterogeneous_rejected,
            seed_count_correct,
            dose_recovered,
        )
    )
    return ReplicationGroundTruthReport(
        true_replication_recovered=true_recovered,
        pseudoreplication_rejected=pseudo_rejected,
        heterogeneous_replication_rejected=heterogeneous_rejected,
        independent_seed_count_correct=seed_count_correct,
        dose_response_recovered=dose_recovered,
        true_primary_estimate=primary.estimate,
        true_ci_low=primary.ci_low,
        true_ci_high=primary.ci_high,
        true_sign_agreement=primary.sign_agreement,
        passed=passed,
    )
