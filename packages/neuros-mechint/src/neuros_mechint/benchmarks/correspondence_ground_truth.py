"""Known-positive and known-decoy ground truth for causal feature correspondence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .correspondence import (
    CausalSubstitutionMetrics,
    CorrespondenceKind,
    CorrespondenceSplit,
    FeatureCorrespondencePolicy,
    FeatureCorrespondenceSpec,
    FeaturePairExample,
    FeatureSpaceIdentity,
    run_feature_correspondence_study,
)


@dataclass(frozen=True, slots=True)
class CorrespondenceGroundTruthReport:
    true_correspondence_passed: bool
    decoy_similarity_high: bool
    decoy_causal_rejected: bool
    true_validation_predictive_r2: float
    true_median_causal_recovery: float
    true_shuffled_margin: float
    true_random_percentile: float | None
    decoy_validation_predictive_r2: float
    decoy_median_source_effect: float
    passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "decoy_causal_rejected": self.decoy_causal_rejected,
            "decoy_median_source_effect": self.decoy_median_source_effect,
            "decoy_similarity_high": self.decoy_similarity_high,
            "decoy_validation_predictive_r2": self.decoy_validation_predictive_r2,
            "passed": self.passed,
            "true_correspondence_passed": self.true_correspondence_passed,
            "true_median_causal_recovery": self.true_median_causal_recovery,
            "true_random_percentile": self.true_random_percentile,
            "true_shuffled_margin": self.true_shuffled_margin,
            "true_validation_predictive_r2": self.true_validation_predictive_r2,
        }


class _SyntheticCausalEvaluator:
    def __init__(self, examples: list[FeaturePairExample], source_names: tuple[str, ...]) -> None:
        self.examples = {item.example_id: item for item in examples}
        self.source_index = {name: index for index, name in enumerate(source_names)}

    def __call__(
        self,
        *,
        target_example_id: str,
        source_example_id: str,
        source_features: tuple[str, ...],
        target_features: tuple[str, ...],
        replacement_values: np.ndarray,
    ) -> CausalSubstitutionMetrics:
        if target_features != ("target_signal",):
            raise ValueError("synthetic evaluator expects target_signal")
        target_example = self.examples[target_example_id]
        source_example = self.examples[source_example_id]
        target_value = float(target_example.target_activation[0])
        source_signal = float(source_example.source_activation[self.source_index["causal_signal"]])
        source_causal = "causal_signal" in source_features
        source_effect = (0.25 + 0.15 * abs(source_signal)) if source_causal else 0.0
        target_effect = 0.30 + 0.10 * abs(target_value)
        replacement = float(np.asarray(replacement_values, dtype=np.float64).reshape(-1)[0])
        error = replacement - target_value
        substituted = 1.0 - (error * error) / max(target_value * target_value, 0.25)
        return CausalSubstitutionMetrics(
            source_clean_metric=1.0,
            source_ablated_metric=1.0 - source_effect,
            target_clean_metric=1.0,
            target_ablated_metric=1.0 - target_effect,
            target_substituted_metric=substituted,
        )


def _spaces() -> tuple[FeatureSpaceIdentity, FeatureSpaceIdentity]:
    source_names = ("causal_signal", "correlated_decoy") + tuple(
        f"nuisance_{index:02d}" for index in range(22)
    )
    source = FeatureSpaceIdentity(
        space_id="source-space",
        model_id="source-model",
        model_revision="source-model@sha256:111",
        representation_id="hidden.source",
        feature_names=source_names,
        architecture="transformer",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="synthetic-correspondence",
        dataset_revision="dataset@sha256:aaa",
        session_id="session-1",
        checkpoint="step-100",
        subject_id="synthetic-subject",
        feature_semantics={
            "causal_signal": "shared latent signal",
            "correlated_decoy": "shared latent signal",
        },
    )
    target = FeatureSpaceIdentity(
        space_id="target-space",
        model_id="target-model",
        model_revision="target-model@sha256:222",
        representation_id="hidden.target",
        feature_names=("target_signal", "target_nuisance_0", "target_nuisance_1"),
        architecture="ssm",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="synthetic-correspondence",
        dataset_revision="dataset@sha256:aaa",
        session_id="session-1",
        checkpoint="step-100",
        subject_id="synthetic-subject",
        feature_semantics={"target_signal": "shared latent signal"},
    )
    return source, target


def _examples(seed: int) -> list[FeaturePairExample]:
    rng = np.random.default_rng(seed)
    source_space, target_space = _spaces()
    examples = []
    for index in range(30):
        latent = float(np.exp(rng.uniform(-1.5, 1.5)))
        source = rng.normal(0.0, 1.0, size=len(source_space.feature_names))
        source[0] = latent
        source[1] = latent + rng.normal(0.0, 0.01)
        target = rng.normal(0.0, 1.0, size=len(target_space.feature_names))
        target[0] = 1.7 * latent + 0.3
        split = CorrespondenceSplit.DISCOVERY if index < 18 else CorrespondenceSplit.VALIDATION
        partition = "discovery-trials" if split is CorrespondenceSplit.DISCOVERY else "validation-trials"
        examples.append(
            FeaturePairExample(
                example_id=f"example-{index:02d}",
                semantic_trial_id=f"trial-{index:02d}",
                split=split,
                partition_id=partition,
                source_activation=source,
                target_activation=target,
            )
        )
    return examples


def _spec(
    *,
    study_id: str,
    source_feature: str,
    source_space: FeatureSpaceIdentity,
    target_space: FeatureSpaceIdentity,
) -> FeatureCorrespondenceSpec:
    return FeatureCorrespondenceSpec(
        study_id=study_id,
        source_space=source_space,
        target_space=target_space,
        source_features=(source_feature,),
        target_features=("target_signal",),
        kind=CorrespondenceKind.ONE_TO_ONE,
        discovery_partition_id="discovery-trials",
        validation_partition_id="validation-trials",
        declared_context_differences=("model_id", "model_revision", "architecture"),
        random_controls=16,
        seed=11,
        policy=FeatureCorrespondencePolicy(
            min_discovery_examples=12,
            min_validation_examples=8,
            min_valid_transfer_fraction=1.0,
            min_validation_predictive_r2=0.95,
            min_median_causal_recovery=0.80,
            min_source_effect=0.05,
            min_target_effect=0.05,
            min_random_percentile=0.90,
            min_shuffled_margin=0.20,
            min_random_margin=0.20,
            max_discovery_validation_r2_drop=0.10,
        ),
    )


def run_correspondence_ground_truth_benchmark(*, seed: int = 0) -> CorrespondenceGroundTruthReport:
    """Verify true causal correspondence and reject a correlated non-causal decoy."""

    source_space, target_space = _spaces()
    examples = _examples(seed)
    evaluator = _SyntheticCausalEvaluator(examples, source_space.feature_names)
    true_result = run_feature_correspondence_study(
        _spec(
            study_id="true-correspondence",
            source_feature="causal_signal",
            source_space=source_space,
            target_space=target_space,
        ),
        examples,
        evaluator=evaluator,
    )
    decoy_result = run_feature_correspondence_study(
        _spec(
            study_id="correlated-decoy",
            source_feature="correlated_decoy",
            source_space=source_space,
            target_space=target_space,
        ),
        examples,
        evaluator=evaluator,
    )
    true_metrics = true_result.validation_metrics
    decoy_metrics = decoy_result.validation_metrics
    true_passed = true_result.promotion.passed
    decoy_similarity_high = (
        decoy_metrics.predictive_r2 >= 0.95
        and decoy_result.candidate.semantic_label_overlap == 1.0
    )
    decoy_rejected = (
        not decoy_result.promotion.passed
        and decoy_metrics.median_source_effect < decoy_result.spec.policy.min_source_effect
    )
    passed = (
        true_passed
        and decoy_similarity_high
        and decoy_rejected
        and true_metrics.shuffled_margin >= true_result.spec.policy.min_shuffled_margin
        and true_metrics.random_control_percentile is not None
        and true_metrics.random_control_percentile >= true_result.spec.policy.min_random_percentile
    )
    return CorrespondenceGroundTruthReport(
        true_correspondence_passed=true_passed,
        decoy_similarity_high=decoy_similarity_high,
        decoy_causal_rejected=decoy_rejected,
        true_validation_predictive_r2=true_metrics.predictive_r2,
        true_median_causal_recovery=true_metrics.median_causal_recovery,
        true_shuffled_margin=true_metrics.shuffled_margin,
        true_random_percentile=true_metrics.random_control_percentile,
        decoy_validation_predictive_r2=decoy_metrics.predictive_r2,
        decoy_median_source_effect=decoy_metrics.median_source_effect,
        passed=passed,
    )
