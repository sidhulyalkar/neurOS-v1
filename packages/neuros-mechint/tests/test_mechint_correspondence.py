from __future__ import annotations

import json

import numpy as np
import pytest

from neuros_mechint.benchmarks.correspondence import (
    CausalSubstitutionMetrics,
    CorrespondenceKind,
    CorrespondenceSplit,
    FeatureCorrespondencePolicy,
    FeatureCorrespondenceSpec,
    FeaturePairExample,
    FeatureSpaceIdentity,
    _random_source_feature_sets,
    fit_feature_correspondence_candidate,
    read_correspondence_artifact,
    run_feature_correspondence_study,
    write_correspondence_artifact,
)
from neuros_mechint.benchmarks.correspondence_ground_truth import (
    run_correspondence_ground_truth_benchmark,
)


def _space(space_id: str, *, model: str, architecture: str) -> FeatureSpaceIdentity:
    return FeatureSpaceIdentity(
        space_id=space_id,
        model_id=model,
        model_revision=f"{model}@sha256:abc",
        representation_id="hidden",
        feature_names=("signal", "nuisance_a", "nuisance_b"),
        architecture=architecture,
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:123",
        session_id="session",
        checkpoint="step-10",
        feature_semantics={"signal": "task signal"},
    )


def _study_inputs():
    source = _space("source", model="source-model", architecture="transformer")
    target = _space("target", model="target-model", architecture="ssm")
    spec = FeatureCorrespondenceSpec(
        study_id="artifact-study",
        source_space=source,
        target_space=target,
        source_features=("signal",),
        target_features=("signal",),
        kind=CorrespondenceKind.ONE_TO_ONE,
        discovery_partition_id="discovery",
        validation_partition_id="validation",
        declared_context_differences=("model_id", "model_revision", "architecture"),
        random_controls=2,
        policy=FeatureCorrespondencePolicy(
            min_discovery_examples=3,
            min_validation_examples=3,
            min_validation_predictive_r2=0.8,
            min_median_causal_recovery=0.8,
            min_random_percentile=0.5,
            min_shuffled_margin=-1.0,
            min_random_margin=0.1,
        ),
    )
    examples = []
    values = (0.4, 0.8, 1.4, 0.55, 1.0, 1.8)
    for index, value in enumerate(values):
        split = CorrespondenceSplit.DISCOVERY if index < 3 else CorrespondenceSplit.VALIDATION
        examples.append(
            FeaturePairExample(
                example_id=f"e{index}",
                semantic_trial_id=f"trial-{index}",
                split=split,
                partition_id="discovery" if index < 3 else "validation",
                source_activation=np.asarray([value, index + 0.2, -index - 0.3]),
                target_activation=np.asarray([2.0 * value + 0.5, -index, index + 1.0]),
            )
        )

    by_id = {item.example_id: item for item in examples}

    def evaluator(**kwargs):
        source_example = by_id[kwargs["source_example_id"]]
        target_example = by_id[kwargs["target_example_id"]]
        source_causal = kwargs["source_features"] == ("signal",)
        source_effect = float(source_example.source_activation[0]) if source_causal else 0.0
        target_value = float(target_example.target_activation[0])
        replacement = float(np.asarray(kwargs["replacement_values"]).reshape(-1)[0])
        return CausalSubstitutionMetrics(
            source_clean_metric=source_effect,
            source_ablated_metric=0.0,
            target_clean_metric=target_value,
            target_ablated_metric=0.0,
            target_substituted_metric=replacement,
        )

    return spec, examples, evaluator


def test_correspondence_ground_truth_rejects_similarity_decoy():
    report = run_correspondence_ground_truth_benchmark(seed=0)
    assert report.passed
    assert report.true_correspondence_passed
    assert report.decoy_similarity_high
    assert report.decoy_causal_rejected
    assert report.decoy_validation_predictive_r2 > 0.99
    assert report.decoy_median_source_effect == 0.0


def test_candidate_fit_receives_discovery_only():
    spec, examples, evaluator = _study_inputs()
    observed_ids = []

    def candidate_fit(inner_spec, discovery_examples):
        observed_ids.extend(item.example_id for item in discovery_examples)
        assert all(item.split is CorrespondenceSplit.DISCOVERY for item in discovery_examples)
        return fit_feature_correspondence_candidate(inner_spec, discovery_examples)

    run_feature_correspondence_study(
        spec,
        examples,
        evaluator=evaluator,
        candidate_fit=candidate_fit,
    )
    assert observed_ids == ["e0", "e1", "e2"]


def test_correspondence_rejects_semantic_trial_leakage():
    spec, examples, evaluator = _study_inputs()
    leaked = list(examples)
    leaked[-1] = FeaturePairExample(
        example_id="renamed-validation",
        semantic_trial_id=examples[0].semantic_trial_id,
        split=CorrespondenceSplit.VALIDATION,
        partition_id="validation",
        source_activation=examples[-1].source_activation,
        target_activation=examples[-1].target_activation,
    )
    with pytest.raises(ValueError, match="semantic_trial_id"):
        run_feature_correspondence_study(spec, leaked, evaluator=evaluator)


def test_correspondence_requires_declared_context_differences():
    source = _space("source", model="source-model", architecture="transformer")
    target = _space("target", model="target-model", architecture="ssm")
    with pytest.raises(ValueError, match="undeclared"):
        FeatureCorrespondenceSpec(
            study_id="bad-context",
            source_space=source,
            target_space=target,
            source_features=("signal",),
            target_features=("signal",),
            kind=CorrespondenceKind.ONE_TO_ONE,
            discovery_partition_id="d",
            validation_partition_id="v",
            declared_context_differences=("model_id",),
        )


def test_one_to_many_mapping_shape_and_prediction():
    source = FeatureSpaceIdentity(
        space_id="source-many",
        model_id="source-model",
        model_revision="source@sha256:1",
        representation_id="hidden",
        feature_names=("signal", "noise"),
        architecture="transformer",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:1",
        session_id="session",
        checkpoint="step",
    )
    target = FeatureSpaceIdentity(
        space_id="target-many",
        model_id="target-model",
        model_revision="target@sha256:1",
        representation_id="hidden",
        feature_names=("out_a", "out_b"),
        architecture="ssm",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:1",
        session_id="session",
        checkpoint="step",
    )
    spec = FeatureCorrespondenceSpec(
        study_id="one-to-many",
        source_space=source,
        target_space=target,
        source_features=("signal",),
        target_features=("out_a", "out_b"),
        kind=CorrespondenceKind.ONE_TO_MANY,
        discovery_partition_id="d",
        validation_partition_id="v",
        declared_context_differences=("model_id", "model_revision", "architecture"),
        policy=FeatureCorrespondencePolicy(min_discovery_examples=3, min_validation_examples=2),
    )
    discovery = tuple(
        FeaturePairExample(
            example_id=f"d{index}",
            semantic_trial_id=f"trial-d{index}",
            split=CorrespondenceSplit.DISCOVERY,
            partition_id="d",
            source_activation=np.asarray([value, index]),
            target_activation=np.asarray([2.0 * value + 1.0, -3.0 * value + 0.5]),
        )
        for index, value in enumerate((0.2, 0.8, 1.7, 2.5))
    )
    candidate = fit_feature_correspondence_candidate(spec, discovery)
    assert np.asarray(candidate.mapping_matrix).shape == (2, 1)
    assert np.allclose(candidate.predict([1.25]), [3.5, -3.25], atol=1e-5)


def test_subspace_mapping_recovers_two_dimensional_basis_change():
    source = FeatureSpaceIdentity(
        space_id="source-subspace",
        model_id="source-model",
        model_revision="source@sha256:2",
        representation_id="hidden",
        feature_names=("x", "y", "noise"),
        architecture="transformer",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:1",
        session_id="session",
        checkpoint="step",
    )
    target = FeatureSpaceIdentity(
        space_id="target-subspace",
        model_id="target-model",
        model_revision="target@sha256:2",
        representation_id="hidden",
        feature_names=("u", "v", "noise"),
        architecture="ssm",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:1",
        session_id="session",
        checkpoint="step",
    )
    spec = FeatureCorrespondenceSpec(
        study_id="subspace",
        source_space=source,
        target_space=target,
        source_features=("x", "y"),
        target_features=("u", "v"),
        kind=CorrespondenceKind.SUBSPACE,
        discovery_partition_id="d",
        validation_partition_id="v",
        declared_context_differences=("model_id", "model_revision", "architecture"),
        policy=FeatureCorrespondencePolicy(min_discovery_examples=4, min_validation_examples=2),
    )
    points = ((0.2, 0.5), (0.7, -0.1), (1.2, 0.9), (-0.4, 1.5), (2.0, -0.8))
    discovery = tuple(
        FeaturePairExample(
            example_id=f"d{index}",
            semantic_trial_id=f"trial-d{index}",
            split=CorrespondenceSplit.DISCOVERY,
            partition_id="d",
            source_activation=np.asarray([x, y, index]),
            target_activation=np.asarray([2.0 * x + y + 0.3, -x + 3.0 * y - 0.2, -index]),
        )
        for index, (x, y) in enumerate(points)
    )
    candidate = fit_feature_correspondence_candidate(spec, discovery)
    assert np.asarray(candidate.mapping_matrix).shape == (2, 2)
    assert np.allclose(candidate.predict([0.6, 1.1]), [2.6, 2.5], atol=1e-4)


def test_rank_deficient_subspace_fit_is_finite_without_ridge_regularization():
    source = FeatureSpaceIdentity(
        space_id="source-collinear",
        model_id="source-model",
        model_revision="source@sha256:3",
        representation_id="hidden",
        feature_names=("x", "two_x", "noise"),
        architecture="transformer",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:1",
        session_id="session",
        checkpoint="step",
    )
    target = FeatureSpaceIdentity(
        space_id="target-collinear",
        model_id="target-model",
        model_revision="target@sha256:3",
        representation_id="hidden",
        feature_names=("u",),
        architecture="ssm",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:1",
        session_id="session",
        checkpoint="step",
    )
    spec = FeatureCorrespondenceSpec(
        study_id="rank-deficient",
        source_space=source,
        target_space=target,
        source_features=("x", "two_x"),
        target_features=("u",),
        kind=CorrespondenceKind.SUBSPACE,
        discovery_partition_id="d",
        validation_partition_id="v",
        declared_context_differences=("model_id", "model_revision", "architecture"),
        ridge_alpha=0.0,
        policy=FeatureCorrespondencePolicy(min_discovery_examples=3, min_validation_examples=2),
    )
    discovery = tuple(
        FeaturePairExample(
            example_id=f"d{index}",
            semantic_trial_id=f"trial-d{index}",
            split=CorrespondenceSplit.DISCOVERY,
            partition_id="d",
            source_activation=np.asarray([value, 2.0 * value, index]),
            target_activation=np.asarray([3.0 * value + 0.25]),
        )
        for index, value in enumerate((0.1, 0.6, 1.3, 2.2))
    )
    candidate = fit_feature_correspondence_candidate(spec, discovery)
    assert np.isfinite(np.asarray(candidate.mapping_matrix)).all()
    assert np.isfinite(np.asarray(candidate.intercept)).all()
    assert np.allclose(candidate.predict([0.8, 1.6]), [2.65], atol=1e-8)


def test_large_feature_universe_samples_random_controls_without_enumerating_all_subsets():
    feature_names = tuple(f"feature_{index:05d}" for index in range(10_000))
    source = FeatureSpaceIdentity(
        space_id="source-large",
        model_id="source-model",
        model_revision="source@sha256:4",
        representation_id="sae-features",
        feature_names=feature_names,
        architecture="transformer",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:1",
        session_id="session",
        checkpoint="step",
    )
    target = FeatureSpaceIdentity(
        space_id="target-large",
        model_id="target-model",
        model_revision="target@sha256:4",
        representation_id="sae-features",
        feature_names=("target",),
        architecture="ssm",
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:1",
        session_id="session",
        checkpoint="step",
    )
    spec = FeatureCorrespondenceSpec(
        study_id="large-random-controls",
        source_space=source,
        target_space=target,
        source_features=("feature_00000", "feature_00001", "feature_00002"),
        target_features=("target",),
        kind=CorrespondenceKind.SUBSPACE,
        discovery_partition_id="d",
        validation_partition_id="v",
        declared_context_differences=("model_id", "model_revision", "architecture"),
        random_controls=8,
        seed=7,
    )
    controls = _random_source_feature_sets(spec)
    assert len(controls) == 8
    assert len(set(controls)) == 8
    assert all(len(control) == 3 for control in controls)
    assert all(frozenset(control) != frozenset(spec.source_features) for control in controls)
    assert controls == _random_source_feature_sets(spec)


def test_correspondence_artifact_round_trip_and_tamper_detection(tmp_path):
    spec, examples, evaluator = _study_inputs()
    result = run_feature_correspondence_study(spec, examples, evaluator=evaluator)
    assert result.promotion.passed
    path = tmp_path / "correspondence.json"
    write_correspondence_artifact(result, path)
    loaded = read_correspondence_artifact(path)
    assert loaded["study_fingerprint"] == result.study_fingerprint

    payload = json.loads(path.read_text())
    payload["result"]["validation_metrics"]["predictive_r2"] = -99.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash mismatch"):
        read_correspondence_artifact(path)
