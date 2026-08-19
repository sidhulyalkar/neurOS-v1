import pytest
import torch

from neuros_mechint.adapters import SAELensFeatureAdapter
from neuros_mechint.benchmarks import (
    CircuitCandidate,
    FaithfulnessPolicy,
    evaluate_circuit_faithfulness,
    evaluate_sae_feature_faithfulness,
    run_circuit_faithfulness_benchmark,
)


def test_generic_faithfulness_prefers_known_circuit_to_same_size_controls():
    weights = {"a": 0.6, "b": 0.4, "c": 0.0}

    def subset_metric(targets):
        return sum(weights[target] for target in targets)

    report = evaluate_circuit_faithfulness(
        all_targets=("a", "b", "c"),
        candidate=CircuitCandidate(name="known", targets=("a", "b")),
        subset_metric=subset_metric,
        random_trials=100,
        policy=FaithfulnessPolicy(
            min_sufficiency_fraction=0.99,
            min_necessity_fraction=0.99,
            min_random_percentile=0.99,
        ),
    )
    assert report.sufficiency_fraction == pytest.approx(1.0)
    assert report.necessity_fraction == pytest.approx(1.0)
    assert report.sufficiency_random_percentile == pytest.approx(1.0)
    assert report.necessity_random_percentile == pytest.approx(1.0)
    assert report.joint_random_percentile == pytest.approx(1.0)
    assert report.passed is True


def test_random_percentile_requires_strict_superiority_not_ties():
    weights = {"a": 1.0 / 3.0, "b": 1.0 / 3.0, "c": 1.0 / 3.0}

    def subset_metric(targets):
        return sum(weights[target] for target in targets)

    report = evaluate_circuit_faithfulness(
        all_targets=("a", "b", "c"),
        candidate=CircuitCandidate(name="tied", targets=("a",)),
        subset_metric=subset_metric,
        random_trials=100,
    )
    assert report.joint_faithfulness == pytest.approx(1.0 / 3.0)
    assert report.joint_random_percentile == pytest.approx(0.0)
    assert report.passed is False


def test_faithfulness_ground_truth_gate_recovers_known_route():
    benchmark = run_circuit_faithfulness_benchmark(seed=7)
    assert benchmark["passed"] is True
    assert benchmark["report"]["sufficiency_fraction"] == pytest.approx(1.0)
    assert benchmark["report"]["necessity_fraction"] == pytest.approx(1.0)
    assert benchmark["report"]["joint_random_percentile"] == pytest.approx(1.0)


class _IdentitySAE:
    def encode(self, activations):
        return activations.clone()

    def decode(self, features):
        return features.clone()


def test_sae_faithfulness_uses_reconstruction_as_explicit_baseline():
    activations = torch.tensor([[0.6, 0.4, 0.0]])
    report = evaluate_sae_feature_faithfulness(
        adapter=SAELensFeatureAdapter(_IdentitySAE()),
        activations=activations,
        scorer=lambda value: value.sum(),
        target_features=(0, 1, 2),
        candidate_features=(0, 1),
        random_trials=100,
        policy=FaithfulnessPolicy(
            min_sufficiency_fraction=0.99,
            min_necessity_fraction=0.99,
            min_random_percentile=0.99,
        ),
    )
    assert report.sufficiency_fraction == pytest.approx(1.0)
    assert report.necessity_fraction == pytest.approx(1.0)
    assert report.metadata["reconstruction_gap"] == pytest.approx(0.0)
    assert report.joint_random_percentile == pytest.approx(1.0)
    assert report.passed is True
