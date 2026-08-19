"""Known-circuit scientific gate for the v0.5 faithfulness benchmark."""

from __future__ import annotations

from neuros_mechint.adapters import PyTorchAdapter
from neuros_mechint.core import OutputMetric

from .faithfulness import (
    CircuitCandidate,
    FaithfulnessPolicy,
    evaluate_adapter_circuit_faithfulness,
)
from .ground_truth import GroundTruthCausalMLP, make_ground_truth_pair


def run_circuit_faithfulness_benchmark(seed: int = 0) -> dict[str, object]:
    """Verify that the known causal route beats same-size random circuits."""

    model = GroundTruthCausalMLP()
    pair = make_ground_truth_pair()
    report = evaluate_adapter_circuit_faithfulness(
        adapter=PyTorchAdapter(model),
        inputs=pair.clean,
        metric=OutputMetric(lambda output: output.mean(), name="mean_output"),
        all_targets=("source", "causal", "nuisance"),
        candidate=CircuitCandidate(
            name="known-causal-route",
            targets=pair.expected_causal_components,
            source="synthetic-ground-truth",
        ),
        ablation_mode="zero",
        random_trials=100,
        seed=seed,
        policy=FaithfulnessPolicy(
            min_sufficiency_fraction=0.99,
            min_necessity_fraction=0.99,
            min_random_percentile=0.99,
        ),
    )
    return {
        "benchmark": "ground_truth_circuit_faithfulness",
        "expected_circuit": list(pair.expected_causal_components),
        "report": report.to_dict(),
        "passed": report.passed,
    }
