"""Known-shift benchmark for discovery/validation evidence-pack logic."""

from __future__ import annotations

import torch
from torch import nn

from neuros_mechint.adapters import PyTorchAdapter
from neuros_mechint.core import EvidenceTier, OutputMetric

from .evidence_pack import (
    EvidenceExample,
    EvidencePackPolicy,
    EvidencePackSpec,
    EvidenceSplit,
    discover_ablation_effect_candidate,
    run_adapter_evidence_pack,
)


class DiscoveryShiftMLP(nn.Module):
    """Discovery examples use route A while validation examples use route B."""

    def __init__(self) -> None:
        super().__init__()
        self.route_a = nn.Linear(2, 2, bias=False)
        self.route_b = nn.Linear(2, 2, bias=False)
        self.nuisance = nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            self.route_a.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 0.0]]))
            self.route_b.weight.copy_(torch.tensor([[0.0, 1.0], [0.0, 0.0]]))
            self.nuisance.weight.zero_()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        route_a = self.route_a(inputs)
        route_b = self.route_b(inputs)
        nuisance = self.nuisance(inputs)
        return route_a[..., :1] + route_b[..., :1] + nuisance[..., :1]


def run_evidence_pack_generalization_benchmark(seed: int = 0) -> dict[str, object]:
    """Verify that discovery success cannot masquerade as held-out validation."""

    model = DiscoveryShiftMLP()
    adapter = PyTorchAdapter(model)
    metric = OutputMetric(lambda output: output.mean(), name="mean_output")
    examples = (
        EvidenceExample("discover-1", torch.tensor([[1.0, 0.0]]), EvidenceSplit.DISCOVERY),
        EvidenceExample("discover-2", torch.tensor([[2.0, 0.0]]), EvidenceSplit.DISCOVERY),
        EvidenceExample("validate-1", torch.tensor([[0.0, 1.0]]), EvidenceSplit.VALIDATION),
        EvidenceExample("validate-2", torch.tensor([[0.0, 2.0]]), EvidenceSplit.VALIDATION),
    )
    seen_by_discovery: list[str] = []

    def _discover(adapter_, discovery_examples, targets):
        seen_by_discovery.extend(item.example_id for item in discovery_examples)
        return discover_ablation_effect_candidate(
            adapter_,
            discovery_examples,
            targets,
            metric=metric,
            k=1,
            name="discovery-top-route",
        )

    result = run_adapter_evidence_pack(
        spec=EvidencePackSpec(
            pack_id="ground-truth-held-out-shift",
            model_id="DiscoveryShiftMLP",
            dataset_id="synthetic:discovery-validation-shift",
            metric_name=metric.name,
            target_universe=("route_a", "route_b", "nuisance"),
            discovery_method="single-target-zero-ablation",
            intervention_baselines=("zero", "mean"),
            random_trials=100,
            seed=seed,
            evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
        ),
        adapter=adapter,
        metric=metric,
        examples=examples,
        discover_candidate=_discover,
        pack_policy=EvidencePackPolicy(
            min_validation_examples=2,
            min_validation_pass_rate=0.8,
            min_validation_joint_median=0.5,
            max_joint_generalization_drop=0.25,
            require_multiple_intervention_baselines=True,
            bootstrap_samples=200,
        ),
    )

    discovery_ids = {"discover-1", "discover-2"}
    validation_ids = {"validate-1", "validate-2"}
    passed = (
        set(seen_by_discovery) == discovery_ids
        and not (set(seen_by_discovery) & validation_ids)
        and result.candidate.targets == ("route_a",)
        and result.discovery_aggregate.pass_rate == 1.0
        and result.validation_aggregate.pass_rate == 0.0
        and result.promotion.passed is False
        and len(result.mean_ablation_references) == 3
    )
    return {
        "candidate": result.candidate.to_dict(),
        "discovery_aggregate": result.discovery_aggregate.to_dict(),
        "discovery_saw_example_ids": seen_by_discovery,
        "mean_ablation_references": dict(result.mean_ablation_references),
        "passed": passed,
        "promotion": result.promotion.to_dict(),
        "study_fingerprint": result.study_fingerprint,
        "validation_aggregate": result.validation_aggregate.to_dict(),
    }
