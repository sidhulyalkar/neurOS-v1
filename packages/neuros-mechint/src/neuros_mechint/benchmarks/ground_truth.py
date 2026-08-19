"""Ground-truth systems for validating localization methods scientifically."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass

import torch
from torch import nn

from neuros_mechint.adapters import PyTorchAdapter
from neuros_mechint.core import (
    ComponentRef,
    CounterfactualPair,
    EvidenceTier,
    MechanisticExperiment,
    OutputMetric,
    PatchIntervention,
)


class GroundTruthCausalMLP(nn.Module):
    """Tiny network with a known causal route and an explicit nuisance route."""

    def __init__(self) -> None:
        super().__init__()
        self.source = nn.Linear(2, 2, bias=False)
        self.causal = nn.Linear(2, 1, bias=False)
        self.nuisance = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.source.weight.copy_(torch.eye(2))
            self.causal.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.nuisance.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.source(x)
        return self.causal(hidden) + self.nuisance(hidden)


@dataclass(frozen=True, slots=True)
class GroundTruthPair:
    clean: torch.Tensor
    corrupted: torch.Tensor
    expected_causal_components: tuple[str, ...]
    expected_nuisance_components: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LocalizationReport:
    precision_at_k: float
    recall_at_k: float
    average_precision: float
    causal_min_abs_score: float
    nuisance_max_abs_score: float
    separation_margin: float
    passed_separation: bool

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)


def make_ground_truth_pair() -> GroundTruthPair:
    return GroundTruthPair(
        clean=torch.tensor([[1.0, 0.0]]),
        corrupted=torch.tensor([[0.0, 0.0]]),
        expected_causal_components=("source", "causal"),
        expected_nuisance_components=("nuisance",),
    )


def _ranking(scores: Mapping[str, float]) -> list[str]:
    return sorted(scores, key=lambda key: abs(scores[key]), reverse=True)


def localization_precision_at_k(
    scores: Mapping[str, float],
    expected: tuple[str, ...],
    k: int,
) -> float:
    ranked = _ranking(scores)[:k]
    expected_set = set(expected)
    return sum(item in expected_set for item in ranked) / max(1, k)


def localization_recall_at_k(
    scores: Mapping[str, float],
    expected: tuple[str, ...],
    k: int,
) -> float:
    expected_set = set(expected)
    ranked = set(_ranking(scores)[:k])
    return len(ranked & expected_set) / max(1, len(expected_set))


def localization_average_precision(
    scores: Mapping[str, float],
    expected: tuple[str, ...],
) -> float:
    expected_set = set(expected)
    if not expected_set:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for rank, component in enumerate(_ranking(scores), start=1):
        if component in expected_set:
            hits += 1
            precision_sum += hits / rank
    return precision_sum / len(expected_set)


def evaluate_localization(
    scores: Mapping[str, float],
    pair: GroundTruthPair,
) -> LocalizationReport:
    k = len(pair.expected_causal_components)
    causal_scores = [abs(scores.get(name, 0.0)) for name in pair.expected_causal_components]
    nuisance_scores = [
        abs(scores.get(name, 0.0)) for name in pair.expected_nuisance_components
    ]
    causal_min = min(causal_scores, default=0.0)
    nuisance_max = max(nuisance_scores, default=0.0)
    margin = causal_min - nuisance_max
    return LocalizationReport(
        precision_at_k=localization_precision_at_k(
            scores, pair.expected_causal_components, k
        ),
        recall_at_k=localization_recall_at_k(
            scores, pair.expected_causal_components, k
        ),
        average_precision=localization_average_precision(
            scores, pair.expected_causal_components
        ),
        causal_min_abs_score=causal_min,
        nuisance_max_abs_score=nuisance_max,
        separation_margin=margin,
        passed_separation=margin > 0.0,
    )


def run_ground_truth_benchmark(seed: int = 0) -> dict[str, object]:
    """Run activation patching where the true mechanism is known in advance."""

    model = GroundTruthCausalMLP()
    pair = make_ground_truth_pair()
    experiment = MechanisticExperiment(
        adapter=PyTorchAdapter(model),
        pair=CounterfactualPair(
            clean=pair.clean,
            corrupted=pair.corrupted,
            metadata={"benchmark": "ground_truth_causal_mlp"},
        ),
        metric=OutputMetric(lambda output: output.mean(), name="mean_output"),
        experiment_name="ground-truth-localization",
        model_id="GroundTruthCausalMLP",
        dataset_id="synthetic:ground-truth-causal-mlp",
        seed=seed,
        evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
    )
    result = experiment.run(
        [
            PatchIntervention(ComponentRef("source")),
            PatchIntervention(ComponentRef("causal")),
            PatchIntervention(ComponentRef("nuisance")),
        ]
    )
    scores = {effect.component: effect.effect for effect in result.effects}
    localization = evaluate_localization(scores, pair)
    return {
        "experiment": result.to_dict(),
        "scores": scores,
        "localization": localization.to_dict(),
        "expected_causal_components": list(pair.expected_causal_components),
        "expected_nuisance_components": list(pair.expected_nuisance_components),
    }
