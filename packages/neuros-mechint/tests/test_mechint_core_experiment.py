import torch

from neuros_mechint import (
    ComponentRef,
    CounterfactualPair,
    EvidenceTier,
    MechanisticExperiment,
    OutputMetric,
    PatchIntervention,
    PyTorchAdapter,
)
from neuros_mechint.benchmarks import GroundTruthCausalMLP, make_ground_truth_pair


def test_mechanistic_experiment_recovers_causal_route_and_records_provenance():
    model = GroundTruthCausalMLP().eval()
    pair = make_ground_truth_pair()
    result = MechanisticExperiment(
        adapter=PyTorchAdapter(model),
        pair=CounterfactualPair(pair.clean, pair.corrupted),
        metric=OutputMetric(lambda output: output.mean(), name="mean"),
        experiment_name="known-route",
        model_id="ground-truth",
        evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
    ).run(
        [
            PatchIntervention(ComponentRef("source")),
            PatchIntervention(ComponentRef("causal")),
        ],
        controls=[PatchIntervention(ComponentRef("nuisance"))],
    )

    effects = {item.component: item.effect for item in result.effects}
    assert effects == {"source": 1.0, "causal": 1.0}
    assert result.controls[0].effect == 0.0
    assert result.specificity_gap == 1.0
    assert result.manifest.evidence_tier is EvidenceTier.SCIENTIFIC_SYNTHETIC
    assert result.manifest.model_hash is not None
    assert result.manifest.dataset_hash is not None
    assert result.manifest.benchmark is not None
    assert result.manifest.benchmark.benchmark_id == "mechint:known-route"


def test_interventions_are_independent_not_cumulative():
    model = GroundTruthCausalMLP().eval()
    pair = make_ground_truth_pair()
    result = MechanisticExperiment(
        adapter=PyTorchAdapter(model),
        pair=CounterfactualPair(pair.clean, pair.corrupted),
        metric=OutputMetric(lambda output: output.mean()),
        experiment_name="independent",
        model_id="ground-truth",
    ).run(
        [
            PatchIntervention(ComponentRef("source")),
            PatchIntervention(ComponentRef("nuisance")),
        ]
    )
    assert [item.intervened_metric for item in result.effects] == [1.0, 0.0]
