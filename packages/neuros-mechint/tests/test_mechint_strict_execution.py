"""Fail-closed controls for opt-in strict mechanistic experiment execution."""

import pytest
import torch
from torch import nn

from neuros_mechint import (
    AblationIntervention,
    ComponentRef,
    CounterfactualPair,
    MechanisticExperiment,
    OutputMetric,
    PatchIntervention,
    PyTorchAdapter,
)
from neuros_mechint.benchmarks import GroundTruthCausalMLP, make_ground_truth_pair


def _experiment(model=None, *, metadata=None, strict=True):
    pair = make_ground_truth_pair()
    return MechanisticExperiment(
        adapter=PyTorchAdapter(model if model is not None else GroundTruthCausalMLP().eval()),
        pair=CounterfactualPair(pair.clean, pair.corrupted, metadata=metadata or {}),
        metric=OutputMetric(lambda output: output.mean(), name="mean_output"),
        experiment_name="strict-known-route",
        model_id="synthetic:ground-truth",
        dataset_id="synthetic:known-route",
        strict_execution=strict,
        seed=31,
    )


def test_strict_recovers_known_route_and_fingerprints_full_intervention_protocol():
    experiment = _experiment()
    intervention = PatchIntervention(ComponentRef("source"))
    result = experiment.run([intervention], controls=[PatchIntervention(ComponentRef("nuisance"))])
    assert result.effects[0].effect == pytest.approx(1.0)
    assert result.controls[0].effect == pytest.approx(0.0)
    assert result.manifest.method_version == "2"
    assert result.manifest.dataset_hash
    assert result.manifest.model_hash
    identity = result.manifest.parameters["intervention_protocol"][0]
    assert identity["type"].endswith("PatchIntervention")
    assert identity["parameters"]["component"]["path"] == "source"
    assert result.manifest.parameters["strict_execution"] is True


def test_strict_identity_changes_when_intervention_semantics_change():
    left = _experiment().run([AblationIntervention(ComponentRef("source"), mode="zero")])
    right = _experiment().run([AblationIntervention(ComponentRef("source"), mode="mean")])
    assert left.manifest.scientific_fingerprint != right.manifest.scientific_fingerprint
    assert left.manifest.parameters["intervention_protocol"][0]["parameters"]["mode"] == "zero"
    assert right.manifest.parameters["intervention_protocol"][0]["parameters"]["mode"] == "mean"


def test_strict_rejects_training_model_and_preserves_mode():
    model = GroundTruthCausalMLP()
    with pytest.raises(ValueError, match=r"model.eval"):
        _experiment(model).run([PatchIntervention(ComponentRef("source"))])
    assert model.training is True


class RandomRoute(nn.Module):
    def __init__(self):
        super().__init__()
        self.route = nn.Identity()

    def forward(self, x):
        return self.route(x[:, :1] + torch.randn_like(x[:, :1]))


def test_strict_rejects_nondeterministic_baseline():
    with pytest.raises(ValueError, match="non-deterministic baseline"):
        _experiment(RandomRoute().eval()).run([PatchIntervention(ComponentRef("route"))])


class StatefulRoute(nn.Module):
    def __init__(self):
        super().__init__()
        self.route = nn.Identity()
        self.register_buffer("calls", torch.zeros((), dtype=torch.long))

    def forward(self, x):
        self.calls.add_(1)
        return self.route(x[:, :1])


def test_strict_rejects_mutating_model_state():
    model = StatefulRoute().eval()
    with pytest.raises(ValueError, match="model-state mutation"):
        _experiment(model).run([PatchIntervention(ComponentRef("route"))])


def test_strict_rejects_nonhashable_data():
    with pytest.raises(ValueError, match="content-hashable"):
        _experiment(metadata={"opaque": object()}).run(
            [PatchIntervention(ComponentRef("source"))]
        )


def test_strict_execution_does_not_consume_global_rng():
    torch.manual_seed(123)
    before = torch.random.get_rng_state().clone()
    _experiment().run([PatchIntervention(ComponentRef("source"))])
    assert torch.equal(before, torch.random.get_rng_state())


def test_failed_strict_execution_does_not_consume_global_rng():
    torch.manual_seed(127)
    before = torch.random.get_rng_state().clone()
    with pytest.raises(ValueError, match="non-deterministic"):
        _experiment(RandomRoute().eval()).run([PatchIntervention(ComponentRef("route"))])
    assert torch.equal(before, torch.random.get_rng_state())


def test_legacy_mode_is_still_available_without_a_new_protocol():
    report = _experiment(strict=False).run([PatchIntervention(ComponentRef("source"))])
    assert report.manifest.method_version == "1"
    assert "intervention_protocol" not in report.manifest.parameters
