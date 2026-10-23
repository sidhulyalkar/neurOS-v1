"""Adversarial contracts for PyTorch tensor-leaf and call-specific interventions."""

import pytest
import torch
from torch import nn

from neuros_mechint import ModuleOutputTarget, PyTorchAdapter, stable_hash


class StructuredBlock(nn.Module):
    def forward(self, x):
        return x * 2, {"state": [x * 3, x * 0]}


class StructuredModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.structured = StructuredBlock()

    def forward(self, x):
        activation, info = self.structured(x)
        return activation + info["state"][0]


class RepeatedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Identity()

    def forward(self, x):
        return self.shared(x * 2) + self.shared(x * 3)


class RecurrentModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3, hidden_size=4, batch_first=True)
        self.readout = nn.Linear(4, 1)

    def forward(self, x):
        output, (hidden, cell) = self.lstm(x)
        del cell
        return self.readout(output[:, -1, :] + hidden[-1])


class OptionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.optional = nn.Identity()

    def forward(self, x):
        return self.optional(x) if x.sum() > 0 else x


def test_nested_tensor_capture_and_disjoint_replacement_preserves_original():
    model = StructuredModel().eval()
    adapter = PyTorchAdapter(model, targets={
        "activation": ModuleOutputTarget("structured", selector=(0,)),
        "hidden": ModuleOutputTarget("structured", selector=(1, "state", 0)),
    })
    x = torch.tensor([[1.0, 2.0]])
    before = adapter.forward(x)
    captured = adapter.capture_outputs(x, ["activation", "hidden"])
    assert torch.equal(captured["activation"], 2 * x)
    assert torch.equal(captured["hidden"], 3 * x)
    patched = adapter.forward_with_replacements(
        x, {"activation": torch.zeros_like(x), "hidden": torch.ones_like(x)}
    )
    assert torch.equal(patched, torch.ones_like(x))
    assert torch.equal(before, adapter.forward(x))
    assert not model.structured._forward_hooks


def test_nested_selectors_fail_closed_and_clear_hooks():
    model = StructuredModel()
    targets = {
        "missing_index": ModuleOutputTarget("structured", selector=(9,)),
        "missing_key": ModuleOutputTarget("structured", selector=(1, "absent")),
        "wrong_type": ModuleOutputTarget("structured", selector=("state",)),
    }
    adapter = PyTorchAdapter(model, targets=targets)
    with pytest.raises(KeyError, match="out of bounds"):
        adapter.capture_outputs(torch.ones(1, 2), ["missing_index"])
    with pytest.raises(KeyError, match="absent"):
        adapter.capture_outputs(torch.ones(1, 2), ["missing_key"])
    with pytest.raises(TypeError, match="selector"):
        adapter.capture_outputs(torch.ones(1, 2), ["wrong_type"])
    assert not model.structured._forward_hooks


def test_repeated_module_calls_have_independent_occurrence_targets():
    model = RepeatedModel().eval()
    adapter = PyTorchAdapter(model, targets={
        "first": ModuleOutputTarget("shared", occurrence=0, expected_calls=2),
        "second": ModuleOutputTarget("shared", occurrence=1, expected_calls=2),
    })
    x = torch.tensor([[2.0]])
    captured = adapter.capture_outputs(x, ["first", "second"])
    assert captured["first"].item() == 4.0
    assert captured["second"].item() == 6.0
    first_only = adapter.forward_with_replacements(x, {"first": torch.zeros_like(x)})
    second_only = adapter.forward_with_replacements(x, {"second": torch.zeros_like(x)})
    assert first_only.item() == 6.0
    assert second_only.item() == 4.0
    with pytest.raises(ValueError, match="called 2 times"):
        adapter.capture_outputs(x, ["shared"])
    assert not model.shared._forward_hooks


def test_call_count_drift_rejected_after_hook_cleanup():
    model = RepeatedModel()
    adapter = PyTorchAdapter(model, targets={
        "first": ModuleOutputTarget("shared", occurrence=0, expected_calls=3),
    })
    with pytest.raises(ValueError, match="expected 3"):
        adapter.forward_with_replacements(torch.ones(1, 1), {
            "first": torch.zeros(1, 1),
        })
    assert not model.shared._forward_hooks


def test_overlapping_selectors_are_rejected_without_installed_hooks():
    model = StructuredModel()
    adapter = PyTorchAdapter(model, targets={
        "outer": ModuleOutputTarget("structured", selector=(1,)),
        "inner": ModuleOutputTarget("structured", selector=(1, "state", 0)),
    })
    with pytest.raises(ValueError, match="overlapping selectors"):
        adapter.forward_with_replacements(torch.ones(1, 2), {
            "outer": torch.zeros(1, 2),
            "inner": torch.zeros(1, 2),
        })
    assert not model.structured._forward_hooks


def test_incompatible_tensor_shape_rejected_and_hooks_removed():
    model = StructuredModel()
    adapter = PyTorchAdapter(model, targets={
        "hidden": ModuleOutputTarget("structured", selector=(1, "state", 0)),
    })
    with pytest.raises(ValueError, match="shape"):
        adapter.forward_with_replacements(torch.ones(1, 2), {
            "hidden": torch.zeros(2, 2),
        })
    assert not model.structured._forward_hooks


def test_aliases_are_immutable_and_bound_to_effective_model_fingerprint():
    model = StructuredModel()
    targets = {"value": ModuleOutputTarget("structured", selector=(0,))}
    adapter = PyTorchAdapter(model, targets=targets)
    targets["value"] = ModuleOutputTarget("structured", selector=(1, "state", 0))
    assert adapter.targets["value"].selector == (0,)
    with pytest.raises(TypeError):
        adapter.targets["new"] = ModuleOutputTarget("structured")
    other = PyTorchAdapter(model, targets=targets)
    assert stable_hash(adapter.model_fingerprint_payload()) != stable_hash(
        other.model_fingerprint_payload()
    )


def test_invalid_target_definitions_are_rejected():
    with pytest.raises(ValueError, match="expected_calls"):
        ModuleOutputTarget("shared", occurrence=1)
    with pytest.raises(ValueError, match="smaller"):
        ModuleOutputTarget("shared", occurrence=2, expected_calls=2)
    with pytest.raises(ValueError, match="selector"):
        ModuleOutputTarget("foo", selector=(-1,))
    with pytest.raises(ValueError, match="selector"):
        ModuleOutputTarget("foo", selector=(True,))
    with pytest.raises(TypeError, match="selector"):
        ModuleOutputTarget("foo", selector="state")
    with pytest.raises(KeyError, match="unknown module"):
        PyTorchAdapter(RepeatedModel(), targets={"x": ModuleOutputTarget("missing")})
    with pytest.raises(ValueError, match="collides"):
        PyTorchAdapter(RepeatedModel(), targets={"shared": ModuleOutputTarget("shared")})


def test_original_tensor_module_path_remains_supported():
    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(2)
    adapter = PyTorchAdapter(model)
    x = torch.ones(1, 1)
    assert adapter.capture_outputs(x, ["0"])["0"].item() == 2.0
    assert adapter.forward_with_replacements(x, {"0": torch.zeros_like(x)}).item() == 0.0
    assert not model[0]._forward_hooks


def test_conditional_module_not_called_is_rejected():
    model = OptionalModel()
    adapter = PyTorchAdapter(model, targets={
        "sometimes": ModuleOutputTarget("optional"),
    })
    with pytest.raises(ValueError, match="called 0 times"):
        adapter.capture_outputs(-torch.ones(1, 1), ["sometimes"])
    assert not model.optional._forward_hooks


def test_real_lstm_supports_tuple_output_and_nested_hidden_state():
    model = RecurrentModel().eval()
    adapter = PyTorchAdapter(model, targets={
        "lstm_output": ModuleOutputTarget("lstm", selector=(0,)),
        "lstm_hidden": ModuleOutputTarget("lstm", selector=(1, 0)),
    })
    x = torch.ones(2, 4, 3)
    baseline = adapter.forward(x)
    captured = adapter.capture_outputs(x, ["lstm_output", "lstm_hidden"])
    assert captured["lstm_output"].shape == (2, 4, 4)
    assert captured["lstm_hidden"].shape == (1, 2, 4)
    patched = adapter.forward_with_replacements(x, {
        "lstm_output": torch.zeros_like(captured["lstm_output"]),
        "lstm_hidden": torch.zeros_like(captured["lstm_hidden"]),
    })
    assert baseline.shape == patched.shape == (2, 1)
    assert not torch.allclose(baseline, patched)
    assert not model.lstm._forward_hooks


def test_named_target_works_with_existing_mechanistic_experiment_api():
    from neuros_mechint import (
        ComponentRef,
        CounterfactualPair,
        MechanisticExperiment,
        OutputMetric,
        PatchIntervention,
    )

    model = StructuredModel().eval()
    adapter = PyTorchAdapter(model, targets={
        "hidden": ModuleOutputTarget("structured", selector=(1, "state", 0)),
    })
    experiment = MechanisticExperiment(
        adapter=adapter,
        pair=CounterfactualPair(
            clean=torch.tensor([[1.0, 1.0]]),
            corrupted=torch.tensor([[0.0, 0.0]]),
        ),
        metric=OutputMetric(lambda x: x.mean(), name="mean"),
        experiment_name="nested-patch",
        model_id="StructuredModel",
    )
    result = experiment.run([PatchIntervention(ComponentRef("hidden"))])
    assert result.clean_metric == pytest.approx(5.0)
    assert result.corrupted_metric == pytest.approx(0.0)
    assert result.effects[0].effect == pytest.approx(3.0)
    assert result.manifest.model_hash is not None
    assert not model.structured._forward_hooks
