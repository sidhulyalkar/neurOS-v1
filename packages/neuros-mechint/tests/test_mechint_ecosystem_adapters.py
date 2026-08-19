from contextlib import contextmanager

import pytest
import torch
from torch import nn

from neuros_mechint.adapters import (
    CircuitTracerAdapter,
    NNsightAdapter,
    NNsightTarget,
    SAEReconstructionAudit,
    SAELensFeatureAdapter,
    TransformerLensAdapter,
)


class FakeTransformerLens(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def _compute(self, x, hooks=(), names=()):
        hooks = dict(hooks)
        cache = {}
        source = x * self.scale
        if "hook_source" in hooks:
            source = hooks["hook_source"](source, object())
        if "hook_source" in names:
            cache["hook_source"] = source
        causal = source[..., :1]
        if "hook_causal" in hooks:
            causal = hooks["hook_causal"](causal, object())
        if "hook_causal" in names:
            cache["hook_causal"] = causal
        return causal, cache

    def forward(self, x):
        return self._compute(x)[0]

    def run_with_cache(self, x, *, names_filter):
        return self._compute(x, names=tuple(names_filter))

    def run_with_hooks(self, x, *, fwd_hooks):
        return self._compute(x, hooks=tuple(fwd_hooks))[0]


def test_transformer_lens_adapter_uses_native_cache_and_hook_contract():
    adapter = TransformerLensAdapter(FakeTransformerLens())
    x = torch.tensor([[2.0, 4.0]])
    cache = adapter.capture_outputs(x, ["hook_source", "hook_causal"])
    assert cache["hook_source"].tolist() == [[2.0, 4.0]]
    assert cache["hook_causal"].tolist() == [[2.0]]
    output = adapter.forward_with_replacements(
        x,
        {"hook_causal": torch.zeros_like(cache["hook_causal"])},
    )
    assert output.item() == 0.0


class _SavedTensor:
    def __init__(self, value):
        self.value = value

    def save(self):
        return self.value.clone()

    def __setitem__(self, key, value):
        self.value[key] = value


class _Envoy:
    def __init__(self):
        self._output = _SavedTensor(torch.zeros(1, 1))

    @property
    def output(self):
        return self._output


class _Underlying(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Identity()
        self.layer2 = nn.Identity()


class FakeNNsight:
    def __init__(self):
        self._module = _Underlying()
        self.layer1 = _Envoy()
        self.layer2 = _Envoy()

    @contextmanager
    def trace(self, x):
        self.layer1._output = _SavedTensor(x.clone())
        self.layer2._output = _SavedTensor((x * 3).clone())
        yield self

    @property
    def output(self):
        return self.layer2.output


def test_nnsight_adapter_saves_and_replaces_trace_outputs():
    adapter = NNsightAdapter(FakeNNsight())
    x = torch.tensor([[2.0]])
    cache = adapter.capture_outputs(x, ["layer1", "layer2"])
    assert cache["layer1"].item() == 2.0
    assert cache["layer2"].item() == 6.0
    output = adapter.forward_with_replacements(x, {"layer2": torch.tensor([[9.0]])})
    assert output.item() == 9.0
    assert NNsightTarget.parse("transformer.h.0::0").output_index == 0


class IdentitySAE:
    def encode(self, activations):
        return activations.clone()

    def decode(self, features):
        return features.clone()


def test_saelens_adapter_reports_reconstruction_and_feature_subset_effects():
    adapter = SAELensFeatureAdapter(IdentitySAE())
    activations = torch.tensor([[1.0, 2.0, 3.0]])
    audit = adapter.reconstruction_audit(activations, lambda value: value.sum())
    assert isinstance(audit, SAEReconstructionAudit)
    assert audit.reconstruction_gap == 0.0
    edited = adapter.reconstruct_with_feature_subset(
        activations,
        target_features=[0, 1, 2],
        retained_features=[0, 2],
    )
    assert edited.tolist() == [[1.0, 0.0, 3.0]]


class FakeGraph:
    def __init__(self):
        self.active_features = torch.tensor([[0, 0, 2], [1, 3, 4]])
        self.adjacency_matrix = torch.zeros(5, 5)
        self.adjacency_matrix[-1, 0] = 0.7
        self.adjacency_matrix[-1, 1] = -0.2
        self.logit_probabilities = torch.tensor([1.0])


def test_circuit_tracer_adapter_keeps_attribution_separate_from_causation():
    adapter = CircuitTracerAdapter()
    summary = adapter.summarize_graph(FakeGraph())
    assert summary.feature_scores["feature:L0:P0:F2"] == pytest.approx(0.7)
    assert summary.feature_scores["feature:L1:P3:F4"] == pytest.approx(-0.2)
    candidate = adapter.candidate(FakeGraph(), k=1)
    assert candidate.targets == ("feature:L0:P0:F2",)
    assert candidate.source == "circuit-tracer-attribution"
