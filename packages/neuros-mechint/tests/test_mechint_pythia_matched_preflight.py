"""Offline model stubs for the separately versioned matched-repetition study."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn


SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "25_pythia_matched_repetition_preflight.py"
spec = importlib.util.spec_from_file_location("pythia_matched_repetition_preflight", SCRIPT)
assert spec is not None and spec.loader is not None
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class FakeTokenizer:
    def encode(self, value, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [pilot.WORDS.index(value.strip()) + 3]


class FrozenPriorModel(nn.Module):
    def __init__(self, *, selective):
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.selective = selective
        self.n_forward_calls = 0
        self.seen = []

    def forward(self, *, input_ids, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        self.n_forward_calls += 1
        packed = input_ids[0].tolist()
        self.seen.append(tuple(packed))
        d = packed[3]
        is_repeated = packed[4:7] == packed[:3]
        response = 5.0 if (not self.selective or is_repeated) else -1.0
        logits = torch.zeros((1, len(packed), 128), dtype=torch.float32)
        logits[0, -1, d] = response
        return SimpleNamespace(logits=logits)


def test_prior_only_positive_margin_fails_matched_control_gate(tmp_path):
    model = FrozenPriorModel(selective=False)
    outcome = pilot.run_discovery_only(
        pretrained=model,
        tokenizer=FakeTokenizer(),
        revision="a" * 40,
        output_dir=tmp_path,
    )
    assert outcome["study_status"] == "DISCOVERY_BEHAVIORAL_GATE_REJECTED"
    assert outcome["heldout_model_outputs_observed"] is False
    assert model.n_forward_calls == 8 * 3 * 2
    assert all(x["split"] == "discovery" for x in json.loads(
        (tmp_path / "matched-inputs.json").read_text()
    )["trials"][:8])
    all_tokens = json.loads((tmp_path / "matched-inputs.json").read_text())
    heldout = {
        tuple(t["repeated_tokens"][:7])
        for t in all_tokens["trials"] if t["split"] == "validation"
    }
    assert not heldout.intersection(model.seen)
    assert not outcome["discovery"]["passes_discovery_behavioral_gate"]


def test_planted_copy_effect_can_pass_discovery_gate_without_claiming_mechanism(tmp_path):
    model = FrozenPriorModel(selective=True)
    outcome = pilot.run_discovery_only(
        pretrained=model,
        tokenizer=FakeTokenizer(),
        revision="b" * 40,
        output_dir=tmp_path,
    )
    assert outcome["study_status"] == "DISCOVERY_BEHAVIORAL_GATE_PASSED_NOT_CAUSAL"
    assert outcome["discovery"]["mean_advantage_vs_reordered"] == pytest.approx(6)
    assert outcome["discovery"]["mean_advantage_vs_novel"] == pytest.approx(6)
    assert outcome["heldout_model_outputs_observed"] is False
    status = json.loads((tmp_path / "status.json").read_text())
    assert status["dataset_hash"] == outcome["dataset_hash"]


def test_moving_revision_labels_are_rejected_before_any_evaluation(tmp_path):
    model = FrozenPriorModel(selective=True)
    with pytest.raises(ValueError, match="40-hex"):
        pilot.run_discovery_only(
            pretrained=model, tokenizer=FakeTokenizer(),
            revision="main", output_dir=tmp_path,
        )
    assert model.n_forward_calls == 0
