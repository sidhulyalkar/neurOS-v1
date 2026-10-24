"""Offline, deterministic contract checks for the *real-model* pilot entrypoint.

These tests do not download checkpoints and do not qualify an LLM mechanism.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from neuros_mechint import PyTorchAdapter

_SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "examples"
    / "24_pythia_real_model_pilot.py"
)
_spec = importlib.util.spec_from_file_location("pythia_real_model_pilot", _SCRIPT)
assert _spec is not None and _spec.loader is not None
pilot = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pilot)


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        return [pilot.WORDS.index(text.strip()) + 3]


class BadTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        return [1, 2]


class FakeAttention(nn.Module):
    def forward(self, x):
        return (x + 0.1,)


class FakeLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = FakeAttention()
        self.mlp = nn.Identity()

    def forward(self, x):
        return self.mlp(self.attention(x)[0])


class FakeLanguageModel(nn.Module):
    def __init__(self, *, competent=True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.gpt_neox = nn.Module()
        self.gpt_neox.layers = nn.ModuleList([FakeLayer() for _ in range(6)])
        self.competent = competent

    def forward(self, *, input_ids, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        hidden = input_ids.float().unsqueeze(-1)
        for layer in self.gpt_neox.layers:
            hidden = layer(hidden)
        b, s = input_ids.shape
        logits = torch.zeros((b, s, 128), dtype=torch.float32)
        wanted = input_ids[:, 3].view(b, 1, 1).expand(b, s, 1)
        base = 5.0 if self.competent else -20.0
        # Responses are intentionally sensitive to hooks (nonzero all/null span).
        signal = hidden[:, -1:, :] / 20.0
        logits.scatter_(2, wanted, (base + signal).expand(b, s, 1))
        return SimpleNamespace(logits=logits)


def test_requires_exact_immutable_revision():
    assert pilot.validate_revision("a" * 40) == "a" * 40
    for bad in ("main", "step143000", "a" * 7, "g" * 40):
        with pytest.raises(ValueError, match="40-hex"):
            pilot.validate_revision(bad)


def test_tokenized_task_is_reproducible_and_lexically_disjoint():
    left = pilot.build_examples(FakeTokenizer(), seed=17)
    right = pilot.build_examples(FakeTokenizer(), seed=17)
    assert len(left) == len(right) == 2 * pilot.PER_SPLIT
    assert [x.input_hash for x in left] == [x.input_hash for x in right]
    assert len(set(x.input_hash for x in left)) == len(left)

    split_tokens = {}
    for split in (pilot.EvidenceSplit.DISCOVERY, pilot.EvidenceSplit.VALIDATION):
        examples = [item for item in left if item.split is split]
        assert len(examples) == pilot.PER_SPLIT
        split_tokens[split] = set()
        for example in examples:
            a, b, c, d, aa, bb, cc, good, bad = example.inputs[0].tolist()
            assert [a, b, c] == [aa, bb, cc]
            assert good == d
            assert bad not in (a, b, c, d)
            split_tokens[split].update((a, b, c, d, bad))
    assert not split_tokens[pilot.EvidenceSplit.DISCOVERY].intersection(
        split_tokens[pilot.EvidenceSplit.VALIDATION]
    )


def test_incompatible_tokenizer_is_rejected():
    with pytest.raises(ValueError, match="fewer than 24"):
        pilot.build_examples(BadTokenizer())


def test_fake_attention_and_mlp_hook_surfaces_are_executed():
    model = pilot.NextTokenMarginModel(FakeLanguageModel())
    targets = pilot.make_attention_mlp_targets(model)
    assert len(targets) == 12
    assert targets["attn_0"].selector == (0,)
    assert targets["mlp_5"].selector == ()
    adapter = PyTorchAdapter(model, targets=targets)
    example = pilot.build_examples(FakeTokenizer())[0]
    score = float(adapter.forward(example.inputs).item())
    assert score > 5.0
    captured = adapter.capture_outputs(example.inputs, tuple(targets))
    assert len(captured) == 12
    modified = adapter.forward_with_replacements(
        example.inputs, {"attn_0": torch.zeros_like(captured["attn_0"])}
    )
    assert modified.shape == (1, 1)
    assert not model.language_model.gpt_neox.layers[0].attention._forward_hooks


def test_incompetent_model_yields_diagnostic_and_no_promotion(tmp_path):
    pretrained = FakeLanguageModel(competent=False)
    torch.manual_seed(741)
    before_rng = torch.random.get_rng_state().clone()
    result = pilot.run_study(
        pretrained_model=pretrained,
        tokenizer=FakeTokenizer(),
        revision="a" * 40,
        output_dir=tmp_path,
    )
    assert torch.equal(before_rng, torch.random.get_rng_state())
    assert result["study_status"] == "BASELINE_NOT_QUALIFIED"
    assert result["task_competence"]["discovery"]["fraction_positive"] == 0.0
    assert (tmp_path / "dataset_manifest.json").exists()
    assert (tmp_path / "status.json").exists()
    assert not list(tmp_path.glob("high-effect-discovery.json"))
    assert not list(tmp_path.glob("low-effect-decoy.json"))


def test_nonfinite_intervention_metrics_are_rejected():
    for value in (float("nan"), float("inf"), -float("inf")):
        with pytest.raises(ValueError, match="nonfinite"):
            pilot.checked_margin(torch.tensor([[value]]))
    assert pilot.checked_margin(torch.tensor([[2.0]])).item() == pytest.approx(2.0)


class StochasticFakeLanguageModel(FakeLanguageModel):
    def forward(self, *, input_ids, use_cache, return_dict):
        result = super().forward(
            input_ids=input_ids, use_cache=use_cache, return_dict=return_dict
        )
        result.logits = result.logits + torch.randn_like(result.logits)
        return result


def test_nondeterministic_pretrained_forward_fails_preflight(tmp_path):
    with pytest.raises(ValueError, match="non-deterministic"):
        pilot.run_study(
            pretrained_model=StochasticFakeLanguageModel(),
            tokenizer=FakeTokenizer(),
            revision="a" * 40,
            output_dir=tmp_path,
        )


def test_offline_full_pipeline_persists_negative_evidence(monkeypatch, tmp_path):
    """Exercise the full evidence pack on a fake LM at a NON-scientific CI budget."""
    from neuros_mechint.benchmarks import read_evidence_pack_artifact

    monkeypatch.setattr(pilot, "PER_SPLIT", 2)
    monkeypatch.setattr(pilot, "RANDOM_TRIALS", 2)
    result = pilot.run_study(
        pretrained_model=FakeLanguageModel(competent=True),
        tokenizer=FakeTokenizer(),
        revision="b" * 40,
        output_dir=tmp_path,
    )
    assert result["study_status"] == "EVALUATED_NOT_INDEPENDENTLY_REPRODUCED"
    assert list(result["results"]) == [
        "high-effect-discovery",
        "low-effect-decoy",
    ]
    for name in result["results"]:
        artifact = read_evidence_pack_artifact(tmp_path / f"{name}.json")
        assert artifact["study_fingerprint"] == result["results"][name]["study_fingerprint"]
        # Constant synthetic logits cannot support any real causal circuit.
        assert result["results"][name]["promotion_passed"] is False
        # Publication-ready means revisions pinned, NOT a causal finding.
        assert result["results"][name]["provenance_complete"] is True


class ConstantCompetentFakeLanguageModel(FakeLanguageModel):
    """Positive preflight margins with no causal path through the model hooks."""

    def forward(self, *, input_ids, use_cache, return_dict):
        assert use_cache is False and return_dict is True
        b, steps = input_ids.shape
        logits = torch.zeros((b, steps, 128), dtype=torch.float32)
        target = input_ids[:, 3].view(b, 1, 1).expand(b, steps, 1)
        logits.scatter_(2, target, 5.0)
        # Execute the modules, but deliberately make their activations irrelevant.
        hidden = input_ids.float().unsqueeze(-1)
        for layer in self.gpt_neox.layers:
            hidden = layer(hidden)
        assert hidden.shape[:2] == input_ids.shape
        return SimpleNamespace(logits=logits)


def test_pilot_serializes_nonestimable_interventions_as_negative_artifacts(
    monkeypatch, tmp_path
):
    from neuros_mechint.benchmarks import read_evidence_pack_artifact

    monkeypatch.setattr(pilot, "PER_SPLIT", 2)
    monkeypatch.setattr(pilot, "RANDOM_TRIALS", 2)
    outcome = pilot.run_study(
        pretrained_model=ConstantCompetentFakeLanguageModel(),
        tokenizer=FakeTokenizer(),
        revision="c" * 40,
        output_dir=tmp_path,
    )
    assert outcome["study_status"] == "NONESTIMABLE_INTERVENTIONS"
    for label, summary in outcome["results"].items():
        assert summary["status"] == "NONESTIMABLE"
        assert summary["promotion_passed"] is False
        assert summary["validation_joint_median"] is None
        path = tmp_path / summary["artifact"]
        assert path.exists(), label
        payload = read_evidence_pack_artifact(path)
        assert payload["promotion"]["passed"] is False
        assert payload["validation_aggregate"]["estimable"] is False
        assert all(not case["valid"] for case in payload["candidate_cases"])


def test_preflight_only_exercises_hook_contract_without_exposing_validation(tmp_path):
    model = FakeLanguageModel(competent=True)
    result = pilot.run_study(
        pretrained_model=model,
        tokenizer=FakeTokenizer(),
        revision="a" * 40,
        output_dir=tmp_path,
        preflight_only=True,
    )
    assert result["study_status"] == "PREFLIGHT_ONLY_QUALIFIED"
    assert "discovery" in result["task_competence"]
    assert "validation" not in result["task_competence"]
    assert "candidate_ranking" not in result
    assert "candidates" not in result
    assert "results" not in result
    assert (tmp_path / "status.json").exists()
    assert (tmp_path / "dataset_manifest.json").exists()
    assert not list(tmp_path.glob("high-effect-discovery.json"))
    assert not model.gpt_neox.layers[0].attention._forward_hooks


def test_incompetent_preflight_writes_negative_and_no_heldout_scores(tmp_path):
    outcome = pilot.run_study(
        pretrained_model=FakeLanguageModel(competent=False),
        tokenizer=FakeTokenizer(),
        revision="f" * 40,
        output_dir=tmp_path,
        preflight_only=True,
    )
    assert outcome["study_status"] == "BASELINE_NOT_QUALIFIED"
    assert "validation" not in outcome["task_competence"]
    assert not list(tmp_path.glob("*discovery.json"))
