"""Behavioral repetition controls must reject mere positive token priors."""

from __future__ import annotations

import copy

import pytest
import torch

from neuros_mechint.benchmarks.evidence_pack import EvidenceSplit
from neuros_mechint.benchmarks.repetition_preflight import (
    CONTROL_SCHEMA,
    MatchedRepetitionTrial,
    make_matched_repetition_trials,
    score_discovery_repetition_controls,
)

WORDS = tuple("word" + str(i) for i in range(48))


class SingleTokenTokenizer:
    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return [int(text.strip()[4:]) + 3]


def _trials():
    return make_matched_repetition_trials(
        SingleTokenTokenizer(), WORDS, per_split=8, seed=37
    )


def _discovery(trials):
    return tuple(t for t in trials if t.split is EvidenceSplit.DISCOVERY)


def _positive_token_prior(input_ids):
    del input_ids
    return torch.tensor([[5.0]])


def _copy_sensitive(input_ids):
    packed = input_ids[0].tolist()
    if packed[4:7] == packed[:3]:
        return torch.tensor([[5.0]])
    return torch.tensor([[-1.0]])


def _wrong_but_repetition_sensitive(input_ids):
    packed = input_ids[0].tolist()
    value = -1.0 if packed[4:7] == packed[:3] else -4.0
    return torch.tensor([[value]])


def test_reproducible_control_shapes_and_lexical_heldout_partition():
    trials = _trials()
    repeated = _trials()
    assert [x.identity_dict() for x in trials] == [x.identity_dict() for x in repeated]
    assert len(trials) == 16
    by_split = {}
    for split in EvidenceSplit:
        group = [t for t in trials if t.split is split]
        assert len(group) == 8
        by_split[split] = set()
        for trial in group:
            rep, reorder, novel = (
                value[0].tolist() for value in
                (trial.repeated, trial.reordered, trial.novel)
            )
            assert rep[:4] == reorder[:4] == novel[:4]
            assert rep[-2:] == reorder[-2:] == novel[-2:]
            assert sorted(rep[:7]) == sorted(reorder[:7])
            assert rep[-3] == reorder[-3]
            assert tuple(rep[4:7]) != tuple(reorder[4:7])
            by_split[split].update(rep[:7] + novel[:7])
    assert not by_split[EvidenceSplit.DISCOVERY] & by_split[EvidenceSplit.VALIDATION]


def test_positive_raw_margin_without_context_copying_is_rejected():
    result = score_discovery_repetition_controls(
        _positive_token_prior, _discovery(_trials()), bootstrap_samples=30
    )
    assert result["schema_version"] == CONTROL_SCHEMA
    assert result["cases"][0]["repeated_margin"] == 5
    assert result["fraction_positive_repeated_margin"] == 1
    assert result["mean_advantage_vs_reordered"] == 0
    assert result["mean_advantage_vs_novel"] == 0
    assert result["passes_discovery_behavioral_gate"] is False


def test_planted_copying_passes_discovery_control_gate():
    result = score_discovery_repetition_controls(
        _copy_sensitive, _discovery(_trials()), bootstrap_samples=30
    )
    assert result["passes_discovery_behavioral_gate"] is True
    assert result["mean_advantage_vs_reordered"] == pytest.approx(6)
    assert result["mean_advantage_vs_novel"] == pytest.approx(6)
    assert result["ci95_mean_advantage_vs_reordered"] == pytest.approx([6, 6])
    assert len(result["study_fingerprint"]) == 64


def test_incorrect_predictions_cannot_pass_on_advantage_alone():
    result = score_discovery_repetition_controls(
        _wrong_but_repetition_sensitive, _discovery(_trials()), bootstrap_samples=30
    )
    assert result["mean_advantage_vs_reordered"] > 0
    assert result["fraction_positive_repeated_margin"] == 0
    assert result["passes_discovery_behavioral_gate"] is False


def test_validation_examples_never_reach_discovery_forward():
    visited = []

    def forward(input_ids):
        visited.append(tuple(input_ids[0].tolist()))
        return torch.tensor([[1.0]])

    with pytest.raises(ValueError, match="validation"):
        score_discovery_repetition_controls(forward, _trials(), bootstrap_samples=30)
    assert not visited


def test_malformed_control_is_rejected():
    original = _discovery(_trials())[0]
    changed = original.reordered.clone()
    changed[0, 4] = original.repeated[0, 4]
    with pytest.raises(ValueError, match="reordered"):
        MatchedRepetitionTrial(
            original.trial_id, original.split, original.repeated, changed, original.novel
        )


def test_nondeterministic_and_nonfinite_scores_are_rejected():
    count = 0

    def stochastic(ids):
        nonlocal count
        del ids
        count += 1
        return torch.tensor([[float(count)]])

    with pytest.raises(ValueError, match="stochastic"):
        score_discovery_repetition_controls(
            stochastic, _discovery(_trials()), bootstrap_samples=30
        )

    def nonfinite(ids):
        del ids
        return torch.tensor([[float("nan")]])

    with pytest.raises(ValueError, match="nonfinite"):
        score_discovery_repetition_controls(
            nonfinite, _discovery(_trials()), bootstrap_samples=30
        )


def test_score_replay_is_stable_without_mutating_data_or_rng():
    trials = _discovery(_trials())
    frozen = [copy.deepcopy(t.identity_dict()) for t in trials]
    torch.manual_seed(5432)
    before = torch.random.get_rng_state().clone()
    one = score_discovery_repetition_controls(
        _copy_sensitive, trials, bootstrap_samples=30, seed=4
    )
    two = score_discovery_repetition_controls(
        _copy_sensitive, trials, bootstrap_samples=30, seed=4
    )
    assert one == two
    assert [t.identity_dict() for t in trials] == frozen
    assert torch.equal(before, torch.random.get_rng_state())


def test_single_token_induction_can_fail_ordered_repeat_gate():
    """A genuine single-token C->D copying rule need not distinguish ABC from BAC."""

    def single_token_copy(input_ids):
        tokens = input_ids[0].tolist()
        return torch.tensor([[5.0 if tokens[6] == tokens[2] else -1.0]])

    result = score_discovery_repetition_controls(
        single_token_copy, _discovery(_trials()), bootstrap_samples=30
    )
    assert result["fraction_positive_repeated_margin"] == 1.0
    assert result["mean_advantage_vs_novel"] > 0.0
    assert result["mean_advantage_vs_reordered"] == pytest.approx(0.0)
    assert result["passes_discovery_behavioral_gate"] is False
