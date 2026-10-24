"""Matched repetition controls before claiming a token-copying mechanism.

This is a discovery-only *behavioral* gate. It deliberately does not infer a
circuit, expose held-out model outputs, or alter the v1 Pythia pilot's identity.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from neuros_mechint.core.manifest import stable_hash

from .evidence_pack import EvidenceSplit

CONTROL_SCHEMA = "neuros-mechint.matched-repetition-preflight.v1"


@dataclass(frozen=True, slots=True)
class MatchedRepetitionTrial:
    trial_id: str
    split: EvidenceSplit
    repeated: torch.Tensor
    reordered: torch.Tensor
    novel: torch.Tensor

    def __post_init__(self) -> None:
        if not self.trial_id:
            raise ValueError("trial_id must be nonempty")
        if self.split not in (EvidenceSplit.DISCOVERY, EvidenceSplit.VALIDATION):
            raise ValueError("unsupported trial split")
        vectors = (self.repeated, self.reordered, self.novel)
        if any(
            not isinstance(value, torch.Tensor)
            or value.dtype != torch.long
            or tuple(value.shape) != (1, 9)
            for value in vectors
        ):
            raise ValueError("all matched inputs must be int64 [1,9] tensors")
        repeated, reordered, novel = (value[0].tolist() for value in vectors)
        a, b, c, d, aa, bb, cc, target, foil = repeated
        if len({a, b, c, d, foil}) != 5:
            raise ValueError("repeated prompt must use distinct A,B,C,D and foil")
        if [aa, bb, cc, target] != [a, b, c, d]:
            raise ValueError("repeated pattern must be A B C D A B C -> D")
        if reordered != [a, b, c, d, b, a, c, d, foil]:
            raise ValueError("reordered control must permute repeated suffix only")
        f, g, h = novel[4:7]
        if novel[:4] != [a, b, c, d] or novel[7:] != [d, foil]:
            raise ValueError("novel control must retain shared prefix and scoring IDs")
        if len({a, b, c, d, foil, f, g, h}) != 8:
            raise ValueError("novel suffix tokens must be disjoint from prompt and foil")

    def identity_dict(self) -> dict[str, Any]:
        return {
            "trial_id": self.trial_id,
            "split": self.split.value,
            "repeated_tokens": self.repeated[0].tolist(),
            "reordered_tokens": self.reordered[0].tolist(),
            "novel_tokens": self.novel[0].tolist(),
        }


def make_matched_repetition_trials(
    tokenizer: Any,
    words: Sequence[str],
    *,
    per_split: int = 8,
    seed: int = 37,
) -> tuple[MatchedRepetitionTrial, ...]:
    """Freeze paired controls and disjoint lexical pools before model inference."""
    if not 1 <= per_split <= 32:
        raise ValueError("per_split must be 1..32")
    token_ids: dict[int, str] = {}
    for word in words:
        ids = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(ids) == 1:
            token_ids.setdefault(int(ids[0]), word)
    tokens = list(token_ids)
    if len(tokens) < 24:
        raise ValueError("at least 24 distinct one-token words are required")

    rng = random.Random(seed)
    rng.shuffle(tokens)
    pools = {
        EvidenceSplit.DISCOVERY: tokens[:12],
        EvidenceSplit.VALIDATION: tokens[12:24],
    }
    trials = []
    for split in (EvidenceSplit.DISCOVERY, EvidenceSplit.VALIDATION):
        for index in range(per_split):
            a, b, c, d, foil, f, g, h = rng.sample(pools[split], 8)
            trials.append(
                MatchedRepetitionTrial(
                    trial_id=f"{split.value}-{index:02d}",
                    split=split,
                    repeated=torch.tensor([[a, b, c, d, a, b, c, d, foil]]),
                    reordered=torch.tensor([[a, b, c, d, b, a, c, d, foil]]),
                    novel=torch.tensor([[a, b, c, d, f, g, h, d, foil]]),
                )
            )
    identities = [trial.identity_dict() for trial in trials]
    content_keys = [
        stable_hash({
            "repeated": trial.repeated,
            "reordered": trial.reordered,
            "novel": trial.novel,
        })
        for trial in trials
    ]
    if len(set(content_keys)) != len(trials):
        raise ValueError("duplicate matched trial content")
    if len({identity["trial_id"] for identity in identities}) != len(trials):
        raise ValueError("duplicate matched trial identity")
    return tuple(trials)


def _bootstrap_ci(values: Sequence[float], *, samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        estimates.append(sum(rng.choice(values) for _ in values) / len(values))
    estimates.sort()
    return estimates[int(0.025 * (samples - 1))], estimates[int(0.975 * (samples - 1))]


@torch.no_grad()
def score_discovery_repetition_controls(
    forward: Callable[[torch.Tensor], torch.Tensor],
    trials: Sequence[MatchedRepetitionTrial],
    *,
    min_positive_fraction: float = 0.75,
    bootstrap_samples: int = 500,
    seed: int = 37,
) -> dict[str, Any]:
    """Evaluate discovery examples only; repeated margin alone never qualifies."""
    trials = tuple(trials)
    if not trials:
        raise ValueError("at least one discovery trial is required")
    if any(t.split is not EvidenceSplit.DISCOVERY for t in trials):
        raise ValueError("validation model outputs are not permitted during discovery")
    if not 0 < min_positive_fraction <= 1:
        raise ValueError("min_positive_fraction must lie in (0,1]")
    if bootstrap_samples < 20:
        raise ValueError("bootstrap_samples must be >= 20")
    if len({t.trial_id for t in trials}) != len(trials):
        raise ValueError("duplicate discovery trial_id")

    def scalar(input_ids: torch.Tensor) -> float:
        pristine = input_ids.clone()
        output = forward(input_ids)
        replay = forward(input_ids)
        if not torch.equal(pristine, input_ids):
            raise RuntimeError("model mutated frozen trial inputs")
        if (
            not isinstance(output, torch.Tensor)
            or not isinstance(replay, torch.Tensor)
            or output.numel() != 1
            or replay.numel() != 1
        ):
            raise TypeError("margin model must return exactly one tensor scalar")
        if not bool(torch.isfinite(output).all() and torch.isfinite(replay).all()):
            raise ValueError("nonfinite margin in matched control")
        if not torch.equal(output, replay):
            raise ValueError("stochastic model margin: repeated forward differs")
        return float(output.item())

    cases = []
    for trial in trials:
        repeated = scalar(trial.repeated)
        reordered = scalar(trial.reordered)
        novel = scalar(trial.novel)
        cases.append({
            "trial_id": trial.trial_id,
            "repeated_margin": repeated,
            "reordered_margin": reordered,
            "novel_margin": novel,
            "advantage_vs_reordered": repeated - reordered,
            "advantage_vs_novel": repeated - novel,
        })

    by_order = [c["advantage_vs_reordered"] for c in cases]
    by_novel = [c["advantage_vs_novel"] for c in cases]
    fraction_order = sum(value > 0 for value in by_order) / len(cases)
    fraction_novel = sum(value > 0 for value in by_novel) / len(cases)
    fraction_repeated_correct = sum(c["repeated_margin"] > 0 for c in cases) / len(cases)
    mean_order = sum(by_order) / len(by_order)
    mean_novel = sum(by_novel) / len(by_novel)
    decision = (
        fraction_order >= min_positive_fraction
        and fraction_novel >= min_positive_fraction
        and fraction_repeated_correct >= min_positive_fraction
        and mean_order > 0
        and mean_novel > 0
    )
    summary = {
        "schema_version": CONTROL_SCHEMA,
        "split": EvidenceSplit.DISCOVERY.value,
        "n_trials": len(cases),
        "seed": seed,
        "min_positive_fraction": min_positive_fraction,
        "trial_identity_hash": stable_hash([trial.identity_dict() for trial in trials]),
        "cases": cases,
        "fraction_positive_repeated_margin": fraction_repeated_correct,
        "fraction_positive_vs_reordered": fraction_order,
        "fraction_positive_vs_novel": fraction_novel,
        "mean_advantage_vs_reordered": mean_order,
        "mean_advantage_vs_novel": mean_novel,
        "ci95_mean_advantage_vs_reordered": list(
            _bootstrap_ci(by_order, samples=bootstrap_samples, seed=seed)
        ),
        "ci95_mean_advantage_vs_novel": list(
            _bootstrap_ci(by_novel, samples=bootstrap_samples, seed=seed + 1)
        ),
        "passes_discovery_behavioral_gate": decision,
    }
    summary["study_fingerprint"] = stable_hash(summary)
    return summary
