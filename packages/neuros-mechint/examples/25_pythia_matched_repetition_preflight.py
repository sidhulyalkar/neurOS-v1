"""Discovery-only matched-repetition preflight for one pinned Pythia checkpoint.

This study has a distinct design ID from the v1 causal-auditor pilot.
It is a behavioral gate, not a circuit discovery or held-out causal result.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from torch import nn

from neuros_mechint.benchmarks.repetition_preflight import (
    make_matched_repetition_trials,
    score_discovery_repetition_controls,
)
from neuros_mechint.benchmarks.evidence_pack import EvidenceSplit
from neuros_mechint.core.manifest import stable_hash

MODEL_ID = "EleutherAI/pythia-70m-deduped"
DESIGN_ID = "pythia70m-matched-repetition-discovery-v2"
WORDS = (
    "cat", "dog", "bird", "fish", "tree", "leaf", "wind", "rain",
    "snow", "moon", "star", "sun", "book", "door", "wall", "road",
    "rock", "lake", "river", "hill", "train", "plane", "ship", "boat",
    "bread", "milk", "rice", "salt", "apple", "orange", "green", "blue",
    "black", "white", "small", "large", "quick", "slow", "music", "light",
    "house", "garden", "table", "chair", "horse", "sheep", "cloud", "field",
)


class TokenMargin(nn.Module):
    """Seven prompt tokens, followed by answer/foil tokens not sent to the LM."""

    def __init__(self, pretrained: nn.Module) -> None:
        super().__init__()
        self.pretrained = pretrained

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 2 or tuple(packed.shape) != (1, 9):
            raise ValueError("expected one 7-token prompt with two scoring IDs")
        scores = self.pretrained(
            input_ids=packed[:, :7], use_cache=False, return_dict=True
        ).logits[:, -1, :]
        return scores.gather(-1, packed[:, 7:8]) - scores.gather(-1, packed[:, 8:9])


@torch.no_grad()
def run_discovery_only(
    *,
    pretrained: nn.Module,
    tokenizer: object,
    revision: str,
    output_dir: Path,
) -> dict[str, object]:
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("pin an immutable lowercase 40-hex model revision")
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen = make_matched_repetition_trials(tokenizer, WORDS, per_split=8, seed=37)
    manifest = {
        "design_id": DESIGN_ID,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "target": "D versus E margin on 7-token context",
        "protocol": "matched repeated/reordered/novel controls",
        "seed": 37,
        "trials": [trial.identity_dict() for trial in frozen],
    }
    manifest["dataset_hash"] = stable_hash(manifest["trials"])
    (output_dir / "matched-inputs.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # No validation example is passed to the model at any point in this function.
    discovery = tuple(t for t in frozen if t.split is EvidenceSplit.DISCOVERY)
    scorer = TokenMargin(pretrained.eval()).eval().to(
        next(pretrained.parameters()).device
    )
    # Copy only the discovery examples to the selected model device.
    from neuros_mechint.benchmarks.repetition_preflight import MatchedRepetitionTrial

    working = tuple(
        MatchedRepetitionTrial(
            t.trial_id, t.split,
            t.repeated.to(next(pretrained.parameters()).device),
            t.reordered.to(next(pretrained.parameters()).device),
            t.novel.to(next(pretrained.parameters()).device),
        )
        for t in discovery
    )
    evidence = score_discovery_repetition_controls(
        scorer.forward, working, min_positive_fraction=0.75,
        bootstrap_samples=500, seed=37,
    )
    record = {
        "design_id": DESIGN_ID,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "dataset_hash": manifest["dataset_hash"],
        "heldout_model_outputs_observed": False,
        "evidence_tier": "scientific_synthetic",
        "discovery": evidence,
        "study_status": (
            "DISCOVERY_BEHAVIORAL_GATE_PASSED_NOT_CAUSAL"
            if evidence["passes_discovery_behavioral_gate"]
            else "DISCOVERY_BEHAVIORAL_GATE_REJECTED"
        ),
    }
    (output_dir / "status.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()

    # Model weights are never downloaded by importing this module or running CI.
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if re.fullmatch(r"[0-9a-f]{40}", args.revision) is None:
        raise SystemExit("pin an immutable lowercase 40-hex model revision")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, revision=args.revision, trust_remote_code=False,
        use_safetensors=True,
    ).to(args.device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=args.revision, trust_remote_code=False
    )
    report = run_discovery_only(
        pretrained=model, tokenizer=tokenizer,
        revision=args.revision, output_dir=args.out,
    )
    print(json.dumps({
        "status": report["study_status"],
        "out": str(args.out),
        "dataset_hash": report["dataset_hash"],
        "discovery_summary": report["discovery"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
