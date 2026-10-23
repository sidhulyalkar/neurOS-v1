"""First real-checkpoint Circuit Auditor pilot, with no model downloads in CI.

Requires an immutable 40-character Hugging Face model commit. Results are
research diagnostics until independently reproduced and externally reviewed.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

from neuros_mechint import ModuleOutputTarget, OutputMetric, PyTorchAdapter, stable_hash
from neuros_mechint.benchmarks import (
    CircuitCandidate,
    EvidenceExample,
    EvidencePackPolicy,
    EvidencePackSpec,
    EvidenceSplit,
    discover_ablation_effect_candidate,
    run_adapter_evidence_pack,
    write_evidence_pack_artifact,
)
from neuros_mechint.core import EvidenceTier

MODEL_ID = "EleutherAI/pythia-70m-deduped"
DESIGN_ID = "pythia70m-attention-mlp-repetition-v1"
SEED = 17
PER_SPLIT = 4
CANDIDATE_SIZE = 2
RANDOM_TRIALS = 24
TASK_THRESHOLD = 0.75
WORDS = (
    "cat", "dog", "bird", "fish", "tree", "leaf", "wind", "rain",
    "snow", "moon", "star", "sun", "book", "door", "wall", "road",
    "rock", "lake", "river", "hill", "train", "plane", "ship", "boat",
    "bread", "milk", "rice", "salt", "apple", "orange", "green", "blue",
    "black", "white", "small", "large", "quick", "slow", "music", "light",
    "house", "garden", "table", "chair", "horse", "sheep", "cloud", "field",
)


def validate_revision(revision: str) -> str:
    """Require the full immutable HF commit, never a mutable branch or tag."""
    if re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("revision must be an immutable 40-hex Hugging Face commit SHA")
    return revision.lower()


def build_examples(tokenizer: Any, *, seed: int = SEED) -> tuple[EvidenceExample, ...]:
    """Predefined repeated-four-token task with disjoint lexical split pools.

    Each example encodes [A,B,C,D,A,B,C | expected=D | foil=E]. The model
    sees only the seven-token prefix; target/foil IDs are used for scoring.
    """
    token_ids = {}
    for word in WORDS:
        pieces = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(pieces) == 1:
            token_ids[word] = int(pieces[0])
    unique: dict[int, str] = {}
    for word, token in token_ids.items():
        unique.setdefault(token, word)
    ids = list(unique)
    if len(ids) < 24:
        raise ValueError("fewer than 24 unique one-token words; incompatible tokenizer")

    rng = random.Random(seed)
    rng.shuffle(ids)
    groups = {
        EvidenceSplit.DISCOVERY: ids[:12],
        EvidenceSplit.VALIDATION: ids[12:24],
    }
    examples = []
    for split in (EvidenceSplit.DISCOVERY, EvidenceSplit.VALIDATION):
        for index in range(PER_SPLIT):
            a, b, c, d, foil = rng.sample(groups[split], 5)
            packed = [a, b, c, d, a, b, c, d, foil]
            examples.append(
                EvidenceExample(
                    example_id=f"{split.value}-{index:02d}",
                    inputs=torch.tensor([packed], dtype=torch.long),
                    split=split,
                    metadata={"task": "four-token-repetition", "sequence_length": 7},
                )
            )
    if len({item.input_hash for item in examples}) != len(examples):
        raise ValueError("repeated prompt or collision in frozen synthetic task")
    return tuple(examples)


class NextTokenMarginModel(nn.Module):
    """Expose a scalar next-token margin without passing labels to the LM."""

    def __init__(self, language_model: nn.Module) -> None:
        super().__init__()
        self.language_model = language_model

    def forward(self, packed: torch.Tensor) -> torch.Tensor:
        if packed.ndim != 2 or packed.shape[-1] != 9:
            raise ValueError("expected [batch, seven prompt IDs + two scoring IDs]")
        prefix = packed[:, :-2]
        correct = packed[:, -2:-1]
        distractor = packed[:, -1:]
        raw = self.language_model(
            input_ids=prefix,
            use_cache=False,
            return_dict=True,
        )
        logits = raw.logits[:, -1, :]
        return logits.gather(-1, correct) - logits.gather(-1, distractor)


def make_attention_mlp_targets(model: nn.Module) -> dict[str, ModuleOutputTarget]:
    """Fail if the pinned model is no longer six-layer GPT-NeoX."""
    layers = getattr(getattr(model.language_model, "gpt_neox", None), "layers", None)
    if layers is None or len(layers) != 6:
        raise ValueError("study v1 requires exactly six GPT-NeoX transformer layers")
    targets: dict[str, ModuleOutputTarget] = {}
    for index in range(6):
        targets[f"attn_{index}"] = ModuleOutputTarget(
            f"language_model.gpt_neox.layers.{index}.attention", selector=(0,)
        )
        targets[f"mlp_{index}"] = ModuleOutputTarget(
            f"language_model.gpt_neox.layers.{index}.mlp"
        )
    return targets


def task_competence(
    adapter: PyTorchAdapter, examples: tuple[EvidenceExample, ...]
) -> dict[str, Any]:
    scores = {}
    with torch.no_grad():
        for item in examples:
            score = float(adapter.forward(item.inputs).mean().item())
            if not torch.isfinite(torch.tensor(score)):
                raise ValueError("nonfinite pretrained-model baseline")
            scores[item.example_id] = score
    return {
        "scores": scores,
        "fraction_positive": sum(value > 0 for value in scores.values()) / len(scores),
    }


def model_and_tokenizer(revision: str, device: str) -> tuple[nn.Module, Any, str]:
    # Import only when running the explicitly requested public checkpoint.
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        revision=revision,
        trust_remote_code=False,
        use_safetensors=True,
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID, revision=revision, trust_remote_code=False
    )
    return model, tokenizer, transformers.__version__


def run_study(
    *,
    pretrained_model: nn.Module,
    tokenizer: Any,
    revision: str,
    output_dir: Path,
    seed: int = SEED,
) -> dict[str, Any]:
    """Run fixed discovery and held-out tests; never assert positive findings."""
    revision = validate_revision(revision)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    model = NextTokenMarginModel(pretrained_model.eval()).eval()
    device = next(pretrained_model.parameters()).device
    examples_cpu = build_examples(tokenizer, seed=seed)
    examples = tuple(
        EvidenceExample(item.example_id, item.inputs.to(device), item.split, item.metadata)
        for item in examples_cpu
    )
    targets = make_attention_mlp_targets(model)
    adapter = PyTorchAdapter(model, targets=targets)
    # This is an intentional abort on incompatible structured model surfaces.
    adapter.capture_outputs(examples[0].inputs, tuple(targets))
    discovery = tuple(e for e in examples if e.split is EvidenceSplit.DISCOVERY)
    validation = tuple(e for e in examples if e.split is EvidenceSplit.VALIDATION)
    # Validation model outputs must not be observed before candidate freezing.
    competence = {"discovery": task_competence(adapter, discovery)}
    dataset_inputs = [
        {"id": e.example_id, "split": e.split.value, "tokens": e.inputs.cpu().tolist()[0]}
        for e in examples
    ]
    dataset_hash = stable_hash(dataset_inputs)
    record: dict[str, Any] = {
        "design_id": DESIGN_ID,
        "model_id": MODEL_ID,
        "model_revision": revision,
        "tokenizer_revision": revision,
        "seed": seed,
        "dataset_hash": dataset_hash,
        "dataset_examples": dataset_inputs,
        "task_competence": competence,
        "task_threshold": TASK_THRESHOLD,
        "target_universe": [name for name in targets],
        "target_definitions": {name: value.to_dict() for name, value in targets.items()},
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if competence["discovery"]["fraction_positive"] < TASK_THRESHOLD:
        record["study_status"] = "BASELINE_NOT_QUALIFIED"
        # No circuit inference is licensed by a model that fails the discovery task.
        (output_dir / "status.json").write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return record

    metric = OutputMetric(lambda values: values.mean(), name="next_token_logit_margin")
    ranking = discover_ablation_effect_candidate(
        adapter, discovery, tuple(targets), metric=metric, k=len(targets)
    )
    frozen_candidates = (
        CircuitCandidate(
            name="high-effect-discovery", targets=ranking.targets[:CANDIDATE_SIZE],
            scores=ranking.scores, source="discovery-single-target-ablation",
        ),
        CircuitCandidate(
            name="low-effect-decoy", targets=ranking.targets[-CANDIDATE_SIZE:],
            scores=ranking.scores, source="discovery-inverse-ablation-ranking",
        ),
    )
    record["candidate_ranking"] = list(ranking.targets)
    record["candidates"] = [candidate.to_dict() for candidate in frozen_candidates]
    results = {}
    for candidate in frozen_candidates:
        spec = EvidencePackSpec(
            pack_id=f"{DESIGN_ID}:{candidate.name}",
            model_id=MODEL_ID,
            model_revision=revision,
            tokenizer_id=MODEL_ID,
            tokenizer_revision=revision,
            dataset_id=f"synthetic:{DESIGN_ID}",
            dataset_revision=f"sha256:{dataset_hash}",
            metric_name=metric.name,
            target_universe=tuple(targets),
            discovery_method=candidate.source,
            intervention_baselines=("zero", "mean"),
            random_trials=RANDOM_TRIALS,
            seed=seed,
            evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
            metadata={
                "held_out_lexical_sets": True,
                "task_threshold": TASK_THRESHOLD,
                "exploratory": False,
            },
        )
        result = run_adapter_evidence_pack(
            spec=spec, adapter=adapter, metric=metric, examples=examples,
            candidate=candidate,
            pack_policy=EvidencePackPolicy(
                min_validation_examples=PER_SPLIT,
                bootstrap_samples=500,
            ),
        )
        artifact = write_evidence_pack_artifact(
            result, output_dir / f"{candidate.name}.json"
        )
        results[candidate.name] = {
            "artifact": artifact.name,
            "promotion_passed": result.promotion.passed,
            "publication_ready": result.publication_ready,
            "validation_pass_rate": result.validation_aggregate.pass_rate,
            "validation_joint_median": result.validation_aggregate.median_joint_faithfulness,
            "study_fingerprint": result.study_fingerprint,
            "reasons": list(result.promotion.reasons),
        }

    competence["validation"] = task_competence(adapter, validation)
    record["results"] = results
    record["study_status"] = (
        "TASK_VALIDATION_FAILED"
        if competence["validation"]["fraction_positive"] < TASK_THRESHOLD
        else "EVALUATED_NOT_INDEPENDENTLY_REPRODUCED"
    )
    # Passing a pack is not equivalent to a true circuit or independent replication.
    (output_dir / "status.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="exact 40-hex HF commit")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    args = parser.parse_args()
    revision = validate_revision(args.revision)
    pretrained, tokenizer, version = model_and_tokenizer(revision, args.device)
    record = run_study(
        pretrained_model=pretrained, tokenizer=tokenizer,
        revision=revision, output_dir=args.out,
    )
    record["transformers_version"] = version
    record["torch_version"] = torch.__version__
    (args.out / "status.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": record["study_status"],
        "output_dir": str(args.out),
        "task_competence": record["task_competence"],
        "results": record.get("results", {}),
    }, indent=2))
    if record["study_status"] == "BASELINE_NOT_QUALIFIED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
