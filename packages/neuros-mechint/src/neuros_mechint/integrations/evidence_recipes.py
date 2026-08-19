"""Pinned-at-execution recipes for the first v0.6 real-model evidence packs.

Recipes are intentionally declarative. They name small or single-GPU-feasible
upstream models and the supported neurOS adapter path, but require callers to
resolve immutable model/tokenizer/SAE/transcoder revisions before publishing an
artifact. A mutable model alias is never treated as reproducible evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any


@dataclass(frozen=True, slots=True)
class ExternalEvidenceRecipe:
    """Reproducible starting point for one external-model evidence run."""

    recipe_id: str
    ecosystem: str
    install_extra: str
    model_id: str
    tokenizer_id: str | None
    discovery_method: str
    target_surface: str
    candidate_size: int
    recommended_device: str
    revision_policy: str = "resolve-and-pin-immutable-revision-before-run"
    dataset_id: str = "neurOS:mechint-heldout-prompts-v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.candidate_size <= 0:
            raise ValueError("candidate_size must be positive")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_size": self.candidate_size,
            "dataset_id": self.dataset_id,
            "discovery_method": self.discovery_method,
            "ecosystem": self.ecosystem,
            "install_extra": self.install_extra,
            "metadata": dict(self.metadata),
            "model_id": self.model_id,
            "recommended_device": self.recommended_device,
            "recipe_id": self.recipe_id,
            "revision_policy": self.revision_policy,
            "target_surface": self.target_surface,
            "tokenizer_id": self.tokenizer_id,
        }


def external_evidence_recipes() -> tuple[ExternalEvidenceRecipe, ...]:
    """Return the maintained v0.6 external-model evidence starting points."""

    return (
        ExternalEvidenceRecipe(
            recipe_id="transformer-lens-tinystories-21m",
            ecosystem="transformer_lens",
            install_extra="transformer-lens",
            model_id="tiny-stories-1L-21M",
            tokenizer_id="gpt2",
            discovery_method="single-target-zero-ablation on hook points",
            target_surface=(
                "blocks.0.hook_resid_pre / blocks.0.hook_attn_out / "
                "blocks.0.hook_mlp_out / blocks.0.hook_resid_post"
            ),
            candidate_size=2,
            recommended_device="cpu-or-small-gpu",
            metadata={
                "upstream_contract": "TransformerLens run_with_cache/run_with_hooks",
                "reason": "small TransformerLens model suitable for reproducible smoke evidence",
            },
        ),
        ExternalEvidenceRecipe(
            recipe_id="nnsight-gpt2",
            ecosystem="nnsight",
            install_extra="nnsight",
            model_id="openai-community/gpt2",
            tokenizer_id="openai-community/gpt2",
            discovery_method="single-target-zero-ablation across transformer blocks",
            target_surface="transformer.h.{0..11}::0",
            candidate_size=3,
            recommended_device="cpu-or-small-gpu",
            metadata={
                "upstream_contract": "LanguageModel trace/output.save/intervention assignment",
                "reason": "NNsight upstream quick-start model with stable block addressing",
            },
        ),
        ExternalEvidenceRecipe(
            recipe_id="sae-lens-gpt2-residual",
            ecosystem="sae_lens",
            install_extra="sae-lens",
            model_id="gpt2-small",
            tokenizer_id="gpt2",
            discovery_method="feature activation or causal feature ablation on discovery prompts",
            target_surface="SAE release gpt2-small-res-jb / blocks.8.hook_resid_pre",
            candidate_size=16,
            recommended_device="small-gpu",
            metadata={
                "sae_id": "blocks.8.hook_resid_pre",
                "sae_release": "gpt2-small-res-jb",
                "upstream_contract": "SAE.from_pretrained + encode/decode",
            },
        ),
        ExternalEvidenceRecipe(
            recipe_id="circuit-tracer-gemma2-2b",
            ecosystem="circuit_tracer",
            install_extra="official-source-install",
            model_id="google/gemma-2-2b",
            tokenizer_id="google/gemma-2-2b",
            discovery_method="circuit-tracer attribution graph, then frozen top-feature candidate",
            target_surface="transcoder features nominated by AttributionGraphSummary",
            candidate_size=32,
            recommended_device="single-gpu-15gb-or-more",
            metadata={
                "transcoder_set": "mntss/gemma-scope-transcoders",
                "upstream_contract": "ReplacementModel + attribute + feature interventions",
                "warning": "attribution graph is discovery only; held-out intervention is mandatory",
            },
        ),
    )


def external_evidence_recipe_dicts() -> list[dict[str, Any]]:
    return [recipe.to_dict() for recipe in external_evidence_recipes()]
