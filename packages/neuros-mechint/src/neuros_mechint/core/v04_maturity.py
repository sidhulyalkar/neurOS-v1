"""v0.4 method cards kept separate from the legacy maturity registry body."""

from __future__ import annotations

from .maturity import MethodCard, MethodMaturity, register_method_card


def register_v04_method_cards() -> None:
    """Register v0.4 scientific claim boundaries idempotently."""

    cards = (
        MethodCard(
            name="checkpoint_mechanism_emergence",
            maturity=MethodMaturity.RESEARCH,
            establishes=(
                "when a matched checkpoint trajectory acquires its final causal-effect structure",
            ),
            limitations=(
                "uses the final observed checkpoint as a reference rather than biological ground truth",
                "emergence steps depend on explicit magnitude and stability thresholds",
            ),
            required_controls=(
                "checkpoint as the only varying scientific axis",
                "multiple adjacent checkpoints",
                "held-out seeds or training runs",
                "known-ground-truth emergence benchmark",
            ),
        ),
        MethodCard(
            name="orion_tokenizer_mechanism_study",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "matched causal-profile agreement or divergence when only neural tokenization changes",
            ),
            limitations=(
                "tokenizers must share semantic time alignment",
                "downstream models can adapt to tokenizer-specific artifacts",
            ),
            required_controls=(
                "matched token budget and temporal resolution",
                "matched downstream model/task",
                "within-window shuffle controls",
                "held-out sessions and subjects",
            ),
        ),
        MethodCard(
            name="neurofm_internal_representation_probe",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "reproducible extraction of tensor-valued internal NeuroFM states",
            ),
            limitations=(
                "capturing a representation does not establish its causal role",
                "compressed latent states require explicit timestamps rather than inferred alignment",
            ),
            required_controls=(
                "explicit module path",
                "explicit temporal alignment for compressed states",
                "model eval-mode capture",
            ),
        ),
        MethodCard(
            name="neurofm_mechanism_lab",
            maturity=MethodMaturity.RESEARCH,
            establishes=(
                "joint architecture comparison and checkpoint-emergence analysis over NeuroFM states",
            ),
            limitations=(
                "architecture claims require matched performance and checkpoint maturity",
                "shared causal profiles do not prove identical biological implementation",
            ),
            required_controls=(
                "isolated architecture contrasts",
                "checkpoint-only longitudinal groups",
                "matched event alignment",
                "alternative intervention families",
            ),
        ),
    )
    for card in cards:
        register_method_card(card, replace=True)
