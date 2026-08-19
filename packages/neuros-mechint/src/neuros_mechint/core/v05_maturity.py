"""v0.5 method cards for ecosystem adapters and circuit faithfulness."""

from __future__ import annotations

from .maturity import MethodCard, MethodMaturity, register_method_card


def register_v05_method_cards() -> None:
    """Register v0.5 scientific claim boundaries idempotently."""

    cards = (
        MethodCard(
            name="circuit_faithfulness_benchmark",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "necessity and sufficiency of a nominated target set under a specified intervention",
                "performance relative to equal-cardinality random target sets",
            ),
            limitations=(
                "faithfulness is metric-, input-, and intervention-specific",
                "passing does not establish uniqueness of the circuit",
                "off-manifold ablations can exaggerate necessity",
            ),
            required_controls=(
                "all-target versus null span",
                "equal-cardinality random circuits",
                "held-out examples",
                "alternative intervention baselines",
            ),
        ),
        MethodCard(
            name="transformer_lens_model_adapter",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "reproducible capture and replacement of named TransformerLens hook points",
            ),
            limitations=(
                "hook-point availability depends on the wrapped architecture",
                "adapter compatibility does not establish circuit correctness",
            ),
            required_controls=(
                "explicit hook names",
                "shape-matched replacements",
                "faithfulness testing for nominated circuits",
            ),
        ),
        MethodCard(
            name="nnsight_model_adapter",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "reproducible NNsight trace-time activation capture and replacement",
            ),
            limitations=(
                "tuple outputs require explicit selectors",
                "remote and model-specific tracing behavior remains an external dependency",
            ),
            required_controls=(
                "forward-order trace access",
                "explicit output selector for structured outputs",
                "faithfulness testing for nominated circuits",
            ),
        ),
        MethodCard(
            name="saelens_feature_adapter",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "SAE feature encoding, decoding, reconstruction, and feature-subset interventions",
            ),
            limitations=(
                "SAE reconstruction error can change the downstream metric before any feature edit",
                "SAE features need not correspond to unique causal concepts",
            ),
            required_controls=(
                "report original-versus-reconstruction metric gap",
                "equal-cardinality random feature sets",
                "held-out activations",
                "multiple SAE seeds or dictionaries for strong claims",
            ),
        ),
        MethodCard(
            name="circuit_tracer_attribution_adapter",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "normalized feature-to-target attribution scores from circuit-tracer graphs",
            ),
            limitations=(
                "attribution edges are not causal intervention effects",
                "candidate circuits require separate necessity and sufficiency tests",
            ),
            required_controls=(
                "intervention-based faithfulness evaluation",
                "equal-cardinality random circuits",
                "held-out prompts or examples",
            ),
        ),
    )
    for card in cards:
        register_method_card(card, replace=True)
