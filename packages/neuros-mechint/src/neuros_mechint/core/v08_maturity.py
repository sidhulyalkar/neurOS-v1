"""v0.8 method cards for causal feature correspondence."""

from __future__ import annotations

from .maturity import MethodCard, MethodMaturity, register_method_card


def register_v08_method_cards() -> None:
    cards = (
        MethodCard(
            name="feature_correspondence_design",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "a frozen source-to-target feature mapping fit only on discovery examples",
                "separate held-out similarity, predictive-transfer, and causal-transfer evidence",
            ),
            limitations=(
                "linear ridge mapping is not a universal representation-alignment method",
                "a statistically valid mapping is not itself causal correspondence",
            ),
            required_controls=(
                "content-distinct discovery and validation partitions",
                "immutable feature-space identities and model/data revisions",
                "same-cardinality random-source mappings",
                "shuffled semantic-pair donors",
            ),
        ),
        MethodCard(
            name="held_out_causal_feature_substitution",
            maturity=MethodMaturity.RESEARCH,
            establishes=(
                "conditional causal substitutability of a frozen source feature mapping for target features",
                "source and target causal relevance under the declared intervention backend",
            ),
            limitations=(
                "feature replacement can be off manifold",
                "successful substitution does not establish feature uniqueness or biological homology",
                "the result depends on the declared feature projector and metric",
            ),
            required_controls=(
                "source-feature ablation",
                "target-feature ablation",
                "shuffled source-example substitution",
                "same-cardinality random-source correspondence controls",
                "held-out validation examples unavailable to mapping fit",
            ),
        ),
        MethodCard(
            name="model_adapter_feature_correspondence",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "execution of paired feature capture, ablation, and substitution through ModelAdapter",
            ),
            limitations=(
                "the default tensor projector averages non-feature axes",
                "temporal or token-specific claims require a projector that preserves those semantics",
            ),
            required_controls=(
                "model-state mutation guards",
                "explicit feature-axis semantics",
                "matched semantic trials across source and target models",
            ),
        ),
        MethodCard(
            name="factorial_to_correspondence_link",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "provenance linking a v0.8 correspondence study to an estimable v0.7 contrast",
            ),
            limitations=(
                "an upstream factorial interaction nominates a comparison; it does not identify corresponding features",
            ),
            required_controls=(
                "estimable upstream factorial contrast",
                "frozen factorial-study fingerprint",
                "independent held-out correspondence validation",
            ),
        ),
    )
    for card in cards:
        register_method_card(card, replace=True)
