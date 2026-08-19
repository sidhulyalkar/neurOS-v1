"""v0.6 method cards for held-out evidence packs."""

from __future__ import annotations

from .maturity import MethodCard, MethodMaturity, register_method_card


def register_v06_method_cards() -> None:
    """Register v0.6 scientific claim boundaries idempotently."""

    cards = (
        MethodCard(
            name="held_out_circuit_evidence_pack",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "candidate selection and intervention-donor fitting occur before held-out evaluation",
                "cross-example necessity/sufficiency summaries with explicit promotion criteria",
                "structured provenance, runtime telemetry, and self-checking JSON artifacts",
            ),
            limitations=(
                "held-out validation is only as independent as the supplied split",
                "a passing pack remains conditional on the metric, target universe, and interventions",
                "bootstrap intervals quantify example resampling, not every source of model uncertainty",
            ),
            required_controls=(
                "immutable discovery/validation example identities",
                "at least two held-out validation examples for promoted evidence",
                "equal-cardinality random circuits",
                "multiple intervention baselines",
                "frozen model, tokenizer, dataset, and external-tool revisions",
            ),
        ),
        MethodCard(
            name="discovery_single_target_ablation",
            maturity=MethodMaturity.RESEARCH,
            establishes=(
                "a reproducible component ranking by discovery-set single-target ablation effect",
            ),
            limitations=(
                "individual ablations can miss synergistic or redundant circuits",
                "the ranking is a candidate generator rather than a circuit-identification theorem",
            ),
            required_controls=(
                "discovery data only",
                "held-out circuit faithfulness after freezing the candidate",
                "comparison with activation-magnitude and random baselines",
            ),
        ),
        MethodCard(
            name="discovery_activation_magnitude_baseline",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "a deterministic same-size baseline ranked by discovery activation magnitude",
            ),
            limitations=(
                "activation magnitude is not causal importance",
                "different hook-point scales can make raw magnitude comparisons misleading",
            ),
            required_controls=(
                "discovery examples only",
                "same candidate cardinality as the primary method",
                "held-out comparison using identical interventions",
            ),
        ),
        MethodCard(
            name="external_model_evidence_recipe",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "a maintained execution recipe for supported external interpretability ecosystems",
            ),
            limitations=(
                "a recipe is not measured evidence",
                "mutable upstream aliases must be resolved to immutable revisions before publication",
            ),
            required_controls=(
                "record exact model/tokenizer/SAE/transcoder revisions",
                "publish the generated evidence-pack artifact",
                "report negative held-out results as well as successes",
            ),
        ),
    )
    for card in cards:
        register_method_card(card, replace=True)
