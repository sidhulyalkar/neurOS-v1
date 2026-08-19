"""v0.7 method cards for factorial mechanism studies."""

from __future__ import annotations

from .maturity import MethodCard, MethodMaturity, register_method_card


def register_v07_method_cards() -> None:
    """Register v0.7 factorial-science claim boundaries idempotently."""

    cards = (
        MethodCard(
            name="factorial_mechanism_design",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "an explicit architecture x tokenizer study design with pinned cell identities",
                "missing cells, matched covariates, and preregistered contrasts are preserved",
                "non-estimable contrasts remain visible instead of being silently dropped",
            ),
            limitations=(
                "design validity depends on correctly declared covariates and scientific axes",
                "a factorial contrast is conditional on the chosen cell-level mechanism outcomes",
            ),
            required_controls=(
                "held-out evidence pack for every observed cell",
                "pinned model, tokenizer, dataset, and checkpoint revisions",
                "explicit missing-cell reasons",
                "matched target universes or declared target-overlap policy",
            ),
        ),
        MethodCard(
            name="factorial_mechanism_contrast",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "matched architecture, tokenizer, checkpoint, and architecture-tokenizer interaction contrasts",
                "difference-in-differences interaction estimates over held-out mechanism outcomes",
                "confound rejection using task performance, checkpoint maturity, and declared covariates",
            ),
            limitations=(
                "difference-in-differences does not prove a unique mechanistic cause",
                "task-performance matching can reduce but not remove every training confound",
                "small factorial grids do not characterize the entire architecture or tokenizer family",
            ),
            required_controls=(
                "preregistered contrast definitions",
                "matched training seed/session/subject/dataset where the contrast does not vary them",
                "matched token budget, temporal resolution, capacity, and training compute where claimed",
                "held-out necessity, sufficiency, and same-size controls within each cell",
            ),
        ),
        MethodCard(
            name="cross_session_factorial_replication",
            maturity=MethodMaturity.RESEARCH,
            establishes=(
                "replication summaries for the same preregistered contrast across distinct sessions",
                "sign agreement and median effect across estimable replications",
            ),
            limitations=(
                "session replication does not replace subject-level or training-seed uncertainty",
                "sign agreement can hide magnitude heterogeneity",
            ),
            required_controls=(
                "at least two distinct sessions",
                "identical contrast semantics and matched-covariate policy",
                "report every non-estimable replication",
            ),
        ),
        MethodCard(
            name="factorial_evidence_pack_bridge",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "conversion of v0.6 held-out evidence packs into v0.7 factorial cell outcomes",
                "optional attachment of v0.3/v0.4 causal effect maps without treating geometry as faithfulness",
            ),
            limitations=(
                "the bridge does not improve weak or confounded source evidence packs",
                "effect-map correspondence still requires aligned intervention targets",
            ),
            required_controls=(
                "source study fingerprints and run hashes",
                "evidence-pack provenance must match the declared factorial cell",
                "causal-map context metadata must agree with the cell when available",
            ),
        ),
    )
    for card in cards:
        register_method_card(card, replace=True)
