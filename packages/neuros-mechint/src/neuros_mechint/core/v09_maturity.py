"""v0.9 method cards for hierarchical replication and intervention robustness."""

from __future__ import annotations

from .maturity import MethodCard, MethodMaturity, register_method_card


def register_v09_method_cards() -> None:
    cards = (
        MethodCard(
            name="claim_aware_hierarchical_replication",
            maturity=MethodMaturity.RESEARCH,
            establishes=(
                "uncertainty and replication at an explicitly declared independent scientific unit",
                "unit-balanced effects that do not weight a claim by lower-level sample count",
            ),
            limitations=(
                "hierarchical bootstrap intervals depend on the declared nesting structure",
                "few independent subjects or model seeds can still yield unstable uncertainty estimates",
                "replication under one hierarchy does not imply transfer to undeclared datasets or tasks",
            ),
            required_controls=(
                "preregistered claim axis and null",
                "explicit model-seed, subject, session, and trial identities as applicable",
                "minimum independent-unit count",
                "independent-unit sign agreement",
                "negative and non-estimable replicas retained in the analysis",
            ),
        ),
        MethodCard(
            name="hierarchical_factorial_uncertainty",
            maturity=MethodMaturity.INTEGRATED,
            establishes=(
                "conversion of estimable v0.7 factorial effects into claim-aware v0.9 replication observations",
            ),
            limitations=(
                "the bridge does not repair a non-estimable factorial design",
                "valid random-effects interpretation still depends on independent training/data units",
            ),
            required_controls=(
                "estimability status preserved from every source contrast",
                "replication family fixed before aggregation",
                "claim axis chosen above repeated lower-level observations",
            ),
        ),
        MethodCard(
            name="correspondence_replication",
            maturity=MethodMaturity.RESEARCH,
            establishes=(
                "stability of v0.8 causal correspondence metrics across declared independent replicas",
            ),
            limitations=(
                "a replicated correspondence remains conditional on feature surface, projector, and intervention family",
                "dictionary/projector robustness requires independent dictionary or projector replicas when claimed",
            ),
            required_controls=(
                "source v0.8 study fingerprints",
                "independent model, subject, dictionary, or dataset units appropriate to the claim",
                "hierarchical confidence interval and sign agreement",
            ),
        ),
        MethodCard(
            name="intervention_dose_response",
            maturity=MethodMaturity.RESEARCH,
            establishes=(
                "whether intervention effect changes coherently over a preregistered dose grid",
                "explicit activation-manifold assumptions for replacement interventions",
            ),
            limitations=(
                "monotonic response is supportive rather than sufficient evidence of mechanism",
                "zero and mean interventions can remain off manifold",
                "in-manifold expectations depend on the donor or generator construction",
            ),
            required_controls=(
                "dose 0 and dose 1 endpoints",
                "multiple independent dose-response units",
                "common dose grid or an explicitly different analysis",
                "donor-pool and discovery-fit provenance for learned manifold controls",
            ),
        ),
    )
    for card in cards:
        register_method_card(card, replace=True)
