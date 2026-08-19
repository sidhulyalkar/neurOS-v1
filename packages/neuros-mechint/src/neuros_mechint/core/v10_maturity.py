"""v1 method-maturity registrations."""

from __future__ import annotations

from .maturity import MethodCard, MethodMaturity, register_method_card


def register_v10_method_cards() -> None:
    cards = (
        MethodCard(
            name="versioned_artifact_schema_contract",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "machine-readable identity and integrity rules for published artifact families",
                "backwards migration of supported pre-v1 manifest/artifact envelopes",
            ),
            limitations=(
                "schema compatibility does not establish scientific validity of artifact contents",
            ),
            required_controls=("hash validation", "migration compatibility tests"),
        ),
        MethodCard(
            name="independent_artifact_reproduction",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "whether a distinct execution recovers a preregistered qualitative decision within numerical tolerances",
            ),
            limitations=(
                "distinct execution IDs do not prove organizational or statistical independence",
                "reproduction of a flawed protocol reproduces the flaw",
            ),
            required_controls=(
                "same scientific fingerprint",
                "distinct run hash and execution identity",
                "preregistered metric tolerances",
            ),
        ),
        MethodCard(
            name="v1_evidence_closure_reporting",
            maturity=MethodMaturity.STABLE,
            establishes=(
                "explicit separation of software-contract readiness from empirical evidence completion",
            ),
            limitations=(
                "pending empirical requirements must be satisfied by real study artifacts outside synthetic CI",
            ),
            required_controls=("published requirement status", "artifact fingerprints for satisfied evidence"),
        ),
    )
    for card in cards:
        try:
            register_method_card(card)
        except ValueError:
            continue
