import pytest

from neuros_mechint import EvidenceTier, get_method_card


def test_evidence_tiers_match_repository_ladder():
    assert EvidenceTier.UNIT == 1
    assert EvidenceTier.SCIENTIFIC_SYNTHETIC == 5
    assert EvidenceTier.CLINICAL_EVIDENCE == 9
    assert EvidenceTier.coerce("real dataset") is EvidenceTier.REAL_DATASET


def test_maturity_and_evidence_are_distinct_contracts():
    card = get_method_card("orion_token_causal_audit")
    assert card.maturity.value == "integrated"
    assert "matched random windows" in card.required_controls


def test_unknown_evidence_tier_fails_loudly():
    with pytest.raises(ValueError):
        EvidenceTier.coerce("trust me")
