import pytest

from neuros.compatibility import compatibility_inventory, compatibility_payload, get_integration


def test_registry_ids_are_unique_and_deterministic():
    records = compatibility_inventory()
    ids = [record.integration_id for record in records]

    assert len(ids) == len(set(ids))
    assert ids[:5] == ["brainflow", "lsl", "nwb", "zarr", "moabb"]


def test_supported_integrations_have_evidence_and_tier():
    supported = [item for item in compatibility_inventory() if item.status == "supported"]

    assert supported
    for record in supported:
        assert record.evidence_tier is not None
        assert record.evidence_paths


def test_planned_integrations_cannot_claim_qualification_tier():
    planned = [item for item in compatibility_inventory() if item.status == "planned"]

    assert planned
    for record in planned:
        assert record.evidence_tier is None


def test_live_sources_stop_at_software_contract_tier():
    brainflow = get_integration("brainflow")
    lsl = get_integration("lsl")

    assert brainflow.evidence_tier == "software-contract"
    assert lsl.evidence_tier == "software-contract"
    assert "hardware" not in brainflow.capabilities
    assert "hardware" not in lsl.capabilities


def test_openbci_is_indirect_not_hardware_qualified():
    record = get_integration("openbci")

    assert record.status == "indirect"
    assert record.evidence_tier is None
    assert "BrainFlow" in record.notes


def test_payload_is_json_friendly_and_filterable():
    payload = compatibility_payload("lsl")

    assert len(payload) == 1
    assert payload[0]["integration_id"] == "lsl"
    assert isinstance(payload[0]["capabilities"], tuple)


def test_unknown_integration_fails_with_known_ids():
    with pytest.raises(KeyError, match="Unknown integration.*brainflow.*lsl"):
        get_integration("not-real")
