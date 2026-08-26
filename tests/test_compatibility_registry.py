import pytest

from neuros.compatibility import (
    EvidenceTier,
    IntegrationStatus,
    compatibility_inventory,
    compatibility_payload,
    get_integration,
)


def test_registry_ids_are_unique_and_deterministic():
    records = compatibility_inventory()
    ids = [record.integration_id for record in records]

    assert len(ids) == len(set(ids))
    assert ids[:6] == ["brainflow", "lsl", "mne", "nwb", "zarr", "moabb"]


def test_supported_integrations_have_evidence_and_tier():
    supported = [
        item for item in compatibility_inventory() if item.status is IntegrationStatus.SUPPORTED
    ]

    assert supported
    for record in supported:
        assert record.evidence_tier is not None
        assert record.evidence_paths


def test_every_evidence_bearing_integration_has_executable_paths():
    for record in compatibility_inventory():
        if record.evidence_tier is not None:
            assert record.evidence_paths


def test_planned_integrations_cannot_claim_evidence_tier():
    planned = [
        item for item in compatibility_inventory() if item.status is IntegrationStatus.PLANNED
    ]

    assert planned
    for record in planned:
        assert record.evidence_tier is None
        assert not record.evidence_paths


def test_live_sources_stop_below_hardware_qualification():
    brainflow = get_integration("brainflow")
    lsl = get_integration("lsl")

    assert brainflow.evidence_tier is EvidenceTier.SOFTWARE_CONTRACT
    assert lsl.evidence_tier is EvidenceTier.SOFTWARE_CONTRACT
    assert "hardware" not in brainflow.capabilities
    assert "hardware" not in lsl.capabilities


def test_mne_is_real_object_interoperability_not_a_preprocessing_claim():
    record = get_integration("mne")

    assert record.status is IntegrationStatus.SUPPORTED
    assert record.evidence_tier is EvidenceTier.INTEGRATION
    assert "signalframe-bridge" in record.capabilities
    assert "preprocessing" not in record.capabilities


def test_moabb_real_dataset_evidence_remains_experimental_surface():
    record = get_integration("moabb")

    assert record.status is IntegrationStatus.EXPERIMENTAL
    assert record.evidence_tier is EvidenceTier.REAL_DATASET
    assert "longitudinal-authority" in record.capabilities


def test_braindecode_is_integration_qualified_not_real_dataset_promoted():
    record = get_integration("braindecode")

    assert record.status is IntegrationStatus.EXPERIMENTAL
    assert record.evidence_tier is EvidenceTier.INTEGRATION
    assert "neural-window" in record.capabilities
    assert "training-bridge" in record.capabilities
    assert "tests/test_braindecode_adapter.py" in record.evidence_paths
    assert ".github/workflows/braindecode-ci.yml" in record.evidence_paths
    assert record.install_hint == 'pip install "neuros[braindecode]"'
    assert "Real-dataset" in record.notes
    assert "mechint" not in record.capabilities
    assert "hardware" not in record.capabilities


def test_snap_is_a_numerical_evidence_contract_not_a_biological_claim():
    record = get_integration("snap")

    assert record.status is IntegrationStatus.EXPERIMENTAL
    assert record.evidence_tier is EvidenceTier.SOFTWARE_CONTRACT
    assert "null-space-invariant-evidence" in record.capabilities
    assert "reproduced-paper" not in record.capabilities
    assert "biological" not in record.capabilities
    assert "spectral_alignment.py" in " ".join(record.evidence_paths)


def test_ngclearn_governed_hebbian_adaptation_is_integration_qualified_only():
    record = get_integration("ngclearn")

    assert record.status is IntegrationStatus.EXPERIMENTAL
    assert record.evidence_tier is EvidenceTier.INTEGRATION
    assert "rate-cell-transform" in record.capabilities
    assert "predictive-reconstruction" in record.capabilities
    assert "iterative-error-feedback" in record.capabilities
    assert "hebbian-predictive-adaptation" in record.capabilities
    assert "adaptation-authority-binding" in record.capabilities
    assert "exact-learning-state-rollback" in record.capabilities
    assert "canonical-adaptation-input-identity" in record.capabilities
    assert "spiking-network" not in record.capabilities
    assert "stdp-learning" not in record.capabilities
    assert "online-adaptation" not in record.capabilities
    assert "real-dataset-utility" not in record.capabilities
    assert "calibration-reduction" not in record.capabilities
    assert record.install_hint == 'pip install "neuros-foundation[ngclearn]"'

    evidence = " ".join(record.evidence_paths)
    assert "ngclearn_predictive_coding.py" in evidence
    assert "test_ngclearn_predictive_coding.py" in evidence
    assert "ngclearn_hebbian.py" in evidence
    assert "test_ngclearn_hebbian.py" in evidence
    assert "run_ngclearn_hebbian_authority.py" in evidence
    assert "test_ngclearn_hebbian_authority.py" in evidence
    assert "ngclearn-hebbian-ci.yml" in evidence

    assert "3.2.x" in record.notes
    assert "HebbianSynapse" in record.notes
    assert "optimizer-state" in record.notes
    assert "not an untouched final assessment set" in record.notes
    assert "remain unqualified" in record.notes


def test_research_organizations_are_not_promoted_as_monolithic_integrations():
    ids = {record.integration_id for record in compatibility_inventory()}
    assert "neuroailab" not in ids
    assert "chung-neuroai-lab" not in ids
    assert "mouse-vision" in ids
    assert "tdann" in ids


def test_legacy_neuroaikit_is_planned_and_isolated():
    record = get_integration("neuroaikit")
    assert record.status is IntegrationStatus.PLANNED
    assert record.evidence_tier is None
    assert record.evidence_paths == ()
    assert "isolated-snu-reference-worker" in record.capabilities


def test_openbci_is_indirect_not_hardware_qualified():
    record = get_integration("openbci")

    assert record.status is IntegrationStatus.INDIRECT
    assert record.evidence_tier is None
    assert "BrainFlow" in record.notes


def test_payload_is_json_friendly_and_filterable():
    payload = compatibility_payload("lsl")

    assert len(payload) == 1
    assert payload[0]["integration_id"] == "lsl"
    assert payload[0]["status"] == "supported"
    assert isinstance(payload[0]["capabilities"], list)
    assert isinstance(payload[0]["evidence_paths"], list)

    planned = compatibility_payload(status="planned")
    assert planned
    assert {item["status"] for item in planned} == {"planned"}


def test_unknown_integration_fails_with_known_ids():
    with pytest.raises(KeyError, match="Unknown integration.*brainflow.*lsl.*mne"):
        get_integration("not-real")


def test_invalid_status_filter_fails_closed():
    with pytest.raises(ValueError):
        compatibility_payload(status="marketing-approved")
