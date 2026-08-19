import json

import pytest

from neuros_mechint import EvidenceTier, ExperimentManifest, stable_hash
from neuros_mechint.benchmarks import (
    DoseResponseObservation,
    DoseResponsePolicy,
    DoseResponseSpec,
    InterventionManifoldAssumption,
    InterventionManifoldKind,
    ReproductionMetricTolerance,
    ReproductionSnapshot,
    ReproductionSpec,
    analyze_dose_response,
    assess_independent_reproduction,
    read_dose_response_artifact,
    run_v1_release_contract_benchmark,
    write_dose_response_artifact,
)
from neuros_mechint.core import migrate_artifact_envelope, migrate_manifest_payload, schema_catalog
from neuros_mechint.release import default_v1_evidence_status


def test_manifest_has_deterministic_scientific_fingerprint_separate_from_run_hash():
    left = ExperimentManifest(
        experiment_name="same-study",
        method="unit",
        model_id="model",
        model_revision="rev-a",
        model_hash="model-hash",
        dataset_id="data",
        dataset_hash="data-hash",
        parameters={"window": 10},
        evidence_tier=EvidenceTier.CONTRACT,
    )
    right = ExperimentManifest(
        experiment_name="same-study",
        method="unit",
        model_id="model",
        model_revision="rev-a",
        model_hash="model-hash",
        dataset_id="data",
        dataset_hash="data-hash",
        parameters={"window": 10},
        evidence_tier=EvidenceTier.CONTRACT,
    )
    assert left.scientific_fingerprint == right.scientific_fingerprint
    assert left.to_dict()["schema_version"] == "3"
    assert left.run_hash == left.content_hash


def test_v2_manifest_migrates_without_runtime_fields_entering_scientific_identity():
    payload = {
        "schema_version": "2",
        "experiment_name": "legacy",
        "method": "unit",
        "method_version": "1",
        "model_id": "m",
        "model_revision": "r",
        "model_hash": "mh",
        "dataset_id": "d",
        "dataset_hash": "dh",
        "parameters": {"x": 1},
        "seed": 3,
        "evidence_tier": {"level": 2, "label": "contract"},
        "benchmark": {"created_at": "runtime-only"},
    }
    migrated = migrate_manifest_payload(payload)
    assert migrated["schema_version"] == "3"
    assert migrated["scientific_fingerprint"] == stable_hash(migrated["scientific_identity"])
    assert "benchmark" not in migrated["scientific_identity"]


def test_pre_v1_artifact_envelope_gets_contract_without_changing_result_hash():
    spec = next(item for item in schema_catalog() if item["family"] == "evidence_pack")
    result = {"schema_version": spec["result_schema"], "study_fingerprint": "known"}
    legacy = {
        "artifact_schema": spec["artifact_schema"],
        "artifact_hash": stable_hash(result),
        "result": result,
    }
    migrated = migrate_artifact_envelope(legacy, family="evidence_pack")
    assert migrated["artifact_hash"] == legacy["artifact_hash"]
    assert migrated["result"] == result
    assert migrated["contract"]["family"] == "evidence_pack"


def test_independent_reproduction_requires_new_execution_and_same_decision():
    spec = ReproductionSpec(
        reproduction_id="r1",
        artifact_family="correspondence",
        required_decision="promoted",
        metric_tolerances=(ReproductionMetricTolerance("recovery", absolute=0.02, relative=0.0),),
    )
    reference = ReproductionSnapshot(
        "correspondence", "study", "run-a", "exec-a", "promoted", {"recovery": 0.8}
    )
    good = ReproductionSnapshot(
        "correspondence", "study", "run-b", "exec-b", "promoted", {"recovery": 0.81}
    )
    duplicate = ReproductionSnapshot(
        "correspondence", "study", "run-a", "exec-a", "promoted", {"recovery": 0.8}
    )
    flipped = ReproductionSnapshot(
        "correspondence", "study", "run-c", "exec-c", "rejected", {"recovery": 0.81}
    )
    assert assess_independent_reproduction(spec, reference, good).passed
    assert not assess_independent_reproduction(spec, reference, duplicate).passed
    assert not assess_independent_reproduction(spec, reference, flipped).passed


def test_dose_response_artifact_roundtrip_and_tamper_detection(tmp_path):
    manifold = InterventionManifoldAssumption(
        InterventionManifoldKind.EMPIRICAL_DONOR,
        "held-out discovery donor pool",
        donor_pool_id="discovery-donors",
        expected_in_manifold=True,
    )
    spec = DoseResponseSpec(
        study_id="dose-v1",
        intervention_id="mapped-feature-substitution",
        expected_direction=1,
        manifold=manifold,
        policy=DoseResponsePolicy(min_units=3, min_doses=5),
    )
    observations = tuple(
        DoseResponseObservation(f"seed-{seed}", dose, dose + seed * 0.001)
        for seed in range(3)
        for dose in (0.0, 0.25, 0.5, 0.75, 1.0)
    )
    result = analyze_dose_response(spec, observations)
    assert result.passed
    path = write_dose_response_artifact(result, tmp_path / "dose.json")
    loaded = read_dose_response_artifact(path)
    assert loaded["schema_version"] == "neuros-mechint.dose-response-study.v1"
    assert loaded["passed"] is True

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["endpoint_effect"] = 999.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_dose_response_artifact(path)


def test_release_status_refuses_to_invent_empirical_evidence():
    status = default_v1_evidence_status()
    assert status.software_contract_ready
    assert not status.empirical_evidence_complete
    assert "real-neural-factorial-study" in status.pending_empirical_requirements


def test_v1_contract_ground_truth_passes_and_requires_empirical_pending_state():
    report = run_v1_release_contract_benchmark()
    assert report.passed
    assert report.empirical_overclaim_rejected
