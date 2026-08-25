from __future__ import annotations

import json
from pathlib import Path

import pytest

from neuros.hardware_qualification import (
    HardwareQualificationThresholds,
    MeasurementOrigin,
    evaluate_hardware_evidence_bundle,
    evaluate_hardware_qualification,
    manifest_from_mapping,
)
from neuros.qualification import qualify_config


def synthetic_evidence() -> dict:
    return {
        "schema_version": 1,
        "evidence_id": "contract-fixture",
        "measurement_origin": "synthetic_contract_test",
        "qualification_bundle_sha256": "a" * 64,
        "physical_run": False,
        "synthetic_contract_test": True,
        "device": {
            "manufacturer": "Synthetic",
            "device": "Contract Fixture",
            "board_id": "fixture-0",
            "firmware_version": "0",
            "acquisition_library": "neuros-test",
            "acquisition_library_version": "1",
            "operating_system": "ci",
            "transport": "memory",
        },
        "signal": {
            "channel_names": ["C3", "C4"],
            "channel_types": ["eeg", "eeg"],
            "units": ["uV", "uV"],
            "nominal_sample_rate_hz": 250.0,
            "measured_sample_rate_hz": 249.95,
        },
        "timing": {
            "timestamp_source": "synthetic_monotonic",
            "clock_domain": "synchronized",
            "offset_p50_ms": 0.2,
            "offset_p95_ms": 0.5,
            "drift_ppm": 5.0,
            "uncertainty_p95_ms": 0.8,
        },
        "reliability": {
            "duration_s": 600.0,
            "expected_samples": 150000,
            "observed_samples": 150000,
            "queue_accepted": 150000,
            "queue_dropped": 0,
            "reconnect_attempts": 1,
            "reconnect_successes": 1,
            "reconnect_tested": True,
        },
        "latency": {
            "source_to_decision_p50_ms": 18.0,
            "source_to_decision_p95_ms": 35.0,
            "source_to_decision_p99_ms": 52.0,
            "sample_count": 1000,
        },
        "notes": {"purpose": "contract test only"},
    }


def test_synthetic_fixture_can_pass_thresholds_but_never_hardware_qualification():
    manifest = manifest_from_mapping(synthetic_evidence())
    result = evaluate_hardware_qualification(
        manifest,
        HardwareQualificationThresholds(require_reconnect_test=True),
    )

    assert manifest.measurement_origin is MeasurementOrigin.SYNTHETIC
    assert result.schema_valid is True
    assert result.measurements_complete is True
    assert result.thresholds_pass is True
    assert result.physical_evidence is False
    assert result.qualification_bundle_verified is False
    assert result.hardware_qualified is False
    assert result.claim_boundary["hardware_qualified"] is False
    physical_gate = next(gate for gate in result.gates if gate.name == "physical_measurement")
    assert physical_gate.status.value == "fail"
    bundle_gate = next(gate for gate in result.gates if gate.name == "verified_runtime_bundle")
    assert bundle_gate.status.value == "not_tested"


def test_matching_root_value_without_actual_bundle_verification_is_not_enough():
    manifest = manifest_from_mapping(synthetic_evidence())
    result = evaluate_hardware_qualification(
        manifest,
        verified_qualification_bundle_sha256="a" * 64,
    )

    assert result.qualification_bundle_verified is True
    assert result.thresholds_pass is True
    assert result.hardware_qualified is False
    assert result.physical_evidence is False


@pytest.mark.asyncio
async def test_synthetic_evidence_can_bind_real_runtime_bundle_but_still_cannot_promote(tmp_path: Path):
    config = Path("configs/examples/mock_bci.yaml").resolve()
    bundle = tmp_path / "qualification"
    qualification = await qualify_config(config, bundle, duration_s=0.04)

    raw = synthetic_evidence()
    raw["qualification_bundle_sha256"] = qualification["bundle_sha256"]
    manifest = manifest_from_mapping(raw)
    result = evaluate_hardware_evidence_bundle(manifest, bundle)

    assert result.qualification_bundle_verified is True
    assert result.claim_boundary["runtime_record_replay_qualified"] is True
    assert result.thresholds_pass is True
    assert result.physical_evidence is False
    assert result.hardware_qualified is False


def test_threshold_failure_is_reported_without_changing_measurements():
    raw = synthetic_evidence()
    raw["reliability"]["observed_samples"] = 149000
    manifest = manifest_from_mapping(raw)
    thresholds = HardwareQualificationThresholds(max_sample_loss_fraction=0.001)
    result = evaluate_hardware_qualification(manifest, thresholds)

    assert result.thresholds_pass is False
    loss_gate = next(gate for gate in result.gates if gate.name == "sample_loss_fraction")
    assert loss_gate.status.value == "fail"
    assert loss_gate.observed == pytest.approx(1000 / 150000)
    assert result.hardware_qualified is False


def test_physical_origin_cannot_be_declared_without_physical_run():
    raw = synthetic_evidence()
    raw["measurement_origin"] = "physical_measurement"
    raw["synthetic_contract_test"] = False

    with pytest.raises(ValueError, match="requires physical_run=true"):
        manifest_from_mapping(raw)


def test_synthetic_flag_blocks_physical_origin():
    raw = synthetic_evidence()
    raw["measurement_origin"] = "physical_measurement"
    raw["physical_run"] = True

    with pytest.raises(ValueError, match="must use synthetic_contract_test origin"):
        manifest_from_mapping(raw)


def test_imported_measurement_also_requires_a_physical_run():
    raw = synthetic_evidence()
    raw["measurement_origin"] = "imported_external_measurement"
    raw["synthetic_contract_test"] = False

    with pytest.raises(ValueError, match="requires physical_run=true"):
        manifest_from_mapping(raw)


def test_geometry_and_percentiles_are_fail_closed():
    raw = synthetic_evidence()
    raw["signal"]["channel_types"] = ["eeg"]
    with pytest.raises(ValueError, match="identical lengths"):
        manifest_from_mapping(raw)

    raw = synthetic_evidence()
    raw["latency"]["source_to_decision_p50_ms"] = 60.0
    with pytest.raises(ValueError, match="p50 <= p95 <= p99"):
        manifest_from_mapping(raw)


def test_invalid_qualification_root_is_rejected():
    raw = synthetic_evidence()
    raw["qualification_bundle_sha256"] = "not-a-digest"
    with pytest.raises(ValueError, match="64-character SHA-256"):
        manifest_from_mapping(raw)


def test_serialized_result_exposes_gate_evidence(tmp_path: Path):
    manifest = manifest_from_mapping(synthetic_evidence())
    result = evaluate_hardware_qualification(manifest)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(result.to_dict(), indent=2, default=str), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))

    assert parsed["hardware_qualified"] is False
    assert parsed["thresholds_pass"] is True
    assert parsed["qualification_bundle_verified"] is False
    assert len(parsed["evidence_sha256"]) == 64
    assert {gate["name"] for gate in parsed["gates"]} >= {
        "physical_measurement",
        "verified_runtime_bundle",
        "minimum_duration",
        "sample_loss_fraction",
        "queue_drop_fraction",
        "absolute_clock_drift_ppm",
        "source_to_decision_p95_ms",
    }
