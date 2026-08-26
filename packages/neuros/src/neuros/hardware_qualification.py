"""Fail-closed hardware evidence contracts for neurOS.

This module defines *what must be measured* before neurOS can promote a named
physical acquisition configuration to the hardware evidence tier. It does not
measure hardware itself and software CI must never fabricate that distinction.

A schema-valid synthetic fixture can exercise every threshold gate while still
being structurally ineligible for ``hardware_qualified=True``. Hardware
promotion additionally requires an independently verified neurOS qualification
bundle root so physical measurements are bound to the exact computational run.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

HARDWARE_QUALIFICATION_SCHEMA_VERSION = 1


class MeasurementOrigin(str, Enum):
    PHYSICAL = "physical_measurement"
    SYNTHETIC = "synthetic_contract_test"
    IMPORTED = "imported_external_measurement"


class GateStatus(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    NOT_TESTED = "not_tested"


@dataclass(frozen=True, slots=True)
class DeviceIdentity:
    manufacturer: str
    device: str
    board_id: str
    firmware_version: str
    acquisition_library: str
    acquisition_library_version: str
    operating_system: str
    transport: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"device.{name} must be a non-empty string")


@dataclass(frozen=True, slots=True)
class SignalGeometry:
    channel_names: tuple[str, ...]
    channel_types: tuple[str, ...]
    units: tuple[str, ...]
    nominal_sample_rate_hz: float
    measured_sample_rate_hz: float

    def validate(self) -> None:
        count = len(self.channel_names)
        if count <= 0:
            raise ValueError("signal.channel_names must contain at least one channel")
        if len(set(self.channel_names)) != count:
            raise ValueError("signal.channel_names must be unique")
        if len(self.channel_types) != count or len(self.units) != count:
            raise ValueError("signal channel names/types/units must have identical lengths")
        if any(not item.strip() for item in self.channel_names):
            raise ValueError("signal.channel_names must not contain empty values")
        if any(not item.strip() for item in self.channel_types):
            raise ValueError("signal.channel_types must not contain empty values")
        if any(not item.strip() for item in self.units):
            raise ValueError("signal.units must not contain empty values")
        if self.nominal_sample_rate_hz <= 0 or not math.isfinite(self.nominal_sample_rate_hz):
            raise ValueError("signal.nominal_sample_rate_hz must be positive and finite")
        if self.measured_sample_rate_hz <= 0 or not math.isfinite(self.measured_sample_rate_hz):
            raise ValueError("signal.measured_sample_rate_hz must be positive and finite")


@dataclass(frozen=True, slots=True)
class TimingEvidence:
    timestamp_source: str
    clock_domain: str
    offset_p50_ms: float
    offset_p95_ms: float
    drift_ppm: float
    uncertainty_p95_ms: float

    def validate(self) -> None:
        if not isinstance(self.timestamp_source, str) or not self.timestamp_source.strip():
            raise ValueError("timing.timestamp_source must be a non-empty string")
        if not isinstance(self.clock_domain, str) or not self.clock_domain.strip():
            raise ValueError("timing.clock_domain must be a non-empty string")
        values = (
            self.offset_p50_ms,
            self.offset_p95_ms,
            self.drift_ppm,
            self.uncertainty_p95_ms,
        )
        if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values):
            raise ValueError("timing measurements must be numeric")
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("timing measurements must be finite")
        if self.offset_p50_ms > self.offset_p95_ms:
            raise ValueError("timing offset percentiles must satisfy p50 <= p95")
        if self.uncertainty_p95_ms < 0:
            raise ValueError("timing.uncertainty_p95_ms must be non-negative")


@dataclass(frozen=True, slots=True)
class ReliabilityEvidence:
    duration_s: float
    expected_samples: int
    observed_samples: int
    queue_accepted: int
    queue_dropped: int
    reconnect_attempts: int = 0
    reconnect_successes: int = 0
    reconnect_tested: bool = False

    def validate(self) -> None:
        if isinstance(self.duration_s, bool) or not isinstance(self.duration_s, (int, float)):
            raise ValueError("reliability.duration_s must be numeric")
        if self.duration_s <= 0 or not math.isfinite(float(self.duration_s)):
            raise ValueError("reliability.duration_s must be positive and finite")
        for name in (
            "expected_samples",
            "observed_samples",
            "queue_accepted",
            "queue_dropped",
            "reconnect_attempts",
            "reconnect_successes",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"reliability.{name} must be an integer")
            if value < 0:
                raise ValueError(f"reliability.{name} must be non-negative")
        if not isinstance(self.reconnect_tested, bool):
            raise ValueError("reliability.reconnect_tested must be boolean")
        if self.expected_samples <= 0:
            raise ValueError("reliability.expected_samples must be positive")
        if self.observed_samples > self.expected_samples:
            raise ValueError("reliability.observed_samples cannot exceed expected_samples")
        if self.reconnect_successes > self.reconnect_attempts:
            raise ValueError("reconnect_successes cannot exceed reconnect_attempts")
        if not self.reconnect_tested and (self.reconnect_attempts or self.reconnect_successes):
            raise ValueError("reconnect counters require reconnect_tested=true")

    @property
    def sample_loss_fraction(self) -> float:
        return (self.expected_samples - self.observed_samples) / self.expected_samples

    @property
    def queue_drop_fraction(self) -> float:
        total = self.queue_accepted + self.queue_dropped
        return 0.0 if total == 0 else self.queue_dropped / total


@dataclass(frozen=True, slots=True)
class LatencyEvidence:
    source_to_decision_p50_ms: float
    source_to_decision_p95_ms: float
    source_to_decision_p99_ms: float
    sample_count: int

    def validate(self) -> None:
        values = (
            self.source_to_decision_p50_ms,
            self.source_to_decision_p95_ms,
            self.source_to_decision_p99_ms,
        )
        if not all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value))
            and value >= 0
            for value in values
        ):
            raise ValueError("latency measurements must be numeric, finite, and non-negative")
        if not (
            self.source_to_decision_p50_ms
            <= self.source_to_decision_p95_ms
            <= self.source_to_decision_p99_ms
        ):
            raise ValueError("latency percentiles must satisfy p50 <= p95 <= p99")
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise ValueError("latency.sample_count must be an integer")
        if self.sample_count <= 0:
            raise ValueError("latency.sample_count must be positive")


@dataclass(frozen=True, slots=True)
class HardwareQualificationThresholds:
    min_duration_s: float = 300.0
    max_sample_loss_fraction: float = 0.001
    max_queue_drop_fraction: float = 0.0
    max_sample_rate_error_fraction: float = 0.01
    max_abs_clock_drift_ppm: float = 100.0
    max_clock_uncertainty_p95_ms: float = 5.0
    max_source_to_decision_p95_ms: float = 100.0
    max_source_to_decision_p99_ms: float = 200.0
    require_reconnect_test: bool = False

    def validate(self) -> None:
        if isinstance(self.min_duration_s, bool) or not isinstance(self.min_duration_s, (int, float)):
            raise ValueError("thresholds.min_duration_s must be numeric")
        if self.min_duration_s <= 0:
            raise ValueError("thresholds.min_duration_s must be positive")
        for name in (
            "max_sample_loss_fraction",
            "max_queue_drop_fraction",
            "max_sample_rate_error_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"thresholds.{name} must be numeric")
            if not 0 <= float(value) <= 1:
                raise ValueError(f"thresholds.{name} must be in [0, 1]")
        for name in (
            "max_abs_clock_drift_ppm",
            "max_clock_uncertainty_p95_ms",
            "max_source_to_decision_p95_ms",
            "max_source_to_decision_p99_ms",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"thresholds.{name} must be numeric")
            if value < 0 or not math.isfinite(float(value)):
                raise ValueError(f"thresholds.{name} must be finite and non-negative")
        if not isinstance(self.require_reconnect_test, bool):
            raise ValueError("thresholds.require_reconnect_test must be boolean")
        if self.max_source_to_decision_p99_ms < self.max_source_to_decision_p95_ms:
            raise ValueError("p99 latency threshold cannot be stricter than p95 threshold")


@dataclass(frozen=True, slots=True)
class HardwareQualificationManifest:
    evidence_id: str
    measurement_origin: MeasurementOrigin
    qualification_bundle_sha256: str
    device: DeviceIdentity
    signal: SignalGeometry
    timing: TimingEvidence
    reliability: ReliabilityEvidence
    latency: LatencyEvidence
    physical_run: bool
    synthetic_contract_test: bool = False
    notes: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = HARDWARE_QUALIFICATION_SCHEMA_VERSION

    def validate(self) -> None:
        if self.schema_version != HARDWARE_QUALIFICATION_SCHEMA_VERSION:
            raise ValueError("unsupported hardware qualification schema version")
        if not isinstance(self.evidence_id, str) or not self.evidence_id.strip():
            raise ValueError("evidence_id must be a non-empty string")
        if not isinstance(self.physical_run, bool):
            raise ValueError("physical_run must be boolean")
        if not isinstance(self.synthetic_contract_test, bool):
            raise ValueError("synthetic_contract_test must be boolean")
        _normalize_sha256(self.qualification_bundle_sha256)
        self.device.validate()
        self.signal.validate()
        self.timing.validate()
        self.reliability.validate()
        self.latency.validate()
        if self.measurement_origin is MeasurementOrigin.PHYSICAL and not self.physical_run:
            raise ValueError("physical_measurement origin requires physical_run=true")
        if self.measurement_origin is MeasurementOrigin.IMPORTED and not self.physical_run:
            raise ValueError("imported_external_measurement origin requires physical_run=true")
        if self.measurement_origin is MeasurementOrigin.SYNTHETIC and not self.synthetic_contract_test:
            raise ValueError("synthetic_contract_test origin requires synthetic_contract_test=true")
        if self.synthetic_contract_test and self.measurement_origin is not MeasurementOrigin.SYNTHETIC:
            raise ValueError("synthetic contract evidence must use synthetic_contract_test origin")


@dataclass(frozen=True, slots=True)
class QualificationGate:
    name: str
    status: GateStatus
    observed: Any
    requirement: str


@dataclass(frozen=True, slots=True)
class HardwareQualificationResult:
    evidence_id: str
    schema_valid: bool
    measurements_complete: bool
    thresholds_pass: bool
    physical_evidence: bool
    qualification_bundle_verified: bool
    hardware_qualified: bool
    gates: tuple[QualificationGate, ...]
    evidence_sha256: str
    claim_boundary: Mapping[str, bool]

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "schema_valid": self.schema_valid,
            "measurements_complete": self.measurements_complete,
            "thresholds_pass": self.thresholds_pass,
            "physical_evidence": self.physical_evidence,
            "qualification_bundle_verified": self.qualification_bundle_verified,
            "hardware_qualified": self.hardware_qualified,
            "gates": [
                {
                    "name": gate.name,
                    "status": gate.status.value,
                    "observed": gate.observed,
                    "requirement": gate.requirement,
                }
                for gate in self.gates
            ],
            "evidence_sha256": self.evidence_sha256,
            "claim_boundary": dict(self.claim_boundary),
        }


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be boolean")
    return value


def _normalize_sha256(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("qualification_bundle_sha256 must be a string")
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise ValueError("qualification_bundle_sha256 must be a 64-character SHA-256 digest")
    return normalized


def _sample_rate_error(signal: SignalGeometry) -> float:
    return abs(signal.measured_sample_rate_hz - signal.nominal_sample_rate_hz) / signal.nominal_sample_rate_hz


def _canonical_payload(manifest: HardwareQualificationManifest) -> dict[str, Any]:
    payload = asdict(manifest)
    payload["measurement_origin"] = manifest.measurement_origin.value
    payload["notes"] = dict(manifest.notes)
    return payload


def evidence_sha256(manifest: HardwareQualificationManifest) -> str:
    raw = json.dumps(
        _canonical_payload(manifest),
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def evaluate_hardware_qualification(
    manifest: HardwareQualificationManifest,
    thresholds: HardwareQualificationThresholds | None = None,
    *,
    verified_qualification_bundle_sha256: str | None = None,
) -> HardwareQualificationResult:
    """Evaluate named hardware evidence without inventing measurements.

    Passing numerical thresholds is deliberately insufficient for promotion.
    ``hardware_qualified`` additionally requires non-synthetic physical evidence
    and an independently verified neurOS qualification-bundle root that exactly
    matches the root named by the hardware manifest.
    """

    thresholds = thresholds or HardwareQualificationThresholds()
    manifest.validate()
    thresholds.validate()

    manifest_root = _normalize_sha256(manifest.qualification_bundle_sha256)
    verified_root = (
        _normalize_sha256(verified_qualification_bundle_sha256)
        if verified_qualification_bundle_sha256 is not None
        else None
    )
    qualification_bundle_verified = verified_root == manifest_root
    sample_rate_error = _sample_rate_error(manifest.signal)
    physical_evidence = (
        manifest.physical_run
        and not manifest.synthetic_contract_test
        and manifest.measurement_origin in {
            MeasurementOrigin.PHYSICAL,
            MeasurementOrigin.IMPORTED,
        }
    )

    def threshold_gate(name: str, observed: float, maximum: float) -> QualificationGate:
        return QualificationGate(
            name=name,
            status=GateStatus.PASS if observed <= maximum else GateStatus.FAIL,
            observed=observed,
            requirement=f"<= {maximum}",
        )

    gates: list[QualificationGate] = [
        QualificationGate(
            name="physical_measurement",
            status=GateStatus.PASS if physical_evidence else GateStatus.FAIL,
            observed={
                "measurement_origin": manifest.measurement_origin.value,
                "physical_run": manifest.physical_run,
                "synthetic_contract_test": manifest.synthetic_contract_test,
            },
            requirement=(
                "physical/imported measured evidence, physical_run=true, "
                "synthetic_contract_test=false"
            ),
        ),
        QualificationGate(
            name="verified_runtime_bundle",
            status=(
                GateStatus.PASS
                if qualification_bundle_verified
                else GateStatus.NOT_TESTED
                if verified_root is None
                else GateStatus.FAIL
            ),
            observed={
                "manifest_root": manifest_root,
                "verified_root": verified_root,
            },
            requirement="verified neurOS qualification bundle root exactly matches manifest root",
        ),
        QualificationGate(
            name="minimum_duration",
            status=(
                GateStatus.PASS
                if manifest.reliability.duration_s >= thresholds.min_duration_s
                else GateStatus.FAIL
            ),
            observed=manifest.reliability.duration_s,
            requirement=f">= {thresholds.min_duration_s} s",
        ),
        threshold_gate(
            "sample_loss_fraction",
            manifest.reliability.sample_loss_fraction,
            thresholds.max_sample_loss_fraction,
        ),
        threshold_gate(
            "queue_drop_fraction",
            manifest.reliability.queue_drop_fraction,
            thresholds.max_queue_drop_fraction,
        ),
        threshold_gate(
            "sample_rate_error_fraction",
            _sample_rate_error(manifest.signal),
            thresholds.max_sample_rate_error_fraction,
        ),
        threshold_gate(
            "absolute_clock_drift_ppm",
            abs(manifest.timing.drift_ppm),
            thresholds.max_abs_clock_drift_ppm,
        ),
        threshold_gate(
            "clock_uncertainty_p95_ms",
            manifest.timing.uncertainty_p95_ms,
            thresholds.max_clock_uncertainty_p95_ms,
        ),
        threshold_gate(
            "source_to_decision_p95_ms",
            manifest.latency.source_to_decision_p95_ms,
            thresholds.max_source_to_decision_p95_ms,
        ),
        threshold_gate(
            "source_to_decision_p99_ms",
            manifest.latency.source_to_decision_p99_ms,
            thresholds.max_source_to_decision_p99_ms,
        ),
    ]
    if thresholds.require_reconnect_test:
        reconnect_pass = (
            manifest.reliability.reconnect_tested
            and manifest.reliability.reconnect_attempts > 0
            and manifest.reliability.reconnect_successes == manifest.reliability.reconnect_attempts
        )
        gates.append(
            QualificationGate(
                name="reconnect_recovery",
                status=GateStatus.PASS if reconnect_pass else GateStatus.FAIL,
                observed={
                    "tested": manifest.reliability.reconnect_tested,
                    "attempts": manifest.reliability.reconnect_attempts,
                    "successes": manifest.reliability.reconnect_successes,
                },
                requirement="tested and every reconnect attempt succeeds",
            )
        )

    prerequisite_names = {"physical_measurement", "verified_runtime_bundle"}
    threshold_gates = [gate for gate in gates if gate.name not in prerequisite_names]
    thresholds_pass = all(gate.status is GateStatus.PASS for gate in threshold_gates)
    hardware_qualified = physical_evidence and qualification_bundle_verified and thresholds_pass

    return HardwareQualificationResult(
        evidence_id=manifest.evidence_id,
        schema_valid=True,
        measurements_complete=True,
        thresholds_pass=thresholds_pass,
        physical_evidence=physical_evidence,
        qualification_bundle_verified=qualification_bundle_verified,
        hardware_qualified=hardware_qualified,
        gates=tuple(gates),
        evidence_sha256=evidence_sha256(manifest),
        claim_boundary={
            "runtime_record_replay_qualified": qualification_bundle_verified,
            "real_dataset_qualified": False,
            "hardware_qualified": hardware_qualified,
            "closed_loop_qualified": False,
            "clinical_qualified": False,
        },
    )


def evaluate_hardware_evidence_bundle(
    manifest: HardwareQualificationManifest,
    qualification_bundle: str | Path,
    thresholds: HardwareQualificationThresholds | None = None,
) -> HardwareQualificationResult:
    """Verify the referenced runtime bundle before evaluating hardware promotion."""

    from neuros.qualification import verify_qualification_bundle

    verification = verify_qualification_bundle(
        qualification_bundle,
        expected_sha256=manifest.qualification_bundle_sha256,
    )
    return evaluate_hardware_qualification(
        manifest,
        thresholds,
        verified_qualification_bundle_sha256=str(verification["bundle_sha256"]),
    )


def _tuple_strings(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be an array")
    result = tuple(str(item) for item in value)
    if any(not item.strip() for item in result):
        raise ValueError(f"{field_name} must not contain empty values")
    return result


def manifest_from_mapping(raw: Mapping[str, Any]) -> HardwareQualificationManifest:
    """Parse and validate one JSON-like hardware evidence mapping."""

    try:
        device_raw = raw["device"]
        signal_raw = raw["signal"]
        timing_raw = raw["timing"]
        reliability_raw = raw["reliability"]
        latency_raw = raw["latency"]
    except KeyError as exc:
        raise ValueError(f"hardware evidence missing required section: {exc.args[0]}") from exc
    for name, section in (
        ("device", device_raw),
        ("signal", signal_raw),
        ("timing", timing_raw),
        ("reliability", reliability_raw),
        ("latency", latency_raw),
    ):
        if not isinstance(section, Mapping):
            raise ValueError(f"hardware evidence section {name} must be an object")

    device_values = dict(device_raw)
    for field_name in (
        "manufacturer",
        "device",
        "board_id",
        "firmware_version",
        "acquisition_library",
        "acquisition_library_version",
        "operating_system",
        "transport",
    ):
        if field_name not in device_values:
            raise ValueError(f"hardware evidence missing device.{field_name}")
        if not isinstance(device_values[field_name], str):
            raise ValueError(f"device.{field_name} must be a string")

    manifest = HardwareQualificationManifest(
        schema_version=int(raw.get("schema_version", HARDWARE_QUALIFICATION_SCHEMA_VERSION)),
        evidence_id=str(raw["evidence_id"]),
        measurement_origin=MeasurementOrigin(str(raw["measurement_origin"])),
        qualification_bundle_sha256=str(raw["qualification_bundle_sha256"]),
        physical_run=_strict_bool(raw["physical_run"], field_name="physical_run"),
        synthetic_contract_test=_strict_bool(
            raw.get("synthetic_contract_test", False),
            field_name="synthetic_contract_test",
        ),
        device=DeviceIdentity(**device_values),
        signal=SignalGeometry(
            channel_names=_tuple_strings(
                signal_raw["channel_names"], field_name="signal.channel_names"
            ),
            channel_types=_tuple_strings(
                signal_raw["channel_types"], field_name="signal.channel_types"
            ),
            units=_tuple_strings(signal_raw["units"], field_name="signal.units"),
            nominal_sample_rate_hz=signal_raw["nominal_sample_rate_hz"],
            measured_sample_rate_hz=signal_raw["measured_sample_rate_hz"],
        ),
        timing=TimingEvidence(**dict(timing_raw)),
        reliability=ReliabilityEvidence(**dict(reliability_raw)),
        latency=LatencyEvidence(**dict(latency_raw)),
        notes=dict(raw.get("notes", {})),
    )
    manifest.validate()
    return manifest


def thresholds_from_mapping(raw: Mapping[str, Any]) -> HardwareQualificationThresholds:
    thresholds = HardwareQualificationThresholds(**dict(raw))
    thresholds.validate()
    return thresholds


def load_hardware_evidence(path: str | Path) -> HardwareQualificationManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("hardware evidence root must be a JSON object")
    return manifest_from_mapping(raw)


def load_thresholds(path: str | Path) -> HardwareQualificationThresholds:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("hardware threshold root must be a JSON object")
    return thresholds_from_mapping(raw)
