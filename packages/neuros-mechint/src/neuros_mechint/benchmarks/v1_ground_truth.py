"""Known-answer gate for v1 schema, provenance, reproduction, and claim-boundary contracts."""

from __future__ import annotations

from dataclasses import dataclass

from neuros_mechint.core.manifest import stable_hash
from neuros_mechint.core.schema import (
    get_artifact_schema,
    migrate_artifact_envelope,
    migrate_manifest_payload,
    schema_catalog,
)
from neuros_mechint.release import default_v1_evidence_status

from .reproduction import (
    ReproductionMetricTolerance,
    ReproductionSnapshot,
    ReproductionSpec,
    assess_independent_reproduction,
)


@dataclass(frozen=True, slots=True)
class V1GroundTruthReport:
    manifest_migration_passed: bool
    legacy_artifact_migration_passed: bool
    independent_reproduction_passed: bool
    duplicate_run_rejected: bool
    decision_flip_rejected: bool
    schema_catalog_complete: bool
    software_contract_ready: bool
    empirical_overclaim_rejected: bool
    passed: bool

    def to_dict(self) -> dict[str, bool]:
        return {
            "manifest_migration_passed": self.manifest_migration_passed,
            "legacy_artifact_migration_passed": self.legacy_artifact_migration_passed,
            "independent_reproduction_passed": self.independent_reproduction_passed,
            "duplicate_run_rejected": self.duplicate_run_rejected,
            "decision_flip_rejected": self.decision_flip_rejected,
            "schema_catalog_complete": self.schema_catalog_complete,
            "software_contract_ready": self.software_contract_ready,
            "empirical_overclaim_rejected": self.empirical_overclaim_rejected,
            "passed": self.passed,
        }


def run_v1_release_contract_benchmark() -> V1GroundTruthReport:
    """Exercise v1 failure modes with answers known before analysis."""

    legacy_manifest = {
        "schema_version": "2",
        "experiment_name": "known-v1-migration",
        "method": "unit",
        "method_version": "1",
        "model_id": "model",
        "model_revision": "rev-a",
        "model_hash": "model-hash",
        "dataset_id": "dataset",
        "dataset_hash": "dataset-hash",
        "parameters": {"window_ms": 25},
        "seed": 7,
        "evidence_tier": {"level": 2, "label": "contract"},
        "benchmark": {"created_at": "historical-runtime-field"},
    }
    migrated_manifest = migrate_manifest_payload(legacy_manifest)
    manifest_migration_passed = (
        migrated_manifest["schema_version"] == "3"
        and len(migrated_manifest["scientific_fingerprint"]) == 64
        and "created_at" not in migrated_manifest["scientific_identity"]
    )

    evidence_schema = get_artifact_schema("evidence_pack")
    legacy_result = {
        "schema_version": evidence_schema.result_schema,
        "study_fingerprint": "known-study",
    }
    legacy_envelope = {
        "artifact_schema": evidence_schema.artifact_schema,
        "artifact_hash": stable_hash(legacy_result),
        "result": legacy_result,
    }
    migrated_envelope = migrate_artifact_envelope(legacy_envelope, family="evidence_pack")
    legacy_artifact_migration_passed = (
        migrated_envelope["artifact_hash"] == legacy_envelope["artifact_hash"]
        and migrated_envelope["result"] == legacy_result
        and migrated_envelope["contract"]["family"] == "evidence_pack"
    )

    reproduction_spec = ReproductionSpec(
        reproduction_id="known-independent-rerun",
        artifact_family="correspondence",
        required_decision="promoted",
        metric_tolerances=(
            ReproductionMetricTolerance("causal_recovery", absolute=0.02, relative=0.0),
        ),
    )
    reference = ReproductionSnapshot(
        artifact_family="correspondence",
        study_fingerprint="same-scientific-study",
        run_hash="run-a",
        execution_id="worker-a",
        decision="promoted",
        metrics={"causal_recovery": 0.80},
    )
    independent = ReproductionSnapshot(
        artifact_family="correspondence",
        study_fingerprint="same-scientific-study",
        run_hash="run-b",
        execution_id="worker-b",
        decision="promoted",
        metrics={"causal_recovery": 0.81},
    )
    duplicate = ReproductionSnapshot(
        artifact_family="correspondence",
        study_fingerprint="same-scientific-study",
        run_hash="run-a",
        execution_id="worker-a",
        decision="promoted",
        metrics={"causal_recovery": 0.80},
    )
    flipped = ReproductionSnapshot(
        artifact_family="correspondence",
        study_fingerprint="same-scientific-study",
        run_hash="run-c",
        execution_id="worker-c",
        decision="rejected",
        metrics={"causal_recovery": 0.81},
    )
    independent_reproduction_passed = assess_independent_reproduction(
        reproduction_spec, reference, independent
    ).passed
    duplicate_run_rejected = not assess_independent_reproduction(
        reproduction_spec, reference, duplicate
    ).passed
    decision_flip_rejected = not assess_independent_reproduction(
        reproduction_spec, reference, flipped
    ).passed

    catalog_families = {item["family"] for item in schema_catalog()}
    schema_catalog_complete = catalog_families == {
        "evidence_pack",
        "factorial",
        "correspondence",
        "replication",
        "dose_response",
    }
    status = default_v1_evidence_status()
    software_contract_ready = status.software_contract_ready
    empirical_overclaim_rejected = (
        not status.empirical_evidence_complete and bool(status.pending_empirical_requirements)
    )

    checks = (
        manifest_migration_passed,
        legacy_artifact_migration_passed,
        independent_reproduction_passed,
        duplicate_run_rejected,
        decision_flip_rejected,
        schema_catalog_complete,
        software_contract_ready,
        empirical_overclaim_rejected,
    )
    return V1GroundTruthReport(
        manifest_migration_passed=manifest_migration_passed,
        legacy_artifact_migration_passed=legacy_artifact_migration_passed,
        independent_reproduction_passed=independent_reproduction_passed,
        duplicate_run_rejected=duplicate_run_rejected,
        decision_flip_rejected=decision_flip_rejected,
        schema_catalog_complete=schema_catalog_complete,
        software_contract_ready=software_contract_ready,
        empirical_overclaim_rejected=empirical_overclaim_rejected,
        passed=all(checks),
    )
