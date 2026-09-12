from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER_PATH = ROOT / "scripts" / "evidence" / "verify_kumar2024_external_qualification.py"
SPEC = importlib.util.spec_from_file_location("kumar_external_verifier", VERIFIER_PATH)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(verifier)


def _identity(schema: str, payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(
            {"schema": schema, "payload": payload},
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_worker(worker: Path) -> str:
    files = {
        name: _sha(worker / name)
        for name in sorted(verifier.WORKER_BUNDLE_FILES)
    }
    root = _identity("neuros.nsq_kumar2024_promoted_worker_bundle.v1", {"files": files})
    _write(
        worker / "artifact_hashes.json",
        {"schema_version": 1, "files": files, "bundle_sha256": root},
    )
    return root


def _seal_qualification(root: Path) -> str:
    files = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "external_artifact_hashes.json":
            continue
        files[path.relative_to(root).as_posix()] = _sha(path)
    bundle = _identity(
        "neuros.promoted_classical_worker_external_bundle.v1",
        {"files": files},
    )
    _write(
        root / "external_artifact_hashes.json",
        {"schema_version": 1, "files": files, "bundle_sha256": bundle},
    )
    return bundle


def _rewrite_qualification(root: Path, **changes) -> None:
    payload = json.loads((root / "qualification_manifest.json").read_text(encoding="utf-8"))
    payload.pop("qualification_sha256")
    payload.update(changes)
    payload["qualification_sha256"] = _identity(
        "neuros.promoted_classical_worker_external_systems_qualification.v1",
        payload,
    )
    _write(root / "qualification_manifest.json", payload)
    _seal_qualification(root)


def _synthetic_bundle(tmp_path: Path) -> tuple[Path, Path, str]:
    root = tmp_path / "qualification"
    worker = root / "worker"
    worker.mkdir(parents=True)
    transport_script = tmp_path / "transport.sh"
    transport_script.write_bytes(b"#!/usr/bin/env bash\nexit 0\n")
    transport_script_sha = _sha(transport_script)
    transport_revision = "a" * 40

    case_sha = "1" * 64
    method_sha = "2" * 64
    processed_sha = "3" * 64
    result_sha = "4" * 64
    comparison_sha = "5" * 64
    dataset_sha = "6" * 64
    protocol_sha = "7" * 64
    preprocessing_sha = "8" * 64

    rows = []
    roles = []
    for budget in verifier.BUDGETS:
        row_sha = f"{budget + 16:064x}"
        rows.append(
            {
                "calibration_per_class": budget,
                "method_id": "mne-csp-lda",
                "subject": 1,
                "held_out_session": "1",
                "split_seed": 2026,
                "model_seed": None,
                "case_authority_sha256": case_sha,
                "method_realization_key": verifier.METHOD_REALIZATION_KEY,
                "qualification_result_row_sha256": row_sha,
                "shard_spec_sha256": verifier.SHARD_SPEC_SHA256,
                "execution_plan_sha256": verifier.EXECUTION_PLAN_SHA256,
                "binding_bundle_sha256": verifier.BINDING_BUNDLE_SHA256,
                "status": "ok",
                "balanced_accuracy": 0.5,
            }
        )
        roles.append(
            {
                "calibration_per_class": budget,
                "qualification_result_row_sha256": row_sha,
                "roles": {},
            }
        )

    shard_payload = {
        "schema_version": 1,
        "execution_plan_sha256": verifier.EXECUTION_PLAN_SHA256,
        "shard_spec_sha256": verifier.SHARD_SPEC_SHA256,
        "comparison_plan_sha256": comparison_sha,
        "study_materialization_sha256": verifier.STUDY_MATERIALIZATION_SHA256,
        "environment_authority_sha256": verifier.ENVIRONMENT_AUTHORITY_SHA256,
        "raw_materialization_sha256": verifier.RAW_MATERIALIZATION_SHA256,
        "dataset_lineage_sha256": dataset_sha,
        "protocol_sha256": protocol_sha,
        "preprocessing_authority_sha256": preprocessing_sha,
        "case_authority_sha256": case_sha,
        "method_spec_sha256": method_sha,
        "rows": rows,
    }
    shard_sha = _identity("neuros.kumar2024_promoted_shard_result.v1", shard_payload)
    _write(worker / "shard_result.json", {**shard_payload, "shard_result_sha256": shard_sha})

    manifest = {
        "schema_version": 1,
        "artifact_kind": "promoted_atomic_worker_result",
        "binding_bundle_sha256": verifier.BINDING_BUNDLE_SHA256,
        "execution_plan_sha256": verifier.EXECUTION_PLAN_SHA256,
        "shard_spec_sha256": verifier.SHARD_SPEC_SHA256,
        "shard_id": verifier.SHARD_ID,
        "source_revision": verifier.SOURCE_REVISION,
        "environment_authority_sha256": verifier.ENVIRONMENT_AUTHORITY_SHA256,
        "study_materialization_sha256": verifier.STUDY_MATERIALIZATION_SHA256,
        "raw_materialization_sha256": verifier.RAW_MATERIALIZATION_SHA256,
        "processed_shard_sha256": processed_sha,
        "case_authority_sha256": case_sha,
        "method_spec_sha256": method_sha,
        "method_realization_key": verifier.METHOD_REALIZATION_KEY,
        "budgets_per_class": list(verifier.BUDGETS),
        "result_sha256": result_sha,
        "shard_result_sha256": shard_sha,
        "global_analysis_performed": False,
        "claim_boundary": "synthetic qualification fixture",
    }
    _write(worker / "worker_manifest.json", manifest)
    _write(
        worker / "case_result.json",
        {
            "schema_version": 1,
            "subject": 1,
            "held_out_session": "1",
            "split_seed": 2026,
            "model_seed": None,
            "method_realization_key": verifier.METHOD_REALIZATION_KEY,
            "result": {"result_sha256": result_sha},
        },
    )
    _write(
        worker / "observation_roles.json",
        {
            "schema_version": 1,
            "shard_spec_sha256": verifier.SHARD_SPEC_SHA256,
            "rows": roles,
        },
    )
    worker_bundle = _seal_worker(worker)

    qualification = {
        "schema_version": 1,
        "artifact_kind": "promoted_classical_worker_external_systems_qualification",
        "transport_provider": "lightning",
        "binding_input_mode": "verified_archive",
        "transport_source_revision": transport_revision,
        "transport_script_sha256": transport_script_sha,
        "transport_semantics": "synthetic transport fixture",
        "source_revision": verifier.SOURCE_REVISION,
        "binding_run_id": verifier.BINDING_RUN_ID,
        "binding_artifact_id": verifier.BINDING_ARTIFACT_ID,
        "binding_artifact_sha256": verifier.BINDING_ARTIFACT_SHA256,
        "binding_bundle_sha256": verifier.BINDING_BUNDLE_SHA256,
        "environment_authority_sha256": verifier.ENVIRONMENT_AUTHORITY_SHA256,
        "raw_materialization_sha256": verifier.RAW_MATERIALIZATION_SHA256,
        "study_materialization_sha256": verifier.STUDY_MATERIALIZATION_SHA256,
        "execution_plan_sha256": verifier.EXECUTION_PLAN_SHA256,
        "shard_spec_sha256": verifier.SHARD_SPEC_SHA256,
        "worker_bundle_sha256": worker_bundle,
        "worker_shard_result_sha256": shard_sha,
        "platform": {"system": "Linux", "machine": "x86_64"},
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "claim_boundary": "synthetic qualification fixture",
    }
    qualification["qualification_sha256"] = _identity(
        "neuros.promoted_classical_worker_external_systems_qualification.v1",
        qualification,
    )
    _write(root / "qualification_manifest.json", qualification)
    _seal_qualification(root)
    return root, transport_script, transport_revision


def test_external_qualification_verifier_accepts_canonical_fixture(tmp_path: Path) -> None:
    root, script, revision = _synthetic_bundle(tmp_path)
    result = verifier.verify(
        root,
        transport_script=script,
        expected_transport_revision=revision,
    )
    assert result["verified"] is True
    assert result["numerical_result_interpretable"] is False
    assert result["external_floor_claim_generated"] is False
    assert result["orion_comparison_permitted"] is False


def test_resealed_interpretability_tamper_rejects(tmp_path: Path) -> None:
    root, _, _ = _synthetic_bundle(tmp_path)
    _rewrite_qualification(root, numerical_result_interpretable=True)
    with pytest.raises(ValueError, match="numerical_result_interpretable"):
        verifier.verify(root)


def test_resealed_shard_semantic_tamper_rejects(tmp_path: Path) -> None:
    root, _, _ = _synthetic_bundle(tmp_path)
    worker = root / "worker"
    shard = json.loads((worker / "shard_result.json").read_text(encoding="utf-8"))
    shard.pop("shard_result_sha256")
    shard["rows"][0]["method_id"] = "performance-selected-method"
    shard_sha = _identity("neuros.kumar2024_promoted_shard_result.v1", shard)
    _write(worker / "shard_result.json", {**shard, "shard_result_sha256": shard_sha})

    manifest = json.loads((worker / "worker_manifest.json").read_text(encoding="utf-8"))
    manifest["shard_result_sha256"] = shard_sha
    _write(worker / "worker_manifest.json", manifest)
    worker_bundle = _seal_worker(worker)
    _rewrite_qualification(
        root,
        worker_bundle_sha256=worker_bundle,
        worker_shard_result_sha256=shard_sha,
    )
    with pytest.raises(ValueError, match="method_id"):
        verifier.verify(root)


def test_unsealed_worker_byte_tamper_rejects_at_outer_root(tmp_path: Path) -> None:
    root, _, _ = _synthetic_bundle(tmp_path)
    target = root / "worker" / "case_result.json"
    target.write_bytes(target.read_bytes() + b" \n")
    with pytest.raises(ValueError, match="external artifact file authority mismatch"):
        verifier.verify(root)
