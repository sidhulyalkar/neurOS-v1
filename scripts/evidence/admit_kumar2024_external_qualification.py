#!/usr/bin/env python3
"""Score-blind admission for one verified Kumar2024 external systems artifact.

This layer consumes only the narrow return value of the independent external
qualification verifier. It never opens a worker result file and never receives a
scientific metric. Its sole positive claim is that one exact sealed external
transport artifact passed the frozen structural verifier.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ADMISSION_SCHEMA = "neuros.nsq_kumar2024_external_systems_admission.v1"
VERIFIER_PATH = Path(__file__).resolve().with_name(
    "verify_kumar2024_external_qualification.py"
)

_SPEC = importlib.util.spec_from_file_location(
    "neuros_kumar2024_external_qualification_verifier_for_admission",
    VERIFIER_PATH,
)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - repository corruption
    raise RuntimeError("cannot load external qualification verifier")
verifier = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = verifier
_SPEC.loader.exec_module(verifier)

_VERIFIED_KEYS = frozenset(
    {
        "verified",
        "transport_provider",
        "binding_input_mode",
        "transport_source_revision",
        "transport_script_sha256",
        "external_bundle_sha256",
        "qualification_sha256",
        "worker_bundle_sha256",
        "shard_result_sha256",
        "numerical_result_interpretable",
        "global_analysis_performed",
        "external_floor_claim_generated",
        "orion_comparison_permitted",
    }
)
_FALSE_CLAIM_KEYS = (
    "numerical_result_interpretable",
    "global_analysis_performed",
    "external_floor_claim_generated",
    "orion_comparison_permitted",
)


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _identity(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        _canonical({"schema": ADMISSION_SCHEMA, "payload": payload})
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hex(name: str, value: Any, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-character lowercase hexadecimal string")
    if value != value.lower() or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


def _canonical_string(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{name} must be a non-empty canonical string")
    return value


def _write_once(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", closefd=True) as stream:
            fd = -1
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        if fd >= 0:
            os.close(fd)
    return path


def build_admission(
    verified: Mapping[str, Any],
    *,
    verifier_script_sha256: str,
) -> dict[str, Any]:
    """Bind the exact independent-verifier output into a systems-only admission."""
    if set(verified) != _VERIFIED_KEYS:
        raise ValueError(
            "independent verifier output contract drifted: "
            f"missing={sorted(_VERIFIED_KEYS - set(verified))}, "
            f"extra={sorted(set(verified) - _VERIFIED_KEYS)}"
        )
    if verified.get("verified") is not True:
        raise ValueError("external qualification is not independently verified")
    for key in _FALSE_CLAIM_KEYS:
        if verified.get(key) is not False:
            raise ValueError(f"external qualification requires {key}=false")

    provider = _canonical_string("transport_provider", verified["transport_provider"])
    if provider not in verifier.ALLOWED_PROVIDERS:
        raise ValueError("transport provider is outside the frozen verifier contract")
    input_mode = _canonical_string("binding_input_mode", verified["binding_input_mode"])
    if input_mode not in verifier.ALLOWED_BINDING_INPUT_MODES:
        raise ValueError("binding input mode is outside the frozen verifier contract")

    transport_revision = _hex(
        "transport_source_revision", verified["transport_source_revision"], 40
    )
    digest_fields = {
        key: _hex(key, verified[key], 64)
        for key in (
            "transport_script_sha256",
            "external_bundle_sha256",
            "qualification_sha256",
            "worker_bundle_sha256",
            "shard_result_sha256",
        )
    }
    verifier_sha = _hex("verifier_script_sha256", verifier_script_sha256, 64)

    payload = {
        "schema_version": 1,
        "artifact_kind": "verified_external_classical_worker_systems_admission",
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
        "transport_provider": provider,
        "binding_input_mode": input_mode,
        "transport_source_revision": transport_revision,
        **digest_fields,
        "verifier_script_sha256": verifier_sha,
        "transport_structurally_qualified": True,
        "admission_basis": [
            "independent_outer_bundle_verification",
            "independent_worker_bundle_verification",
            "frozen_source_and_environment_identity",
            "frozen_execution_and_shard_identity",
            "score_blind_claim_boundary",
        ],
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "production_fleet_authorized": False,
        "orion_comparison_permitted": False,
    }
    return {**payload, "admission_sha256": _identity(payload)}


def verify_admission(
    path: str | Path,
    *,
    require_current_verifier: bool = True,
) -> dict[str, Any]:
    admission_path = Path(path).expanduser().resolve()
    payload = json.loads(admission_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("admission must contain a JSON object")
    declared = payload.pop("admission_sha256", None)
    if not isinstance(declared, str):
        raise ValueError("admission is missing admission_sha256")
    expected = _identity(payload)
    if declared != expected:
        raise ValueError("admission identity mismatch")

    fixed = {
        "schema_version": 1,
        "artifact_kind": "verified_external_classical_worker_systems_admission",
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
        "transport_structurally_qualified": True,
        "scientific_outcomes_inspected": False,
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "production_fleet_authorized": False,
        "orion_comparison_permitted": False,
    }
    for key, value in fixed.items():
        if payload.get(key) != value:
            raise ValueError(f"admission {key} mismatch")
    if payload.get("transport_provider") not in verifier.ALLOWED_PROVIDERS:
        raise ValueError("admission transport provider is not frozen")
    if payload.get("binding_input_mode") not in verifier.ALLOWED_BINDING_INPUT_MODES:
        raise ValueError("admission binding input mode is not frozen")
    _hex("transport_source_revision", payload.get("transport_source_revision"), 40)
    for key in (
        "transport_script_sha256",
        "external_bundle_sha256",
        "qualification_sha256",
        "worker_bundle_sha256",
        "shard_result_sha256",
        "verifier_script_sha256",
    ):
        _hex(key, payload.get(key), 64)
    if require_current_verifier and payload["verifier_script_sha256"] != _file_sha256(VERIFIER_PATH):
        raise ValueError("admission was produced by a different verifier byte identity")
    payload["admission_sha256"] = declared
    return payload


def admit(
    qualification_root: str | Path,
    *,
    transport_script: str | Path,
    expected_transport_revision: str,
    output: str | Path,
) -> dict[str, Any]:
    transport_revision = _hex(
        "expected_transport_revision", expected_transport_revision, 40
    )
    verified = verifier.verify(
        qualification_root,
        transport_script=transport_script,
        expected_transport_revision=transport_revision,
    )
    receipt = build_admission(
        verified,
        verifier_script_sha256=_file_sha256(VERIFIER_PATH),
    )
    _write_once(Path(output).expanduser().resolve(), receipt)
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Admit or verify one score-blind Kumar2024 external systems receipt."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    create = subparsers.add_parser("admit")
    create.add_argument("qualification_root")
    create.add_argument("--transport-script", required=True)
    create.add_argument("--expected-transport-revision", required=True)
    create.add_argument("--output", required=True)

    verify = subparsers.add_parser("verify-admission")
    verify.add_argument("admission")
    verify.add_argument("--allow-historical-verifier", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "admit":
        result = admit(
            args.qualification_root,
            transport_script=args.transport_script,
            expected_transport_revision=args.expected_transport_revision,
            output=args.output,
        )
    else:
        result = verify_admission(
            args.admission,
            require_current_verifier=not args.allow_historical_verifier,
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
