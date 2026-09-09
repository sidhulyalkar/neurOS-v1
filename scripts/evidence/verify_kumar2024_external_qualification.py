#!/usr/bin/env python3
"""Independent verifier for the frozen Kumar2024 external systems qualification.

This verifier intentionally imports no neurOS package. It audits only serialized
bytes and frozen authority constants, so the qualification directory can be
checked on a different machine without rerunning preprocessing or a model.

It never prints numerical scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

SOURCE_REVISION = "56fd0c5132bec17575d68f62256cb80fd5661395"
BINDING_RUN_ID = 33291842755
BINDING_ARTIFACT_ID = 9726471429
BINDING_ARTIFACT_SHA256 = "107a9fc57fc913815131cdf165bc35d3a1130c8300828f1f97672b27441ef0f6"
BINDING_BUNDLE_SHA256 = "45679a95c614e2107f64d7cb9ce1f87f10179c617ff160bccfd899b7ff8688d3"
ENVIRONMENT_AUTHORITY_SHA256 = "c45e15561ab95b8a4be0734f2fecd993fca53bf24a6e38b3c8739e1424cd1cb9"
RAW_MATERIALIZATION_SHA256 = "60b89be5ded4b1ca559260b781dfcce781cf7473ad17e92cec172671e6c70a5b"
STUDY_MATERIALIZATION_SHA256 = "28bd5564ebe87ca396b2a6093094c53b879b3b461c9c413b1422fab92d9da43a"
EXECUTION_PLAN_SHA256 = "987bb3b5566d1d481141d9a549f3588994d34e25d05cc9536baccaaa4a4641ac"
SHARD_SPEC_SHA256 = "b6943a6bd0692fb99c14d3b57b2eea04ea8bf16b79b92a18415912f2b8381ceb"
SHARD_ID = "subject-01/session-1/split-2026/mne-csp-lda/deterministic"
METHOD_REALIZATION_KEY = "mne-csp-lda/deterministic"
BUDGETS = (0, 1, 2, 5, 10)
ALLOWED_PROVIDERS = frozenset(
    {
        "lightning",
        "nvidia-brev",
        "nvidia-cloud-tasks",
        "nvidia-lepton",
        "local-wsl",
        "local-linux",
    }
)
ALLOWED_BINDING_INPUT_MODES = frozenset({"github_artifact", "verified_archive"})
WORKER_BUNDLE_FILES = frozenset(
    {
        "worker_manifest.json",
        "case_result.json",
        "observation_roles.json",
        "shard_result.json",
    }
)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _identity(schema: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"schema": schema, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        raise ValueError(f"{name} mismatch: expected={expected!r}, observed={observed!r}")


def _require_hex(name: str, value: Any, length: int) -> str:
    if not isinstance(value, str) or len(value) != length:
        raise ValueError(f"{name} must be a {length}-character lowercase hexadecimal string")
    if any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} must be lowercase hexadecimal")
    return value


def _verify_recursive_outer_bundle(root: Path) -> tuple[str, dict[str, str]]:
    payload = _load_json(root / "external_artifact_hashes.json")
    _require_equal("external hash schema_version", payload.get("schema_version"), 1)
    declared = payload.get("files")
    if not isinstance(declared, dict) or not declared:
        raise ValueError("external artifact hash manifest requires a non-empty files mapping")

    actual: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "external_artifact_hashes.json":
            continue
        relative = path.relative_to(root).as_posix()
        actual[relative] = _sha256(path)
    if declared != actual:
        missing = sorted(set(declared) - set(actual))
        extra = sorted(set(actual) - set(declared))
        changed = sorted(
            key for key in set(actual) & set(declared) if actual[key] != declared[key]
        )
        raise ValueError(
            "external artifact file authority mismatch: "
            f"missing={missing}, extra={extra}, changed={changed}"
        )
    root_sha = _identity(
        "neuros.promoted_classical_worker_external_bundle.v1",
        {"files": actual},
    )
    _require_equal("external bundle SHA", payload.get("bundle_sha256"), root_sha)
    return root_sha, actual


def _verify_qualification_manifest(
    root: Path,
    *,
    transport_script: Path | None = None,
    expected_transport_revision: str | None = None,
) -> dict[str, Any]:
    payload = _load_json(root / "qualification_manifest.json")
    declared_sha = payload.pop("qualification_sha256", None)
    if not isinstance(declared_sha, str):
        raise ValueError("qualification manifest is missing qualification_sha256")
    recomputed = _identity(
        "neuros.promoted_classical_worker_external_systems_qualification.v1",
        payload,
    )
    _require_equal("qualification SHA", declared_sha, recomputed)

    expected = {
        "schema_version": 1,
        "artifact_kind": "promoted_classical_worker_external_systems_qualification",
        "source_revision": SOURCE_REVISION,
        "binding_run_id": BINDING_RUN_ID,
        "binding_artifact_id": BINDING_ARTIFACT_ID,
        "binding_artifact_sha256": BINDING_ARTIFACT_SHA256,
        "binding_bundle_sha256": BINDING_BUNDLE_SHA256,
        "environment_authority_sha256": ENVIRONMENT_AUTHORITY_SHA256,
        "raw_materialization_sha256": RAW_MATERIALIZATION_SHA256,
        "study_materialization_sha256": STUDY_MATERIALIZATION_SHA256,
        "execution_plan_sha256": EXECUTION_PLAN_SHA256,
        "shard_spec_sha256": SHARD_SPEC_SHA256,
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
    }
    for key, value in expected.items():
        _require_equal(f"qualification {key}", payload.get(key), value)

    provider = payload.get("transport_provider")
    if provider not in ALLOWED_PROVIDERS:
        raise ValueError(f"qualification transport_provider is not frozen: {provider!r}")
    input_mode = payload.get("binding_input_mode")
    if input_mode not in ALLOWED_BINDING_INPUT_MODES:
        raise ValueError(f"qualification binding_input_mode is not frozen: {input_mode!r}")
    transport_revision = _require_hex(
        "qualification transport_source_revision",
        payload.get("transport_source_revision"),
        40,
    )
    transport_script_sha = _require_hex(
        "qualification transport_script_sha256",
        payload.get("transport_script_sha256"),
        64,
    )
    if expected_transport_revision is not None:
        _require_equal(
            "qualification transport_source_revision",
            transport_revision,
            _require_hex("expected transport revision", expected_transport_revision, 40),
        )
    if transport_script is not None:
        script_path = transport_script.expanduser().resolve()
        if not script_path.is_file():
            raise ValueError("--transport-script must name an existing file")
        _require_equal(
            "qualification transport_script_sha256",
            transport_script_sha,
            _sha256(script_path),
        )

    platform = payload.get("platform")
    if not isinstance(platform, dict):
        raise ValueError("qualification platform must be a mapping")
    _require_equal("qualification platform.system", platform.get("system"), "Linux")
    _require_equal("qualification platform.machine", platform.get("machine"), "x86_64")
    _require_hex("qualification worker_bundle_sha256", payload.get("worker_bundle_sha256"), 64)
    _require_hex(
        "qualification worker_shard_result_sha256",
        payload.get("worker_shard_result_sha256"),
        64,
    )
    payload["qualification_sha256"] = declared_sha
    return payload


def _verify_worker_bundle(root: Path, qualification: Mapping[str, Any]) -> dict[str, Any]:
    worker = root / "worker"
    hashes = _load_json(worker / "artifact_hashes.json")
    _require_equal("worker hash schema_version", hashes.get("schema_version"), 1)
    declared = hashes.get("files")
    if not isinstance(declared, dict):
        raise ValueError("worker artifact_hashes files must be a mapping")
    _require_equal("worker managed file set", frozenset(declared), WORKER_BUNDLE_FILES)
    actual = {name: _sha256(worker / name) for name in sorted(WORKER_BUNDLE_FILES)}
    _require_equal("worker file hashes", declared, actual)
    root_sha = _identity(
        "neuros.nsq_kumar2024_promoted_worker_bundle.v1",
        {"files": actual},
    )
    _require_equal("worker bundle SHA", hashes.get("bundle_sha256"), root_sha)
    _require_equal("qualification worker bundle SHA", qualification["worker_bundle_sha256"], root_sha)

    manifest = _load_json(worker / "worker_manifest.json")
    expected_manifest = {
        "schema_version": 1,
        "artifact_kind": "promoted_atomic_worker_result",
        "binding_bundle_sha256": BINDING_BUNDLE_SHA256,
        "execution_plan_sha256": EXECUTION_PLAN_SHA256,
        "shard_spec_sha256": SHARD_SPEC_SHA256,
        "shard_id": SHARD_ID,
        "source_revision": SOURCE_REVISION,
        "environment_authority_sha256": ENVIRONMENT_AUTHORITY_SHA256,
        "study_materialization_sha256": STUDY_MATERIALIZATION_SHA256,
        "raw_materialization_sha256": RAW_MATERIALIZATION_SHA256,
        "method_realization_key": METHOD_REALIZATION_KEY,
        "budgets_per_class": list(BUDGETS),
        "global_analysis_performed": False,
    }
    for key, value in expected_manifest.items():
        _require_equal(f"worker manifest {key}", manifest.get(key), value)

    shard_result = _load_json(worker / "shard_result.json")
    declared_shard_sha = shard_result.pop("shard_result_sha256", None)
    if not isinstance(declared_shard_sha, str):
        raise ValueError("shard_result.json is missing shard_result_sha256")
    recomputed_shard_sha = _identity(
        "neuros.kumar2024_promoted_shard_result.v1",
        shard_result,
    )
    _require_equal("promoted shard-result SHA", declared_shard_sha, recomputed_shard_sha)
    _require_equal("worker manifest shard-result SHA", manifest.get("shard_result_sha256"), recomputed_shard_sha)
    _require_equal(
        "qualification shard-result SHA",
        qualification["worker_shard_result_sha256"],
        recomputed_shard_sha,
    )

    expected_shard = {
        "schema_version": 1,
        "execution_plan_sha256": EXECUTION_PLAN_SHA256,
        "shard_spec_sha256": SHARD_SPEC_SHA256,
        "study_materialization_sha256": STUDY_MATERIALIZATION_SHA256,
        "environment_authority_sha256": ENVIRONMENT_AUTHORITY_SHA256,
        "raw_materialization_sha256": RAW_MATERIALIZATION_SHA256,
    }
    for key, value in expected_shard.items():
        _require_equal(f"shard result {key}", shard_result.get(key), value)
    rows = shard_result.get("rows")
    if not isinstance(rows, list) or len(rows) != len(BUDGETS):
        raise ValueError("shard result must contain exactly five budget rows")
    observed_budgets = tuple(sorted(int(row.get("calibration_per_class", -1)) for row in rows))
    _require_equal("shard result budget frontier", observed_budgets, BUDGETS)
    for row in rows:
        _require_equal("shard row method_id", row.get("method_id"), "mne-csp-lda")
        _require_equal("shard row subject", row.get("subject"), 1)
        _require_equal("shard row held_out_session", str(row.get("held_out_session")), "1")
        _require_equal("shard row split_seed", row.get("split_seed"), 2026)
        _require_equal("shard row model_seed", row.get("model_seed"), None)
        _require_equal("shard row shard_spec_sha256", row.get("shard_spec_sha256"), SHARD_SPEC_SHA256)
        _require_equal(
            "shard row execution_plan_sha256",
            row.get("execution_plan_sha256"),
            EXECUTION_PLAN_SHA256,
        )
        _require_equal(
            "shard row binding_bundle_sha256",
            row.get("binding_bundle_sha256"),
            BINDING_BUNDLE_SHA256,
        )

    case = _load_json(worker / "case_result.json")
    _require_equal("case subject", case.get("subject"), 1)
    _require_equal("case held_out_session", str(case.get("held_out_session")), "1")
    _require_equal("case split_seed", case.get("split_seed"), 2026)
    _require_equal("case model_seed", case.get("model_seed"), None)
    _require_equal("case method_realization_key", case.get("method_realization_key"), METHOD_REALIZATION_KEY)
    result = case.get("result")
    if not isinstance(result, dict):
        raise ValueError("case_result result must be a mapping")
    _require_equal("case result SHA", result.get("result_sha256"), manifest.get("result_sha256"))

    roles = _load_json(worker / "observation_roles.json")
    _require_equal("observation roles shard SHA", roles.get("shard_spec_sha256"), SHARD_SPEC_SHA256)
    role_rows = roles.get("rows")
    if not isinstance(role_rows, list) or len(role_rows) != len(BUDGETS):
        raise ValueError("observation roles must contain exactly five budget rows")
    role_budgets = tuple(sorted(int(row.get("calibration_per_class", -1)) for row in role_rows))
    _require_equal("observation-role budget frontier", role_budgets, BUDGETS)

    return {
        "worker_bundle_sha256": root_sha,
        "shard_result_sha256": recomputed_shard_sha,
    }


def verify(
    root: str | Path,
    *,
    transport_script: str | Path | None = None,
    expected_transport_revision: str | None = None,
) -> dict[str, Any]:
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError("qualification root must be an existing directory")
    script = None if transport_script is None else Path(transport_script)
    external_sha, _ = _verify_recursive_outer_bundle(base)
    qualification = _verify_qualification_manifest(
        base,
        transport_script=script,
        expected_transport_revision=expected_transport_revision,
    )
    worker = _verify_worker_bundle(base, qualification)
    return {
        "verified": True,
        "transport_provider": qualification["transport_provider"],
        "binding_input_mode": qualification["binding_input_mode"],
        "transport_source_revision": qualification["transport_source_revision"],
        "transport_script_sha256": qualification["transport_script_sha256"],
        "external_bundle_sha256": external_sha,
        "qualification_sha256": qualification["qualification_sha256"],
        "worker_bundle_sha256": worker["worker_bundle_sha256"],
        "shard_result_sha256": worker["shard_result_sha256"],
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Independently verify one frozen Kumar2024 external qualification bundle."
    )
    parser.add_argument("qualification_root")
    parser.add_argument(
        "--transport-script",
        help="Optional runner script whose bytes must match transport_script_sha256.",
    )
    parser.add_argument(
        "--expected-transport-revision",
        help="Optional exact 40-character transport repository revision to require.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    print(
        json.dumps(
            verify(
                args.qualification_root,
                transport_script=args.transport_script,
                expected_transport_revision=args.expected_transport_revision,
            ),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
