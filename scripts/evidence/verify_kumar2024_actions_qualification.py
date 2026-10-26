#!/usr/bin/env python3
"""Independently audit the frozen Kumar2024 GitHub Actions one-shard output.

This consumes *downloaded artifact bytes*. It imports only Python stdlib and
the separate stdlib-only external worker verifier, never neurOS or its worker.
No EEG scoring values are printed or scientifically interpreted.

The verification tier is structural and self-consistency only unless a digest
has been pinned through a genuinely separate trusted channel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

from verify_kumar2024_external_qualification import (
    BINDING_ARTIFACT_ID,
    BINDING_BUNDLE_SHA256,
    BINDING_RUN_ID,
    METHOD_REALIZATION_KEY,
    SHARD_ID,
    SHARD_SPEC_SHA256,
    SOURCE_REVISION,
    _identity,
    _load_json,
    _verify_worker_bundle,
)

DOMAIN_ROOT = "neuros.kumar2024_actions_receipt_files.v1"
DOMAIN_QUALIFICATION = "neuros.promoted_classical_worker_systems_qualification.v1"
FILES = frozenset(
    {
        "qualification_manifest.json",
        "worker_run.json",
        "worker/artifact_hashes.json",
        "worker/worker_manifest.json",
        "worker/case_result.json",
        "worker/observation_roles.json",
        "worker/shard_result.json",
    }
)
CLAIM_FLAGS = (
    "numerical_result_interpretable",
    "global_analysis_performed",
    "external_floor_claim_generated",
    "orion_comparison_permitted",
)


def _require_hex(name: str, value: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase 64-hex SHA-256")
    return value


def _require_equal(name: str, observed: Any, expected: Any) -> None:
    if observed != expected:
        # Never print the observed scientific payload, even on mismatch.
        raise ValueError(f"{name} does not match frozen authority")


def _audit_paths(root: Path) -> dict[str, str]:
    if not root.is_dir() or root.is_symlink():
        raise ValueError("qualification artifact must be a real directory")
    observed: set[str] = set()
    digests: dict[str, str] = {}
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            raise ValueError("qualification artifact contains a symlink")
        if path.is_dir():
            if relative != "worker":
                raise ValueError("qualification artifact has unexpected directory")
            continue
        if not path.is_file() or relative not in FILES:
            raise ValueError("qualification artifact has unexpected file or node")
        observed.add(relative)
        digests[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != FILES:
        raise ValueError("qualification artifact file inventory differs from v1")
    return dict(sorted(digests.items()))


def verify(
    root: str | Path,
    *,
    expected_files_root: str | None = None,
    expected_qualification_sha: str | None = None,
) -> dict[str, Any]:
    """Reconstruct both receipts without importing their producing worker."""
    root = Path(root).expanduser()
    hashes = _audit_paths(root)
    qualifying = _load_json(root / "qualification_manifest.json")
    declared_sha = _require_hex(
        "qualification_sha256", qualifying.get("qualification_sha256")
    )
    body = dict(qualifying)
    body.pop("qualification_sha256")
    _require_equal(
        "qualification content identity",
        declared_sha,
        _identity(DOMAIN_QUALIFICATION, body),
    )
    for key, expected in {
        "schema_version": 1,
        "artifact_kind": "promoted_classical_worker_systems_qualification",
        "source_revision": SOURCE_REVISION,
        "binding_run_id": BINDING_RUN_ID,
        "binding_artifact_id": BINDING_ARTIFACT_ID,
        "binding_bundle_sha256": BINDING_BUNDLE_SHA256,
        "shard_spec_sha256": SHARD_SPEC_SHA256,
        "shard_id": SHARD_ID,
        "method_id": "mne-csp-lda",
    }.items():
        _require_equal(f"qualification {key}", qualifying.get(key), expected)
    for flag in CLAIM_FLAGS:
        _require_equal(f"qualification {flag}", qualifying.get(flag), False)

    worker_bundle_sha = _require_hex(
        "worker_bundle_sha256", qualifying.get("worker_bundle_sha256")
    )
    worker_result_sha = _require_hex(
        "worker_shard_result_sha256",
        qualifying.get("worker_shard_result_sha256"),
    )
    validated_worker = _verify_worker_bundle(root, qualifying)
    _require_equal(
        "independently calculated worker bundle", validated_worker["worker_bundle_sha256"],
        worker_bundle_sha,
    )
    _require_equal(
        "independently calculated shard result", validated_worker["shard_result_sha256"],
        worker_result_sha,
    )
    # Worker source/method/budget authority is checked inside the independent
    # verifier, using archived constants. stdout is hashed, never interpreted.
    file_root = _identity(DOMAIN_ROOT, {"files": hashes})
    if expected_files_root is not None:
        _require_equal(
            "externally pinned downloaded-file root",
            file_root, _require_hex("expected_files_root", expected_files_root),
        )
    if expected_qualification_sha is not None:
        _require_equal(
            "externally pinned qualification identity",
            declared_sha,
            _require_hex("expected_qualification_sha", expected_qualification_sha),
        )
    return {
        "verified": True,
        "schema": "neuros.kumar2024_actions_artifact_structural_audit.v1",
        "source_revision": SOURCE_REVISION,
        "shard_spec_sha256": SHARD_SPEC_SHA256,
        "method_realization_key": METHOD_REALIZATION_KEY,
        "artifact_files_root_sha256": file_root,
        "qualification_sha256": declared_sha,
        "worker_bundle_sha256": worker_bundle_sha,
        "worker_shard_result_sha256": worker_result_sha,
        "externally_pinned": expected_files_root is not None,
        "publisher_authenticated": False,
        "numerical_result_interpretable": False,
        "global_analysis_performed": False,
        "external_floor_claim_generated": False,
        "orion_comparison_permitted": False,
        "scope": "serialized-structural-integrity; no model rerun, outcomes, or identity attestation",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("qualification_root", help="Extracted original Actions artifact directory")
    parser.add_argument("--expected-files-root", help="SHA-256 root independently captured earlier")
    parser.add_argument("--expected-qualification-sha", help="Receipt identity from trusted independent channel")
    args = parser.parse_args()
    try:
        report = verify(
            args.qualification_root,
            expected_files_root=args.expected_files_root,
            expected_qualification_sha=args.expected_qualification_sha,
        )
    except (ValueError, OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        print(
            json.dumps({"verified": False, "reason": str(exc)}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
