#!/usr/bin/env python3
"""Compare independent neurOS release builds for byte-for-byte reproducibility."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

BUILD_SCHEMA = "neuros.reproducible_release_build.v1"
REPORT_SCHEMA = "neuros.reproducible_release_comparison.v1"
CORE_TOOLCHAIN = (
    "pip",
    "build",
    "setuptools",
    "wheel",
    "packaging",
    "pyproject-hooks",
)


def _load_build(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "reproducible-build-manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"build manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if payload.get("schema") != BUILD_SCHEMA:
        raise RuntimeError(f"unexpected build manifest schema in {manifest_path}")
    payload["_root"] = str(root)
    return payload


def _artifact_map(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = payload.get("artifacts")
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"build has no artifacts: {payload.get('_root')}")
    result: dict[str, dict[str, Any]] = {}
    for item in rows:
        if not isinstance(item, dict):
            raise RuntimeError("artifact entry is not an object")
        name = str(item.get("canonical_name", ""))
        if not name or name in result:
            raise RuntimeError(f"invalid or duplicate canonical artifact name: {name!r}")
        result[name] = item
    return result


def _zip_diff(reference: dict[str, Any], observed: dict[str, Any]) -> dict[str, Any]:
    ref_entries = {item["name"]: item for item in reference.get("zip_entries", [])}
    obs_entries = {item["name"]: item for item in observed.get("zip_entries", [])}
    names = sorted(set(ref_entries) | set(obs_entries))
    differences: dict[str, Any] = {}
    for name in names:
        left = ref_entries.get(name)
        right = obs_entries.get(name)
        if left is None or right is None:
            differences[name] = {"reference": left, "observed": right}
            continue
        fields = (
            "date_time",
            "compress_type",
            "crc",
            "compressed_size",
            "file_size",
            "external_attr",
            "create_system",
        )
        drift = {
            field: {"reference": left.get(field), "observed": right.get(field)}
            for field in fields
            if left.get(field) != right.get(field)
        }
        if drift:
            differences[name] = drift
    return differences


def compare_builds(builds: list[dict[str, Any]]) -> dict[str, Any]:
    if len(builds) < 2:
        raise RuntimeError("at least two independent builds are required")
    reference = builds[0]
    ref_artifacts = _artifact_map(reference)
    mismatches: list[dict[str, Any]] = []

    ref_source = reference.get("source")
    ref_policy = reference.get("policy")
    ref_authority = reference.get("build_authority", {})
    ref_toolchain = ref_authority.get("toolchain", {})

    for observed in builds[1:]:
        root = observed.get("_root")
        if observed.get("source") != ref_source:
            mismatches.append({
                "kind": "source_authority",
                "build": root,
                "reference": ref_source,
                "observed": observed.get("source"),
            })
        if observed.get("policy") != ref_policy:
            mismatches.append({
                "kind": "release_policy",
                "build": root,
                "reference": ref_policy,
                "observed": observed.get("policy"),
            })

        authority = observed.get("build_authority", {})
        for field in ("python_version", "python_implementation", "python_hash_seed"):
            if authority.get(field) != ref_authority.get(field):
                mismatches.append({
                    "kind": "build_authority",
                    "field": field,
                    "build": root,
                    "reference": ref_authority.get(field),
                    "observed": authority.get(field),
                })
        toolchain = authority.get("toolchain", {})
        for distribution in CORE_TOOLCHAIN:
            if toolchain.get(distribution) != ref_toolchain.get(distribution):
                mismatches.append({
                    "kind": "build_toolchain",
                    "distribution": distribution,
                    "build": root,
                    "reference": ref_toolchain.get(distribution),
                    "observed": toolchain.get(distribution),
                })

        artifacts = _artifact_map(observed)
        if set(artifacts) != set(ref_artifacts):
            mismatches.append({
                "kind": "artifact_set",
                "build": root,
                "reference": sorted(ref_artifacts),
                "observed": sorted(artifacts),
            })
            continue

        for name in sorted(ref_artifacts):
            left = ref_artifacts[name]
            right = artifacts[name]
            identity_fields = ("name", "version", "file", "bytes", "sha256")
            drift = {
                field: {"reference": left.get(field), "observed": right.get(field)}
                for field in identity_fields
                if left.get(field) != right.get(field)
            }
            if drift:
                mismatch: dict[str, Any] = {
                    "kind": "wheel_identity",
                    "distribution": name,
                    "build": root,
                    "drift": drift,
                }
                if left.get("sha256") != right.get("sha256"):
                    mismatch["zip_metadata_drift"] = _zip_diff(left, right)
                mismatches.append(mismatch)

    builders = [
        {
            "root": item.get("_root"),
            "environment": item.get("builder_environment"),
            "toolchain": item.get("build_authority", {}).get("toolchain"),
        }
        for item in builds
    ]
    artifact_identity = {
        name: {
            "file": row.get("file"),
            "bytes": row.get("bytes"),
            "sha256": row.get("sha256"),
        }
        for name, row in sorted(ref_artifacts.items())
    }
    return {
        "schema": REPORT_SCHEMA,
        "status": "pass" if not mismatches else "fail",
        "build_count": len(builds),
        "source": ref_source,
        "policy": ref_policy,
        "builders": builders,
        "reference_artifacts": artifact_identity,
        "mismatches": mismatches,
        "claim_boundary": {
            "byte_identical_wheels": not mismatches,
            "runtime_dependency_environment_reproduced": False,
            "scientific_result_reproduced": False,
            "statement": (
                "A passing comparison establishes byte-identical default pure-Python "
                "wheel artifacts under the recorded build authority. It does not establish "
                "reproducibility of downstream dependency resolution, runtime numerics, "
                "hardware behavior, or scientific results."
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("builds", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        builds = [_load_build(path) for path in args.builds]
        report = compare_builds(builds)
    except Exception as exc:
        print(f"reproducible wheel comparison failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    if report["status"] != "pass":
        print("release wheels are not byte-reproducible; inspect comparison report", file=sys.stderr)
        return 1
    print(f"reproducible release wheels: PASS ({report['build_count']} independent builds)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
