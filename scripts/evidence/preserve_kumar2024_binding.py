#!/usr/bin/env python3
"""Verify and preserve the original Kumar2024 CPU binding ZIP, without neurOS.

This is an archival/transport helper, NOT an experiment runner. Never executes a
worker, downloads neural data, emits model scores, or changes frozen authority.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

SOURCE_REVISION = "56fd0c5132bec17575d68f62256cb80fd5661395"
ARCHIVE_SHA256 = "107a9fc57fc913815131cdf165bc35d3a1130c8300828f1f97672b27441ef0f6"
BUNDLE_SHA256 = "45679a95c614e2107f64d7cb9ce1f87f10179c617ff160bccfd899b7ff8688d3"
ENVIRONMENT_SHA256 = "c45e15561ab95b8a4be0734f2fecd993fca53bf24a6e38b3c8739e1424cd1cb9"
EXECUTION_PLAN_SHA256 = "987bb3b5566d1d481141d9a549f3588994d34e25d05cc9536baccaaa4a4641ac"
BUNDLE_SCHEMA = "neuros.nsq_kumar2024_promoted_binding_bundle.v1"
MANAGED_FILES = frozenset({
    "binding_manifest.json", "case_authorities.json", "environment_lock.json",
    "execution_plan.json", "materialization.json", "method_specs.json",
})
ALL_FILES = MANAGED_FILES | {"artifact_hashes.json"}
COUNTS = {
    "participants": 18,
    "case_authorities": 270,
    "method_realizations": 5,
    "execution_shards": 1350,
    "planned_fit_attempts": 6750,
}


class BindingArchiveError(ValueError):
    """Frozen binding does not satisfy the byte/claim contract."""


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise BindingArchiveError(message)


def _unique_members(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _check(key not in result, "duplicate JSON member")
        result[key] = value
    return result


def _read_json(blob: bytes) -> dict[str, Any]:
    value = json.loads(blob, object_pairs_hook=_unique_members)
    _check(isinstance(value, dict), "expected a JSON object")
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_archive(source: Path) -> dict[str, Any]:
    """Independently check the exact frozen ZIP, six members and bundle root."""
    _check(source.is_file() and not source.is_symlink(), "source must be a regular ZIP file")
    _check(_hash_file(source) == ARCHIVE_SHA256, "original ZIP SHA-256 mismatch")
    with zipfile.ZipFile(source, "r") as archive:
        members = archive.infolist()
        names = [member.filename for member in members]
        _check(len(names) == len(ALL_FILES) and set(names) == ALL_FILES,
               "binding ZIP member inventory mismatch")
        for member in members:
            mode = member.external_attr >> 16
            _check(not member.is_dir() and not stat.S_ISLNK(mode), "invalid ZIP member type")
            _check(member.flag_bits & 1 == 0, "encrypted ZIP member not permitted")
            _check(member.file_size <= 16 * 1024 * 1024, "ZIP member exceeds size cap")
        hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in sorted(MANAGED_FILES)
        }
        index = _read_json(archive.read("artifact_hashes.json"))
        _check(index.get("schema_version") == 1 and index.get("files") == hashes,
               "binding member SHA-256 manifest mismatch")
        root = hashlib.sha256(json.dumps(
            {"schema": BUNDLE_SCHEMA, "payload": {"files": hashes}},
            sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")).hexdigest()
        _check(root == BUNDLE_SHA256 and index.get("bundle_sha256") == root,
               "domain-separated binding root mismatch")
        manifest = _read_json(archive.read("binding_manifest.json"))
        _check(manifest.get("source_revision") == SOURCE_REVISION, "frozen source revision mismatch")
        _check(manifest.get("environment_authority_sha256") == ENVIRONMENT_SHA256,
               "environment authority mismatch")
        _check(manifest.get("execution_plan_sha256") == EXECUTION_PLAN_SHA256,
               "execution plan authority mismatch")
        _check(manifest.get("counts") == COUNTS, "frozen execution counts mismatch")
        for key in (
            "model_execution_performed",
            "final_assessment_predictions_generated",
            "final_assessment_metrics_generated",
        ):
            _check(manifest.get(key) is False, f"binding must declare {key}=false")
    return {
        "schema": "neuros.kumar2024_binding_archive_check.v1",
        "verified": True,
        "source_revision": SOURCE_REVISION,
        "zip_sha256": ARCHIVE_SHA256,
        "bundle_sha256": root,
        "verified_managed_files": len(hashes),
        "counts": COUNTS,
        "model_execution_performed": False,
        "scientific_results_interpretable": False,
        "verification_scope": "archival_bytes_and_binding_declarations",
    }


def preserve(source: Path, destination: Path) -> dict[str, Any]:
    """Publish exactly the verified bytes using a write-once filesystem link."""
    receipt = verify_archive(source)
    _check(destination.parent.is_dir(), "destination parent must already exist")
    _check(not destination.is_symlink(), "destination cannot be a symlink")
    _check(source.resolve() != destination.resolve(), "source and destination must differ")
    # Same-directory temporary and hardlink offer atomic create-if-absent semantics;
    # replacing or modifying an existing archived attempt is forbidden.
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=".neuros-binding-", suffix=".tmp",
            dir=destination.parent, delete=False,
        ) as output:
            temporary = Path(output.name)
            os.chmod(temporary, 0o600)
            with source.open("rb") as incoming:
                shutil.copyfileobj(incoming, output, length=1024 * 1024)
            output.flush()
            os.fsync(output.fileno())
        verify_archive(temporary)  # Reject source mutation during copy.
        try:
            os.link(temporary, destination)
        except FileExistsError as exc:
            raise BindingArchiveError("destination already exists; overwrite forbidden") from exc
        # Re-read the published link and flush the directory entry on POSIX.
        # This is local crash-consistency, not a claim of off-host retention.
        verify_archive(destination)
        if os.name == "posix":
            directory_fd = os.open(
                destination.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        receipt["preserved_at"] = str(destination.absolute())
        receipt["write_once"] = True
        return receipt
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-zip", type=Path, required=True,
                        help="The exact, previously downloaded GitHub Actions ZIP")
    parser.add_argument("--destination", type=Path,
                        help="New write-once archival file (omit for verification only)")
    args = parser.parse_args()
    try:
        result = (verify_archive(args.source_zip) if args.destination is None
                  else preserve(args.source_zip, args.destination))
    except (BindingArchiveError, OSError, zipfile.BadZipFile, UnicodeError,
            json.JSONDecodeError) as exc:
        print(json.dumps({"verified": False, "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
