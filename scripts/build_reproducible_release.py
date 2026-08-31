#!/usr/bin/env python3
"""Build the default neurOS release wheels under an explicit build authority.

This builder is intentionally narrower than the ordinary release workflow. It
expects the caller to install the hash-pinned Python 3.11 build toolchain first,
requires the source commit timestamp as SOURCE_DATE_EPOCH, disables PEP 517
build isolation, and emits diagnostics sufficient to explain byte-level wheel
drift across independent builders.

A passing invocation proves only that one builder produced a self-consistent
set of platform-independent wheels. Cross-builder reproducibility is established
separately by ``check_reproducible_wheels.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from email.parser import Parser
from importlib import metadata
from pathlib import Path
from typing import Any
from zipfile import ZipFile

SCHEMA = "neuros.reproducible_release_build.v1"
TOOLCHAIN_DISTRIBUTIONS = (
    "pip",
    "build",
    "setuptools",
    "wheel",
    "packaging",
    "pyproject-hooks",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _source_authority(repo_root: Path, revision: str, source_date_epoch: int) -> dict[str, Any]:
    head = _git(repo_root, "rev-parse", "HEAD")
    _require(head == revision, f"checkout drift: expected {revision}, observed {head}")
    _require(
        len(revision) == 40 and all(ch in "0123456789abcdef" for ch in revision),
        "source revision must be a 40-character lowercase git SHA",
    )
    tracked_status = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    _require(not tracked_status, f"tracked source tree is dirty before build: {tracked_status}")
    commit_epoch_text = _git(repo_root, "show", "-s", "--format=%ct", revision)
    _require(commit_epoch_text.isdigit(), "source commit timestamp is not an integer")
    commit_epoch = int(commit_epoch_text)
    _require(source_date_epoch == commit_epoch, (
        "SOURCE_DATE_EPOCH must equal the exact source commit timestamp: "
        f"expected {commit_epoch}, observed {source_date_epoch}"
    ))
    return {
        "revision": revision,
        "commit_timestamp_epoch": commit_epoch,
        "source_date_epoch": source_date_epoch,
    }


def _selected_packages(repo_root: Path) -> list[dict[str, Any]]:
    payload = json.loads(
        subprocess.check_output(
            [sys.executable, repo_root / "scripts/list_release_packages.py", "--json"],
            cwd=repo_root,
            text=True,
        )
    )
    _require(payload.get("schema") == "neuros.release_package_inventory.v1", "unexpected release policy inventory schema")
    packages = payload.get("packages")
    _require(isinstance(packages, list) and packages, "release policy selected no packages")
    _require(payload.get("count") == len(packages), "release policy count does not match selected package list")
    return packages


def _clean_build_state(package_root: Path) -> None:
    for path in (package_root / "build", package_root / "dist"):
        if path.exists():
            shutil.rmtree(path)
    for parent in (package_root, package_root / "src"):
        if not parent.is_dir():
            continue
        for path in parent.glob("*.egg-info"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()


def _wheel_metadata(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        members = archive.infolist()
        metadata_names = [item.filename for item in members if item.filename.endswith(".dist-info/METADATA")]
        _require(len(metadata_names) == 1, f"{path.name}: expected exactly one METADATA member")
        message = Parser().parsestr(
            archive.read(metadata_names[0]).decode("utf-8", errors="strict")
        )
        name = message.get("Name")
        version = message.get("Version")
        _require(bool(name) and bool(version), f"{path.name}: missing distribution name/version")
        zip_entries = [
            {
                "name": item.filename,
                "date_time": list(item.date_time),
                "compress_type": item.compress_type,
                "crc": item.CRC,
                "compressed_size": item.compress_size,
                "file_size": item.file_size,
                "external_attr": item.external_attr,
                "create_system": item.create_system,
            }
            for item in members
        ]
    _require(path.name.endswith("-py3-none-any.whl"), f"default release wheel is not platform-independent: {path.name}")
    zip_metadata_sha = hashlib.sha256(
        json.dumps(zip_entries, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "name": str(name),
        "canonical_name": _canonical_name(str(name)),
        "version": str(version),
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "zip_metadata_sha256": zip_metadata_sha,
        "zip_entries": zip_entries,
    }


def _toolchain_identity() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in TOOLCHAIN_DISTRIBUTIONS:
        versions[distribution] = metadata.version(distribution)
    try:
        versions["colorama"] = metadata.version("colorama")
    except metadata.PackageNotFoundError:
        versions["colorama"] = "absent"
    return versions


def build_release(
    *,
    repo_root: Path,
    output: Path,
    source_revision: str,
    source_date_epoch: int,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output = output.resolve()
    _require(not output.exists(), f"output already exists: {output}")
    output.mkdir(parents=True)

    source = _source_authority(repo_root, source_revision, source_date_epoch)
    selected = _selected_packages(repo_root)
    toolchain = _toolchain_identity()
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = str(source_date_epoch)
    env["PYTHONHASHSEED"] = "0"

    built_paths: list[Path] = []
    for item in selected:
        package_root = (repo_root / item["path"]).resolve()
        _require(package_root.is_dir(), f"release package directory does not exist: {package_root}")
        _clean_build_state(package_root)
        try:
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "build",
                    "--no-isolation",
                    "--wheel",
                    "--outdir",
                    str(output),
                    str(package_root),
                ],
                cwd=repo_root,
                env=env,
                check=True,
            )
        finally:
            _clean_build_state(package_root)

    built_paths = sorted(output.glob("*.whl"))
    _require(len(built_paths) == len(selected), (
        f"expected {len(selected)} release wheels, built {len(built_paths)}"
    ))
    artifacts = [_wheel_metadata(path) for path in built_paths]
    expected_names = sorted(_canonical_name(item["distribution"]) for item in selected)
    observed_names = sorted(item["canonical_name"] for item in artifacts)
    _require(observed_names == expected_names, (
        f"built release distribution set differs from policy: {observed_names} != {expected_names}"
    ))

    manifest = {
        "schema": SCHEMA,
        "source": source,
        "policy": {
            "count": len(selected),
            "distributions": expected_names,
        },
        "build_authority": {
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "toolchain": toolchain,
            "python_hash_seed": env["PYTHONHASHSEED"],
        },
        "builder_environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
        },
        "artifacts": artifacts,
    }
    manifest_path = output / "reproducible-build-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--source-date-epoch", type=int, required=True)
    args = parser.parse_args()

    if args.source_date_epoch <= 0:
        raise SystemExit("--source-date-epoch must be positive")
    try:
        build_release(
            repo_root=args.repo_root,
            output=args.output,
            source_revision=args.source_revision,
            source_date_epoch=args.source_date_epoch,
        )
    except Exception as exc:
        print(f"reproducible release build failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
