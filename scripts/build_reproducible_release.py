#!/usr/bin/env python3
"""Build byte-reproducible default neurOS wheels under explicit authority.

The raw setuptools/wheel output is not assumed to be platform-neutral at the
container byte level. In particular, Windows can emit CRLF core metadata and
Windows-specific ZIP creator/permission fields even when the semantic wheel
content is otherwise identical. This builder therefore performs a narrow Wheel
spec-preserving canonicalization after the package backend has finished:

* text metadata controlled by the build backend uses LF line endings;
* ``RECORD`` is rebuilt from the canonical member bytes;
* members are written in lexical order;
* ZIP timestamps are bound to ``SOURCE_DATE_EPOCH``;
* ZIP creator/mode metadata is normalized to one Unix representation; and
* members use ``ZIP_STORED`` so zlib implementation/version cannot become an
  undeclared cross-builder input.

The canonicalizer is limited to the default platform-independent ``py3-none-any``
release set. A passing invocation proves only that one builder produced a
self-consistent release set. Cross-builder identity is established separately by
``check_reproducible_wheels.py``.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from email.parser import Parser
from importlib import metadata
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import ZIP_STORED, ZipFile, ZipInfo

SCHEMA = "neuros.reproducible_release_build.v1"
CANONICALIZATION_SCHEMA = "neuros.wheel_canonicalization.v1"
TOOLCHAIN_DISTRIBUTIONS = (
    "pip",
    "build",
    "setuptools",
    "wheel",
    "packaging",
    "pyproject-hooks",
)
CANONICAL_TEXT_METADATA = {
    "METADATA",
    "WHEEL",
    "entry_points.txt",
    "top_level.txt",
}
CANONICAL_FILE_MODE = 0o100644
CANONICAL_SCRIPT_MODE = 0o100755
CANONICAL_DIRECTORY_MODE = 0o040755


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record_hash(payload: bytes) -> str:
    digest = hashlib.sha256(payload).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _canonical_name(value: str) -> str:
    return value.lower().replace("_", "-").replace(".", "-")


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=repo_root, text=True, stderr=subprocess.STDOUT
    ).strip()


def _source_authority(
    repo_root: Path, revision: str, source_date_epoch: int
) -> dict[str, Any]:
    head = _git(repo_root, "rev-parse", "HEAD")
    _require(head == revision, f"checkout drift: expected {revision}, observed {head}")
    _require(
        len(revision) == 40 and all(ch in "0123456789abcdef" for ch in revision),
        "source revision must be a 40-character lowercase git SHA",
    )
    tracked_status = _git(repo_root, "status", "--porcelain", "--untracked-files=no")
    _require(
        not tracked_status,
        f"tracked source tree is dirty before build: {tracked_status}",
    )
    commit_epoch_text = _git(repo_root, "show", "-s", "--format=%ct", revision)
    _require(commit_epoch_text.isdigit(), "source commit timestamp is not an integer")
    commit_epoch = int(commit_epoch_text)
    _require(
        source_date_epoch == commit_epoch,
        "SOURCE_DATE_EPOCH must equal the exact source commit timestamp: "
        f"expected {commit_epoch}, observed {source_date_epoch}",
    )
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
    _require(
        payload.get("schema") == "neuros.release_package_inventory.v1",
        "unexpected release policy inventory schema",
    )
    packages = payload.get("packages")
    _require(
        isinstance(packages, list) and packages,
        "release policy selected no packages",
    )
    _require(
        payload.get("count") == len(packages),
        "release policy count does not match selected package list",
    )
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


def _canonical_zip_datetime(source_date_epoch: int) -> tuple[int, int, int, int, int, int]:
    # The ZIP timestamp format starts in 1980 and has a two-second resolution.
    timestamp = datetime.fromtimestamp(source_date_epoch, tz=timezone.utc)
    if timestamp.year < 1980:
        timestamp = datetime(1980, 1, 1, tzinfo=timezone.utc)
    if timestamp.year > 2107:
        timestamp = datetime(2107, 12, 31, 23, 59, 58, tzinfo=timezone.utc)
    second = timestamp.second - (timestamp.second % 2)
    return (
        timestamp.year,
        timestamp.month,
        timestamp.day,
        timestamp.hour,
        timestamp.minute,
        second,
    )


def _validate_member_name(name: str) -> None:
    _require(bool(name), "wheel contains an empty member name")
    path = PurePosixPath(name)
    _require(not path.is_absolute(), f"wheel contains an absolute member path: {name}")
    _require(".." not in path.parts, f"wheel member escapes archive root: {name}")
    _require("\\" not in name, f"wheel member uses a non-POSIX separator: {name}")


def _normalize_line_endings(payload: bytes, *, member: str) -> bytes:
    try:
        payload.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"wheel metadata is not UTF-8: {member}") from exc
    return payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")


def _canonical_record(
    members: dict[str, bytes], *, record_name: str
) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    for name in sorted(members):
        if name == record_name:
            continue
        payload = members[name]
        writer.writerow((name, _record_hash(payload), str(len(payload))))
    writer.writerow((record_name, "", ""))
    return stream.getvalue().encode("utf-8")


def _canonical_mode(name: str, *, is_directory: bool) -> int:
    if is_directory:
        return CANONICAL_DIRECTORY_MODE
    if ".data/scripts/" in name:
        return CANONICAL_SCRIPT_MODE
    return CANONICAL_FILE_MODE


def _canonical_zip_info(
    name: str,
    *,
    timestamp: tuple[int, int, int, int, int, int],
    is_directory: bool,
) -> ZipInfo:
    info = ZipInfo(filename=name, date_time=timestamp)
    info.create_system = 3
    info.compress_type = ZIP_STORED
    info.external_attr = _canonical_mode(name, is_directory=is_directory) << 16
    if is_directory:
        info.external_attr |= 0x10
    return info


def _verify_record(members: dict[str, bytes], *, record_name: str) -> None:
    expected = _canonical_record(members, record_name=record_name)
    _require(
        members.get(record_name) == expected,
        f"canonical RECORD verification failed: {record_name}",
    )


def _canonicalize_wheel(path: Path, *, source_date_epoch: int) -> dict[str, Any]:
    _require(
        path.name.endswith("-py3-none-any.whl"),
        f"wheel canonicalizer accepts only platform-independent wheels: {path.name}",
    )
    before_sha256 = _sha256_file(path)
    with ZipFile(path, "r") as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        _require(len(names) == len(set(names)), f"wheel contains duplicate members: {path.name}")
        for name in names:
            _validate_member_name(name)
        members = {item.filename: archive.read(item.filename) for item in infos}
        directory_names = {item.filename for item in infos if item.is_dir()}

    record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    _require(len(record_names) == 1, f"{path.name}: expected exactly one RECORD member")
    _require(len(metadata_names) == 1, f"{path.name}: expected exactly one METADATA member")
    dist_info = record_names[0].rsplit("/", 1)[0] + "/"
    _require(
        metadata_names[0].startswith(dist_info),
        f"{path.name}: METADATA and RECORD belong to different dist-info trees",
    )
    record_name = record_names[0]

    normalized_text: list[str] = []
    for name in sorted(members):
        if not name.startswith(dist_info):
            continue
        basename = name.rsplit("/", 1)[-1]
        if basename not in CANONICAL_TEXT_METADATA:
            continue
        normalized = _normalize_line_endings(members[name], member=name)
        if normalized != members[name]:
            normalized_text.append(name)
        members[name] = normalized

    members[record_name] = _canonical_record(members, record_name=record_name)
    _verify_record(members, record_name=record_name)

    timestamp = _canonical_zip_datetime(source_date_epoch)
    temporary = path.with_suffix(path.suffix + ".canonical.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        with ZipFile(temporary, "w", compression=ZIP_STORED, strict_timestamps=True) as archive:
            for name in sorted(members):
                is_directory = name in directory_names or name.endswith("/")
                payload = b"" if is_directory else members[name]
                info = _canonical_zip_info(
                    name,
                    timestamp=timestamp,
                    is_directory=is_directory,
                )
                archive.writestr(info, payload)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()

    after_sha256 = _sha256_file(path)
    with ZipFile(path, "r") as archive:
        rewritten = {item.filename: archive.read(item.filename) for item in archive.infolist()}
        rewritten_infos = archive.infolist()
    _verify_record(rewritten, record_name=record_name)
    _require(
        [item.filename for item in rewritten_infos] == sorted(rewritten),
        f"canonical wheel member order is not lexical: {path.name}",
    )
    for item in rewritten_infos:
        _require(item.date_time == timestamp, f"non-canonical ZIP timestamp: {item.filename}")
        _require(item.create_system == 3, f"non-canonical ZIP creator: {item.filename}")
        _require(item.compress_type == ZIP_STORED, f"non-canonical compression: {item.filename}")
        expected_attr = _canonical_mode(
            item.filename, is_directory=item.is_dir()
        ) << 16
        if item.is_dir():
            expected_attr |= 0x10
        _require(
            item.external_attr == expected_attr,
            f"non-canonical ZIP mode for {item.filename}: "
            f"expected {expected_attr}, observed {item.external_attr}",
        )

    return {
        "schema": CANONICALIZATION_SCHEMA,
        "wheel": path.name,
        "sha256_before": before_sha256,
        "sha256_after": after_sha256,
        "source_date_epoch": source_date_epoch,
        "zip_datetime_utc": list(timestamp),
        "compression": "stored",
        "create_system": 3,
        "default_file_mode": oct(CANONICAL_FILE_MODE),
        "script_mode": oct(CANONICAL_SCRIPT_MODE),
        "normalized_text_members": normalized_text,
        "record": record_name,
    }


def _wheel_metadata(path: Path) -> dict[str, Any]:
    with ZipFile(path) as archive:
        members = archive.infolist()
        metadata_names = [
            item.filename
            for item in members
            if item.filename.endswith(".dist-info/METADATA")
        ]
        _require(
            len(metadata_names) == 1,
            f"{path.name}: expected exactly one METADATA member",
        )
        message = Parser().parsestr(
            archive.read(metadata_names[0]).decode("utf-8", errors="strict")
        )
        name = message.get("Name")
        version = message.get("Version")
        _require(
            bool(name) and bool(version),
            f"{path.name}: missing distribution name/version",
        )
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
    _require(
        path.name.endswith("-py3-none-any.whl"),
        f"default release wheel is not platform-independent: {path.name}",
    )
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

    for item in selected:
        package_root = (repo_root / item["path"]).resolve()
        _require(
            package_root.is_dir(),
            f"release package directory does not exist: {package_root}",
        )
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
    _require(
        len(built_paths) == len(selected),
        f"expected {len(selected)} release wheels, built {len(built_paths)}",
    )
    canonicalization = [
        _canonicalize_wheel(path, source_date_epoch=source_date_epoch)
        for path in built_paths
    ]
    artifacts = [_wheel_metadata(path) for path in built_paths]
    expected_names = sorted(_canonical_name(item["distribution"]) for item in selected)
    observed_names = sorted(item["canonical_name"] for item in artifacts)
    _require(
        observed_names == expected_names,
        f"built release distribution set differs from policy: "
        f"{observed_names} != {expected_names}",
    )

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
            "wheel_canonicalization": CANONICALIZATION_SCHEMA,
        },
        "builder_environment": {
            "system": platform.system(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "python_executable": sys.executable,
        },
        "canonicalization": canonicalization,
        "artifacts": artifacts,
    }
    manifest_path = output / "reproducible-build-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
        print(
            f"reproducible release build failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
