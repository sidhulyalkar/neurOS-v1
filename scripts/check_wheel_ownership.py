#!/usr/bin/env python3
"""Fail closed when neurOS wheel distributions own ambiguous installed files.

Sharing the ``neuros`` Python namespace is intentional. Sharing ownership of an
installed file is not: pip records file ownership per distribution, so removing
one wheel can delete a path another wheel still expects. A single wheel also
must not encode two archive members that normalize to the same install target.
This checker inspects built wheel payloads and emits a deterministic ownership
manifest.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from email.parser import Parser
from pathlib import Path, PurePosixPath
from typing import Any
from zipfile import BadZipFile, ZipFile

SCHEMA = "neuros.wheel_ownership.v1"
SDK_ROOT = "neuros/__init__.py"
SDK_DISTRIBUTION = "neuros"


def _normalize_name(name: str) -> str:
    """Canonicalize a distribution name using the packaging-name rule."""

    return re.sub(r"[-_.]+", "-", name.strip()).lower()


def _metadata(archive: ZipFile, wheel: Path) -> tuple[str, str]:
    metadata_names = [
        name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
    ]
    if len(metadata_names) != 1:
        raise ValueError(
            f"{wheel.name}: expected exactly one .dist-info/METADATA, found {len(metadata_names)}"
        )
    text = archive.read(metadata_names[0]).decode("utf-8", errors="strict")
    message = Parser().parsestr(text)
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ValueError(f"{wheel.name}: METADATA must contain Name and Version")
    return str(name), str(version)


def _validate_relative_parts(parts: tuple[str, ...], member: str) -> None:
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"unsafe or ambiguous wheel member path: {member}")
    if member.startswith("/") or member.startswith("\\"):
        raise ValueError(f"absolute wheel member path is not allowed: {member}")


def _installed_path(member: str) -> str | None:
    """Map a wheel archive member to its logical installed destination.

    dist-info metadata is distribution-private and intentionally excluded.
    ``.data/purelib`` and ``.data/platlib`` payloads install into the same import
    root as ordinary wheel files, so they are normalized to that shared path.
    Other wheel data schemes retain an explicit scheme prefix.
    """

    if not member or member.endswith("/") or ".dist-info/" in member:
        return None

    parts = PurePosixPath(member).parts
    _validate_relative_parts(parts, member)
    for index, part in enumerate(parts):
        if not part.endswith(".data"):
            continue
        if index + 2 >= len(parts):
            raise ValueError(f"malformed wheel .data member: {member}")
        scheme = parts[index + 1]
        remainder_parts = parts[index + 2 :]
        _validate_relative_parts(remainder_parts, member)
        remainder = PurePosixPath(*remainder_parts).as_posix()
        if scheme in {"purelib", "platlib"}:
            return remainder
        return f"@{scheme}/{remainder}"

    return PurePosixPath(*parts).as_posix()


def inspect_wheels(wheel_dir: Path) -> dict[str, Any]:
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise ValueError(f"no wheels found in {wheel_dir}")

    owners: dict[str, set[str]] = defaultdict(set)
    distributions: dict[str, dict[str, Any]] = {}

    for wheel in wheels:
        try:
            with ZipFile(wheel) as archive:
                display_name, version = _metadata(archive, wheel)
                distribution = _normalize_name(display_name)
                if distribution in distributions:
                    previous = distributions[distribution]["wheel"]
                    raise ValueError(
                        f"duplicate distribution name {display_name!r}: {previous} and {wheel.name}"
                    )

                payload_count = 0
                installed_members: dict[str, str] = {}
                for member in archive.namelist():
                    installed = _installed_path(member)
                    if installed is None:
                        continue
                    previous_member = installed_members.get(installed)
                    if previous_member is not None:
                        raise ValueError(
                            f"{wheel.name}: multiple archive members install to {installed!r}: "
                            f"{previous_member!r} and {member!r}"
                        )
                    installed_members[installed] = member
                    owners[installed].add(distribution)
                    payload_count += 1

                distributions[distribution] = {
                    "name": display_name,
                    "version": version,
                    "wheel": wheel.name,
                    "payload_file_count": payload_count,
                }
        except BadZipFile as exc:
            raise ValueError(f"invalid wheel archive: {wheel}") from exc

    collisions = [
        {"path": path, "owners": sorted(path_owners)}
        for path, path_owners in sorted(owners.items())
        if len(path_owners) > 1
    ]
    root_owners = sorted(owners.get(SDK_ROOT, set()))
    sdk_root_ok = root_owners == [SDK_DISTRIBUTION]

    return {
        "schema": SCHEMA,
        "status": "pass" if not collisions and sdk_root_ok else "fail",
        "wheel_count": len(wheels),
        "distribution_count": len(distributions),
        "distributions": dict(sorted(distributions.items())),
        "ownership": {
            path: sorted(path_owners) for path, path_owners in sorted(owners.items())
        },
        "collisions": collisions,
        "sdk_root": {
            "path": SDK_ROOT,
            "expected_owner": SDK_DISTRIBUTION,
            "owners": root_owners,
            "pass": sdk_root_ok,
        },
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path, help="Directory containing built wheel files")
    parser.add_argument("--output", type=Path, help="Optional JSON ownership-manifest path")
    args = parser.parse_args()

    try:
        report = inspect_wheels(args.wheel_dir.resolve())
    except (OSError, ValueError) as exc:
        print(f"wheel ownership: ERROR: {exc}", file=sys.stderr)
        return 2

    if args.output:
        _write_report(args.output, report)

    if report["status"] != "pass":
        for collision in report["collisions"]:
            print(
                f"wheel ownership collision: {collision['path']} <- "
                + ", ".join(collision["owners"]),
                file=sys.stderr,
            )
        if not report["sdk_root"]["pass"]:
            print(
                "wheel ownership root error: "
                f"{SDK_ROOT} must be owned only by {SDK_DISTRIBUTION}; "
                f"found {report['sdk_root']['owners']}",
                file=sys.stderr,
            )
        return 1

    print(
        "wheel ownership: PASS "
        f"({report['distribution_count']} distributions, "
        f"{len(report['ownership'])} installed paths, zero collisions)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
