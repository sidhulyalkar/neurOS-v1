#!/usr/bin/env python3
"""Verify the default neurOS release wheel set is internally dependency-closed.

The check is intentionally scoped to active *default* dependencies between neurOS
distributions. Optional-extra edges do not enter the default runtime unless an
active internal requirement explicitly requests an extra. Such requested extras
fail closed until this verifier models their dependency propagation explicitly.

External third-party requirements remain the package resolver's responsibility;
this contract prevents a release candidate from silently fetching an omitted
neurOS distribution from a package index.
"""

from __future__ import annotations

import argparse
import json
import sys
from email.parser import Parser
from pathlib import Path
from typing import Any
from zipfile import BadZipFile, ZipFile

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

SCHEMA = "neuros.release_dependency_closure.v1"


def _is_internal(name: str) -> bool:
    canonical = canonicalize_name(name)
    return canonical == "neuros" or canonical.startswith("neuros-")


def _metadata(wheel: Path) -> tuple[str, str, list[str]]:
    try:
        with ZipFile(wheel) as archive:
            metadata_names = [
                name for name in archive.namelist() if name.endswith(".dist-info/METADATA")
            ]
            if len(metadata_names) != 1:
                raise ValueError(
                    f"{wheel.name}: expected exactly one .dist-info/METADATA, "
                    f"found {len(metadata_names)}"
                )
            text = archive.read(metadata_names[0]).decode("utf-8", errors="strict")
    except BadZipFile as exc:
        raise ValueError(f"invalid wheel archive: {wheel}") from exc
    message = Parser().parsestr(text)
    name = message.get("Name")
    version = message.get("Version")
    if not name or not version:
        raise ValueError(f"{wheel.name}: METADATA must contain Name and Version")
    try:
        Version(str(version))
    except InvalidVersion as exc:
        raise ValueError(f"{wheel.name}: invalid Version metadata {version!r}") from exc
    return str(name), str(version), list(message.get_all("Requires-Dist") or [])


def inspect_release_dependency_closure(wheel_dir: Path) -> dict[str, Any]:
    wheels = sorted(wheel_dir.glob("*.whl"))
    if not wheels:
        raise ValueError(f"no wheels found in {wheel_dir}")

    distributions: dict[str, dict[str, Any]] = {}
    requirements_by_distribution: dict[str, list[str]] = {}
    for wheel in wheels:
        display_name, version, requirements = _metadata(wheel)
        canonical = canonicalize_name(display_name)
        if canonical in distributions:
            raise ValueError(f"duplicate distribution in release set: {display_name}")
        distributions[canonical] = {
            "name": display_name,
            "version": version,
            "wheel": wheel.name,
        }
        requirements_by_distribution[canonical] = requirements

    dependencies: list[dict[str, Any]] = []
    for source in sorted(distributions):
        for raw in requirements_by_distribution[source]:
            try:
                requirement = Requirement(raw)
            except InvalidRequirement as exc:
                raise ValueError(
                    f"{distributions[source]['wheel']}: invalid Requires-Dist {raw!r}"
                ) from exc
            target = canonicalize_name(requirement.name)
            if not _is_internal(target):
                continue
            if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
                continue
            if requirement.extras:
                raise ValueError(
                    "active internal runtime dependency requests extras whose closure is not "
                    f"modeled: {source} -> {raw}"
                )
            target_meta = distributions.get(target)
            if target_meta is None:
                raise ValueError(
                    "unsatisfied internal runtime dependency: "
                    f"{source} requires {raw!r}, but {target!r} is absent from release set"
                )
            target_version = Version(target_meta["version"])
            if requirement.specifier and not requirement.specifier.contains(
                target_version, prereleases=True
            ):
                raise ValueError(
                    "incompatible internal runtime dependency: "
                    f"{source} requires {raw!r}, but bundled {target}=={target_version}"
                )
            dependencies.append(
                {
                    "from": source,
                    "requirement": raw,
                    "to": target,
                    "resolved_version": str(target_version),
                }
            )

    return {
        "schema": SCHEMA,
        "status": "pass",
        "distribution_count": len(distributions),
        "distributions": dict(sorted(distributions.items())),
        "dependencies": dependencies,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        report = inspect_release_dependency_closure(args.wheel_dir.resolve())
    except (OSError, ValueError) as exc:
        print(f"release dependency closure: ERROR: {exc}", file=sys.stderr)
        return 2
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(
        "release dependency closure: PASS "
        f"({report['distribution_count']} distributions, "
        f"{len(report['dependencies'])} active internal edges)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
