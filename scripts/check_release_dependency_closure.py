#!/usr/bin/env python3
"""Verify the default neurOS release wheel set is internally dependency-closed.

The check covers active *default* dependencies between neurOS distributions
across the supported Python/runtime matrix, rather than only the CI runner that
happened to build the wheels. Optional-extra edges do not enter the default
runtime unless an active internal requirement explicitly requests an extra. Such
requested extras fail closed until this verifier models their dependency
propagation explicitly.

External third-party requirements remain the package resolver's responsibility;
this contract prevents a release candidate from silently fetching an omitted or
incompatible neurOS distribution from a package index.
"""

from __future__ import annotations

import argparse
import json
import sys
from email.parser import Parser
from itertools import product
from pathlib import Path
from typing import Any, Mapping
from zipfile import BadZipFile, ZipFile

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

SCHEMA = "neuros.release_dependency_closure.v1"
SUPPORTED_PYTHON_VERSIONS = ("3.10", "3.11", "3.12")
_PLATFORM_TARGETS = (
    ("linux-x86_64", "linux", "Linux", "posix", "x86_64"),
    ("linux-aarch64", "linux", "Linux", "posix", "aarch64"),
    ("macos-x86_64", "darwin", "Darwin", "posix", "x86_64"),
    ("macos-arm64", "darwin", "Darwin", "posix", "arm64"),
    ("windows-amd64", "win32", "Windows", "nt", "AMD64"),
    ("windows-arm64", "win32", "Windows", "nt", "ARM64"),
)


def _is_internal(name: str) -> bool:
    canonical = canonicalize_name(name)
    return canonical == "neuros" or canonical.startswith("neuros-")


def _target_environments() -> tuple[tuple[str, dict[str, str]], ...]:
    targets: list[tuple[str, dict[str, str]]] = []
    for python_version, platform in product(SUPPORTED_PYTHON_VERSIONS, _PLATFORM_TARGETS):
        platform_id, sys_platform, platform_system, os_name, platform_machine = platform
        environment = default_environment()
        full_version = f"{python_version}.0"
        environment.update(
            {
                "python_version": python_version,
                "python_full_version": full_version,
                "implementation_name": "cpython",
                "implementation_version": full_version,
                "platform_python_implementation": "CPython",
                "sys_platform": sys_platform,
                "platform_system": platform_system,
                "os_name": os_name,
                "platform_machine": platform_machine,
                "extra": "",
            }
        )
        targets.append((f"cp{python_version}-{platform_id}", environment))
    return tuple(targets)


TARGET_ENVIRONMENTS = _target_environments()


def _active_targets(requirement: Requirement) -> tuple[str, ...]:
    if requirement.marker is None:
        return tuple(target_id for target_id, _ in TARGET_ENVIRONMENTS)
    active = [
        target_id
        for target_id, environment in TARGET_ENVIRONMENTS
        if requirement.marker.evaluate(environment)
    ]
    return tuple(active)


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
            active_targets = _active_targets(requirement)
            if not active_targets:
                continue
            if requirement.extras:
                raise ValueError(
                    "active internal runtime dependency requests extras whose closure is not "
                    f"modeled: {source} -> {raw}; active_targets={list(active_targets)}"
                )
            target_meta = distributions.get(target)
            if target_meta is None:
                raise ValueError(
                    "unsatisfied internal runtime dependency: "
                    f"{source} requires {raw!r}, but {target!r} is absent from release set; "
                    f"active_targets={list(active_targets)}"
                )
            target_version = Version(target_meta["version"])
            if requirement.specifier and not requirement.specifier.contains(
                target_version, prereleases=True
            ):
                raise ValueError(
                    "incompatible internal runtime dependency: "
                    f"{source} requires {raw!r}, but bundled {target}=={target_version}; "
                    f"active_targets={list(active_targets)}"
                )
            dependencies.append(
                {
                    "from": source,
                    "requirement": raw,
                    "to": target,
                    "resolved_version": str(target_version),
                    "active_targets": list(active_targets),
                }
            )

    return {
        "schema": SCHEMA,
        "status": "pass",
        "python_versions": list(SUPPORTED_PYTHON_VERSIONS),
        "target_environments": [target_id for target_id, _ in TARGET_ENVIRONMENTS],
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
        f"{len(report['dependencies'])} active internal edges, "
        f"{len(report['target_environments'])} target environments)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
