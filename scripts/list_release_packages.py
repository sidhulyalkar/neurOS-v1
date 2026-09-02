#!/usr/bin/env python3
"""Validate and list the explicit neurOS release package policy.

Workspace membership answers "is this developed here?". This policy separately
answers "is this distribution part of the default release artifact set?" and
records scientific maturity without using version numbers as a promotion signal.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from list_workspace_packages import ROOT, workspace_members

POLICY_FILE = ROOT / "release" / "package-policy.json"
ALLOWED_TIERS = {
    "public-runtime",
    "qualified-integration",
    "research-extension",
    "internal-preview",
}
ALLOWED_MATURITY = {
    "platform-core",
    "qualified-integration",
    "synthetic-validation",
    "research-qualified",
    "research",
    "prototype",
}


def _project_name(pyproject: Path) -> str:
    text = pyproject.read_text(encoding="utf-8")
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        section = re.search(r"(?ms)^\[project\]\s*(.*?)(?=^\[|\Z)", text)
        if section is None:
            raise ValueError(f"missing [project] section: {pyproject}")
        match = re.search(r'(?m)^name\s*=\s*"([^"\n]+)"\s*$', section.group(1))
        if match is None:
            raise ValueError(f"missing project.name: {pyproject}")
        return match.group(1)
    payload = tomllib.loads(text)
    try:
        name = payload["project"]["name"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"missing project.name: {pyproject}") from exc
    if not isinstance(name, str) or not name:
        raise ValueError(f"project.name must be a non-empty string: {pyproject}")
    return name


def release_policy(path: Path = POLICY_FILE) -> tuple[dict[str, Any], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "neuros.release_package_policy.v1":
        raise ValueError("unexpected release package policy schema")
    raw = payload.get("packages")
    if not isinstance(raw, list) or not raw:
        raise ValueError("release package policy must contain packages")

    entries: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    seen_distributions: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"release package policy entry {index} must be an object")
        expected_fields = {
            "path",
            "distribution",
            "release_tier",
            "scientific_maturity",
            "publish_candidate",
        }
        if set(item) != expected_fields:
            raise ValueError(
                f"release package policy entry {index} fields differ from canonical schema"
            )
        member = item["path"]
        distribution = item["distribution"]
        tier = item["release_tier"]
        maturity = item["scientific_maturity"]
        publish = item["publish_candidate"]
        if not isinstance(member, str) or not member:
            raise ValueError("release package path must be a non-empty string")
        if not isinstance(distribution, str) or not distribution:
            raise ValueError("release distribution must be a non-empty string")
        if tier not in ALLOWED_TIERS:
            raise ValueError(f"unsupported release tier for {member}: {tier}")
        if maturity not in ALLOWED_MATURITY:
            raise ValueError(f"unsupported scientific maturity for {member}: {maturity}")
        if not isinstance(publish, bool):
            raise ValueError(f"publish_candidate must be boolean for {member}")
        if tier == "public-runtime" and not publish:
            raise ValueError(f"public-runtime package must be a publish candidate: {member}")
        if tier != "public-runtime" and publish:
            raise ValueError(
                f"non-public-runtime package cannot enter the default release set: {member}"
            )
        if member in seen_paths:
            raise ValueError(f"duplicate release package path: {member}")
        if distribution in seen_distributions:
            raise ValueError(f"duplicate release distribution: {distribution}")
        seen_paths.add(member)
        seen_distributions.add(distribution)

        pyproject = ROOT / member / "pyproject.toml"
        if not pyproject.is_file():
            raise ValueError(f"release package has no pyproject.toml: {member}")
        actual_name = _project_name(pyproject)
        if actual_name != distribution:
            raise ValueError(
                f"release policy distribution mismatch for {member}: "
                f"policy={distribution!r}, project={actual_name!r}"
            )
        entries.append(dict(item))

    workspace = set(workspace_members())
    policy_members = {item["path"] for item in entries}
    if workspace != policy_members:
        missing = sorted(workspace - policy_members)
        foreign = sorted(policy_members - workspace)
        raise ValueError(
            "release package policy must classify every workspace member exactly once; "
            f"missing={missing}, foreign={foreign}"
        )

    required_runtime = {
        "packages/neuros-core",
        "packages/neuros-drivers",
        "packages/neuros-models",
        "packages/neuros",
    }
    publish_paths = {item["path"] for item in entries if item["publish_candidate"]}
    if not required_runtime.issubset(publish_paths):
        raise ValueError(
            "default release set must contain SDK dependency closure: "
            f"missing={sorted(required_runtime - publish_paths)}"
        )
    return tuple(entries)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="list every classified workspace package")
    parser.add_argument("--count", action="store_true", help="print only the selected package count")
    parser.add_argument("--json", action="store_true", help="print the selected package inventory as JSON")
    parser.add_argument("--tier", choices=sorted(ALLOWED_TIERS), help="select one release tier")
    args = parser.parse_args()

    entries = release_policy()
    if args.all:
        selected = entries
    elif args.tier:
        selected = tuple(item for item in entries if item["release_tier"] == args.tier)
    else:
        selected = tuple(item for item in entries if item["publish_candidate"])

    if args.count:
        print(len(selected))
    elif args.json:
        print(
            json.dumps(
                {
                    "schema": "neuros.release_package_inventory.v1",
                    "count": len(selected),
                    "packages": selected,
                },
                indent=2,
                sort_keys=True,
            )
        )
    else:
        print("\n".join(item["path"] for item in selected))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
