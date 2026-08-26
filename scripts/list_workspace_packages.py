#!/usr/bin/env python3
"""Print and validate the canonical neurOS workspace package inventory.

The repository-level ``pyproject.toml`` is the single authority for maintained
workspace distributions. CI/release tooling should consume this helper instead
of duplicating package lists or hard-coding expected wheel counts.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_FILE = ROOT / "pyproject.toml"


def _fallback_members(text: str) -> tuple[str, ...]:
    """Parse the simple workspace-members array on Python 3.10.

    Python 3.11+ uses ``tomllib`` below. The fallback is deliberately narrow:
    it accepts the repository's quoted-string ``members = [...]`` contract and
    fails closed instead of pretending to be a general TOML parser.
    """

    section_match = re.search(
        r"(?ms)^\[tool\.uv\.workspace\]\s*(.*?)(?=^\[|\Z)",
        text,
    )
    if section_match is None:
        raise ValueError("missing [tool.uv.workspace] section")
    members_match = re.search(r"(?ms)^members\s*=\s*\[(.*?)\]", section_match.group(1))
    if members_match is None:
        raise ValueError("missing tool.uv.workspace.members array")
    members = tuple(re.findall(r'"([^"\n]+)"', members_match.group(1)))
    if not members:
        raise ValueError("tool.uv.workspace.members must not be empty")
    return members


def workspace_members(path: Path = WORKSPACE_FILE) -> tuple[str, ...]:
    text = path.read_text(encoding="utf-8")
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10
        members = _fallback_members(text)
    else:
        payload = tomllib.loads(text)
        try:
            raw_members = payload["tool"]["uv"]["workspace"]["members"]
        except (KeyError, TypeError) as exc:
            raise ValueError("missing tool.uv.workspace.members") from exc
        if not isinstance(raw_members, list) or not raw_members:
            raise ValueError("tool.uv.workspace.members must be a non-empty list")
        if not all(isinstance(item, str) and item for item in raw_members):
            raise ValueError("workspace members must be non-empty strings")
        members = tuple(raw_members)

    if len(set(members)) != len(members):
        raise ValueError("workspace members must be unique")

    for member in members:
        candidate = (ROOT / member).resolve()
        try:
            candidate.relative_to(ROOT.resolve())
        except ValueError as exc:
            raise ValueError(f"workspace member escapes repository root: {member}") from exc
        if candidate == ROOT or not candidate.is_dir():
            raise ValueError(f"workspace member directory does not exist: {member}")
        if not (candidate / "pyproject.toml").is_file():
            raise ValueError(f"workspace member has no pyproject.toml: {member}")

    return members


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    output = parser.add_mutually_exclusive_group()
    output.add_argument("--count", action="store_true", help="print only the validated member count")
    output.add_argument("--json", action="store_true", help="print the validated inventory as JSON")
    args = parser.parse_args()

    members = workspace_members()
    if args.count:
        print(len(members))
    elif args.json:
        print(json.dumps({"schema": "neuros.workspace.v1", "count": len(members), "members": members}, indent=2))
    else:
        print("\n".join(members))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
