#!/usr/bin/env python3
"""Fail closed when GitHub Actions dependencies are not content-addressed."""

from __future__ import annotations

import re
import sys
from pathlib import Path

WORKFLOWS = Path(".github/workflows")
USES_RE = re.compile(r"^\s*(?:-\s*)?uses:\s*['\"]?([^'\"\s#]+)")
GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
DOCKER_SHA_RE = re.compile(r"^docker://.+@sha256:[0-9a-f]{64}$")


def validate_reference(value: str) -> str | None:
    if value.startswith("./"):
        return None
    if value.startswith("docker://"):
        if DOCKER_SHA_RE.fullmatch(value):
            return None
        return "Docker action must be pinned by sha256 digest"
    if "@" not in value:
        return "remote action/workflow must include an @<commit-sha> ref"
    target, ref = value.rsplit("@", 1)
    if not target or target.startswith("/"):
        return "invalid remote action/workflow target"
    if not GIT_SHA_RE.fullmatch(ref):
        return "remote action/workflow must be pinned to a full 40-character lowercase commit SHA"
    return None


def main() -> int:
    violations: list[str] = []
    files = sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")])
    if not files:
        print("No workflow files found", file=sys.stderr)
        return 1

    refs = 0
    for path in files:
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            match = USES_RE.match(line)
            if not match:
                continue
            refs += 1
            value = match.group(1)
            error = validate_reference(value)
            if error:
                violations.append(f"{path}:{line_number}: {value}: {error}")

    if violations:
        print("Unpinned GitHub Actions supply-chain references detected:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print(f"GitHub Actions pin policy passed: {refs} action/workflow references are content-addressed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
