#!/usr/bin/env python3
"""One-shot deterministic migration for GitHub Actions tag references."""

from __future__ import annotations

from pathlib import Path

WORKFLOWS = Path(".github/workflows")
REPLACEMENTS = {
    "actions/checkout@v4": "actions/checkout@11d5960a326750d5838078e36cf38b85af677262",
    "actions/setup-python@v5": "actions/setup-python@a26af69be951a213d495a4c3e4e4022e16d87065",
    "actions/upload-artifact@v4": "actions/upload-artifact@ea165f8d65b6e75b540449e92b4886f43607fa02",
    "actions/setup-dotnet@v4": "actions/setup-dotnet@67a3573c9a986a3f9c594539f4ab511d57bb3ce9",
}


def main() -> int:
    files_changed = 0
    replacements_made = 0
    for path in sorted([*WORKFLOWS.glob("*.yml"), *WORKFLOWS.glob("*.yaml")]):
        original = path.read_text(encoding="utf-8")
        updated = original
        for old, new in REPLACEMENTS.items():
            count = updated.count(old)
            replacements_made += count
            updated = updated.replace(old, new)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            files_changed += 1

    print(f"Pinned {replacements_made} mutable action references across {files_changed} workflow files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
