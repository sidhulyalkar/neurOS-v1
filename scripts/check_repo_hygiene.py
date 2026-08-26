#!/usr/bin/env python3
"""Fail when historical/generated/research artifacts leak into active surfaces."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from list_workspace_packages import workspace_members


EXACT_FORBIDDEN = {
    ".coverage",
    "setup.py",
    "CODEBASE_CLEANUP_COMPLETE.md",
    "CODEBASE_CLEANUP_PLAN.md",
    "DEVELOPMENT_SUMMARY.md",
    "PACKAGE_MIGRATION_VERIFICATION.md",
    "SESSION_SUMMARY.md",
    "scripts/cleanup_repo.py",
    "scripts/convert_imports.py",
    "chatGPT-eval2.pdf",
    "neuroFMx.txt",
    "docs/MODULARIZATION_PLAN.md",
    "docs/MODULARIZATION_STATUS.md",
    "docs/SESSION_SUMMARY_PHASE2.md",
    "docs/OPTIMIZATION.md",
}

ACTIVE_DOCS = (
    "README.md",
    "CONTRIBUTING.md",
    "ROADMAP.md",
    "mkdocs.yml",
    "docs/index.md",
    "docs/PROJECT_STATUS.md",
    "docs/ARCHITECTURE.md",
    "docs/API_REFERENCE.md",
    "docs/getting-started/installation.md",
    *(f"{member}/README.md" for member in workspace_members()),
)

STALE_MARKERS = (
    "github.com/yourusername/neuros-v1",
    "github.com/shulyalk/neuros-v1",
    "github.com/<your-user>/neuros2",
    "neuros.readthedocs.io",
    "pip install -e .",
)


def tracked_files() -> tuple[str, ...]:
    result = subprocess.run(
        ["git", "ls-files"],
        check=True,
        text=True,
        capture_output=True,
    )
    return tuple(line.strip() for line in result.stdout.splitlines() if line.strip())


def main() -> int:
    tracked = tracked_files()
    tracked_set = set(tracked)
    errors: list[str] = []

    for path in sorted(EXACT_FORBIDDEN & tracked_set):
        errors.append(f"forbidden active/history artifact is tracked: {path}")

    for path in tracked:
        if "/" not in path and re.fullmatch(r"SESSION_SUMMARY(?:_.*)?\.md", path):
            errors.append(f"root session note must live under docs/archive/session-notes: {path}")
        if path.startswith("notebooks/dino_"):
            errors.append(f"DINO research must live under experiments/vision/dinov3: {path}")
        if path.startswith("docs/") and not path.startswith("docs/archive/"):
            name = Path(path).name
            if name.startswith("SESSION_SUMMARY") or name in {
                "MODULARIZATION_PLAN.md",
                "MODULARIZATION_STATUS.md",
            }:
                errors.append(f"historical document is still active: {path}")

    required = {
        "docs/archive/README.md",
        "experiments/README.md",
        "experiments/vision/dinov3/README.md",
        "examples/README.md",
        "tutorials/README.md",
        "notebooks/README.md",
        "scripts/archive/README.md",
    }
    for path in sorted(required - tracked_set):
        errors.append(f"missing repository-boundary README: {path}")

    for path in ACTIVE_DOCS:
        file_path = Path(path)
        if not file_path.exists():
            errors.append(f"missing active documentation file: {path}")
            continue
        text = file_path.read_text(encoding="utf-8", errors="replace").lower()
        for marker in STALE_MARKERS:
            if marker.lower() in text:
                errors.append(f"stale installation/repository marker in {path}: {marker}")

    if errors:
        print("Repository hygiene violations:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(f"Repository hygiene passed for {len(tracked)} tracked paths.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
