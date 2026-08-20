#!/usr/bin/env python3
"""Install neurOS workspace packages in dependency order.

This is the supported development bootstrap path for contributors who do not use
``uv``. Package metadata remains owned by each package-level ``pyproject.toml``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

PROFILES = {
    "kernel": ["neuros-core"],
    "bci": ["neuros-core", "neuros-drivers", "neuros-models", "neuros"],
    "orion": ["neuros-core", "orion"],
    "research": [
        "neuros-core",
        "neuros-models",
        "neuros-foundation",
        "neuros-sourceweigher",
        "neuros-mechint",
        "neuros-neurofm",
        "orion",
    ],
    "all": [
        "neuros-core",
        "neuros-drivers",
        "neuros-models",
        "neuros-foundation",
        "neuros-ui",
        "neuros-cloud",
        "neuros-mechint",
        "neuros-neurofm",
        "neuros-sourceweigher",
        "orion",
        "neuros",
    ],
}


def package_path(name: str) -> Path:
    path = ROOT / "packages" / name
    if not path.is_dir():
        raise FileNotFoundError(f"Workspace package not found: {path}")
    return path


def install_package(name: str, *, dry_run: bool) -> None:
    path = package_path(name)
    command = [sys.executable, "-m", "pip", "install", "-e", str(path)]
    print("+", " ".join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=sorted(PROFILES), default="bci")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--test-tools",
        action="store_true",
        help="Install shared pytest/coverage tools after the selected profile",
    )
    args = parser.parse_args()

    for package in PROFILES[args.profile]:
        install_package(package, dry_run=args.dry_run)

    if args.test_tools:
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "pytest>=7.4",
            "pytest-asyncio>=0.21",
            "pytest-cov>=4.1",
        ]
        print("+", " ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
