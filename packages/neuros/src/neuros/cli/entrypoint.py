"""Stable public CLI entrypoint with a narrow developer on-ramp."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Sequence

from neuros.errors import ConfigurationError

from .project_commands import SUPPORTED_PROJECT_TEMPLATES, init_project


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuros init",
        description="Create a minimal runnable neurOS project",
    )
    parser.add_argument(
        "destination",
        nargs="?",
        default="neuros-project",
        help="Project directory to create (default: neuros-project)",
    )
    parser.add_argument(
        "--template",
        choices=SUPPORTED_PROJECT_TEMPLATES,
        default="mock-bci",
        help="Starter project template",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace neurOS-managed starter files but preserve unrelated files",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    return parser


def _run_init(argv: Sequence[str]) -> int:
    args = _init_parser().parse_args(list(argv))
    try:
        result = init_project(
            args.destination,
            template=args.template,
            force=args.force,
        )
    except ConfigurationError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0

    print(f"Created neurOS project: {result['project_root']}")
    print(f"Template: {result['template']}")
    print("Next:")
    print(f"  cd {result['project_root']}")
    for command in result["next_commands"]:
        print(f"  {command}")
    print()
    print(f"Evidence boundary: {result['evidence_boundary']}")
    return 0


def main() -> None:
    """Dispatch the new-user on-ramp, then delegate every existing command."""

    if len(sys.argv) > 1 and sys.argv[1] == "init":
        raise SystemExit(_run_init(sys.argv[2:]))

    from .app import main as runtime_main

    runtime_main()


__all__ = ["main"]
