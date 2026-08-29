#!/usr/bin/env python3
"""Fail-closed validator for the promoted Kumar2024 Python environment.

The promoted binding and worker workflows share this executable contract so the
scheduler cannot quietly use a looser dependency check than the binding job.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
from pathlib import Path
from typing import Iterable

_NORMALIZE_RE = re.compile(r"[-_.]+")

LOCAL_DISTRIBUTIONS = frozenset(
    {
        "neuros",
        "neuros-core",
        "neuros-drivers",
        "neuros-foundation",
        "neuros-models",
        "neuros-orion",
    }
)

REQUIRED_EXTERNAL_DISTRIBUTIONS = frozenset(
    {
        "braindecode",
        "mne",
        "moabb",
        "numpy",
        "pip",
        "pyriemann",
        "scikit-learn",
        "scipy",
        "setuptools",
        "torch",
        "wheel",
    }
)


def _normalize(value: str) -> str:
    return _NORMALIZE_RE.sub("-", value.strip()).lower()


def load_exact_constraints(path: str | Path) -> dict[str, str]:
    """Load an exact ``name==version`` constraint frontier."""

    result: dict[str, str] = {}
    for line_number, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.count("==") != 1:
            raise ValueError(
                f"constraint line {line_number} must be exactly name==version: {line!r}"
            )
        raw_name, raw_version = line.split("==", 1)
        name = _normalize(raw_name)
        version = raw_version.strip()
        if not name or not version:
            raise ValueError(f"constraint line {line_number} is incomplete")
        previous = result.get(name)
        if previous is not None and previous != version:
            raise ValueError(
                f"constraint frontier contains conflicting versions for {name!r}: "
                f"{previous!r} versus {version!r}"
            )
        result[name] = version
    if not result:
        raise ValueError("promoted constraint frontier cannot be empty")
    return result


def observed_distributions() -> dict[str, str]:
    """Return the canonical realized Python distribution map."""

    observed: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        raw_name = distribution.metadata.get("Name")
        if raw_name is None:
            continue
        name = _normalize(raw_name)
        version = str(distribution.version).strip()
        previous = observed.get(name)
        if previous is not None and previous != version:
            raise RuntimeError(
                f"environment exposes conflicting installed versions for {name!r}: "
                f"{previous!r} versus {version!r}"
            )
        observed[name] = version
    return observed


def validate_promoted_environment(
    constraints: dict[str, str],
    *,
    local_distributions: Iterable[str] = LOCAL_DISTRIBUTIONS,
    required_external: Iterable[str] = REQUIRED_EXTERNAL_DISTRIBUTIONS,
) -> dict[str, object]:
    """Require every realized external distribution to belong to the exact frontier."""

    local = {_normalize(name) for name in local_distributions}
    required = {_normalize(name) for name in required_external}
    observed = observed_distributions()
    external = {name: version for name, version in observed.items() if name not in local}

    unexpected = sorted(set(external) - set(constraints))
    mismatched = {
        name: {"expected": constraints[name], "observed": external[name]}
        for name in sorted(set(external) & set(constraints))
        if external[name] != constraints[name]
    }
    missing_required = sorted(required - set(external))
    if unexpected or mismatched or missing_required:
        raise RuntimeError(
            "promoted external dependency frontier differs: "
            f"unexpected={unexpected}, mismatched={mismatched}, "
            f"missing_required={missing_required}"
        )
    return {
        "verified": True,
        "constraint_count": len(constraints),
        "installed_external_count": len(external),
        "installed_local": sorted(set(observed) & local),
        "required_external": sorted(required),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate the realized promoted Kumar2024 Python environment."
    )
    parser.add_argument("--constraints", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    result = validate_promoted_environment(load_exact_constraints(args.constraints))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
