"""Lightweight import validation for the maintained neuros-mechint surface.

This script is retained for developer convenience. The authoritative maintained
contract is exercised by `pytest` and `.github/workflows/neuros-mechint-ci.yml`.
Historical exploratory modules may require optional dependencies and are not
implicitly promoted by this validator.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class ImportCheck:
    module: str
    required: bool = True


CHECKS = (
    ImportCheck("neuros_mechint"),
    ImportCheck("neuros_mechint.core"),
    ImportCheck("neuros_mechint.adapters"),
    ImportCheck("neuros_mechint.benchmarks"),
    ImportCheck("neuros_mechint.benchmarks.evidence_pack"),
    ImportCheck("neuros_mechint.benchmarks.factorial"),
    ImportCheck("neuros_mechint.benchmarks.correspondence"),
    ImportCheck("neuros_mechint.benchmarks.replication"),
    ImportCheck("neuros_mechint.benchmarks.dose_response"),
    ImportCheck("neuros_mechint.integrations.factorial_study"),
    ImportCheck("neuros_mechint.integrations.correspondence"),
    ImportCheck("neuros_mechint.circuits"),
)


def main() -> int:
    failures = []
    for check in CHECKS:
        try:
            importlib.import_module(check.module)
        except Exception as exc:  # pragma: no cover - developer diagnostic
            label = "required" if check.required else "optional"
            print(f"FAIL [{label}] {check.module}: {exc}")
            if check.required:
                failures.append(check.module)
        else:
            print(f"PASS {check.module}")

    try:
        import neuros_mechint

        if neuros_mechint.__version__ != "0.9.0":
            print(
                "FAIL version: expected 0.9.0, "
                f"observed {neuros_mechint.__version__}"
            )
            failures.append("version")
        else:
            print("PASS version 0.9.0")
    except Exception as exc:  # pragma: no cover - diagnostic after import failure
        print(f"FAIL version check: {exc}")
        failures.append("version")

    if failures:
        print("\nMaintained import validation failed:", ", ".join(failures))
        return 1

    print("\nMaintained v0.9 import surface is available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
