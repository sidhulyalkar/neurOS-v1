#!/usr/bin/env python3
"""Enforce the stable neurOS dependency direction.

The kernel is a contract/runtime substrate. Third-party ecosystems and concrete
neurOS implementations must depend on the kernel, never the reverse. This
check is intentionally dependency-free so it can run before workspace install.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CORE_SOURCE = ROOT / "packages" / "neuros-core" / "src"
CORE_PYPROJECT = ROOT / "packages" / "neuros-core" / "pyproject.toml"

# These namespaces belong to concrete integrations, research, presentation, or
# composition layers. Importing them from neuros-core reverses the architecture.
FORBIDDEN_CORE_IMPORTS = (
    "neuros.drivers",
    "neuros.models",
    "neuros.ui",
    "neuros.cloud",
    "neuros.foundation_models",
    "neuros.sourceweigher",
    "neuros.mechint",
    "neuros.neurofm",
    "neuros.interop",
    "orion",
    "mne",
    "brainflow",
    "pylsl",
    "braindecode",
    "moabb",
    "spikeinterface",
    "dandi",
    "pynapple",
    "py_neuromodulation",
)


def _matches_forbidden(module: str) -> str | None:
    for prefix in FORBIDDEN_CORE_IMPORTS:
        if module == prefix or module.startswith(f"{prefix}."):
            return prefix
    return None


def _imports(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            # Relative imports remain inside the package and are not external
            # dependency statements until resolved. Absolute imports are enough
            # to enforce the layer boundary here.
            if node.level == 0 and node.module:
                found.append((node.lineno, node.module))
    return found


def _check_python_imports() -> list[str]:
    errors: list[str] = []
    for path in sorted(CORE_SOURCE.rglob("*.py")):
        for line, module in _imports(path):
            forbidden = _matches_forbidden(module)
            if forbidden is not None:
                rel = path.relative_to(ROOT)
                errors.append(
                    f"{rel}:{line} imports {module!r}; neuros-core may not depend on {forbidden!r}"
                )
    return errors


def _check_core_metadata() -> list[str]:
    text = CORE_PYPROJECT.read_text(encoding="utf-8").lower()
    errors: list[str] = []
    # neuros-core is allowed to declare storage/evaluation extras such as
    # pynwb/zarr/scikit-learn because those implement kernel-owned persistence
    # and generic evaluation contracts. It may not depend on sibling neurOS
    # distributions or external runtime/model ecosystems.
    forbidden_distribution_markers = (
        "neuros-drivers",
        "neuros-models",
        "neuros-foundation",
        "neuros-sourceweigher",
        "neuros-mechint",
        "neuros-neurofm",
        "neuros-ui",
        "neuros-cloud",
        "neuros-orion",
        "brainflow",
        "pylsl",
        "mne>=",
        "braindecode",
        "moabb",
        "spikeinterface",
        "dandi",
        "py-neuromodulation",
    )
    for marker in forbidden_distribution_markers:
        if marker in text:
            errors.append(
                f"packages/neuros-core/pyproject.toml contains forbidden dependency marker {marker!r}"
            )
    return errors


def main() -> int:
    if not CORE_SOURCE.exists() or not CORE_PYPROJECT.exists():
        print("dependency-boundary check must run from a complete neurOS checkout", file=sys.stderr)
        return 2

    errors = _check_python_imports() + _check_core_metadata()
    if errors:
        print("Dependency-boundary violations:")
        for error in errors:
            print(f"  - {error}")
        return 1

    checked = len(tuple(CORE_SOURCE.rglob("*.py")))
    print(
        f"Dependency boundaries passed for {checked} neuros-core Python files; "
        "kernel remains independent of concrete ecosystems."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
