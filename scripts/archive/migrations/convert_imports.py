#!/usr/bin/env python3
"""
Convert relative imports to absolute imports in neuros packages.

This script systematically converts all relative imports (e.g., `from .models import X`)
to absolute imports (e.g., `from neuros.models import X`) to support the modular
package structure.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple


def detect_module_path(file_path: Path, packages_root: Path) -> str:
    """
    Determine the module path for a file.

    Example: packages/neuros-core/src/neuros/agents/base_agent.py
             -> neuros.agents
    """
    # Get path relative to src/neuros
    try:
        # Find the 'neuros' directory in the path
        parts = file_path.parts
        neuros_idx = parts.index('neuros')
        # Module path is everything after 'neuros' up to the filename
        module_parts = parts[neuros_idx:-1]
        return '.'.join(module_parts)
    except (ValueError, IndexError):
        return ""


def convert_relative_import(line: str, current_module: str) -> str:
    """
    Convert a relative import to an absolute import.

    Examples:
        from .models import X              -> from neuros.models import X
        from ..drivers.base import Y       -> from neuros.drivers.base import Y
        from .submodule.thing import Z     -> from neuros.{current}.submodule.thing import Z
    """
    # Pattern: from .something import ...
    # Pattern: from ..something import ...
    pattern = r'^(\s*from\s+)(\.+)([a-zA-Z_][a-zA-Z0-9_\.]*)?(\s+import\s+.+)$'
    match = re.match(pattern, line)

    if not match:
        return line

    prefix, dots, relative_path, import_clause = match.groups()

    # Count leading dots
    level = len(dots)

    # Build absolute path
    if level == 1:
        # from .module -> from neuros.current_module.module
        if relative_path:
            if current_module:
                absolute = f"{current_module}.{relative_path}"
            else:
                absolute = relative_path
        else:
            # from . import something
            absolute = current_module if current_module else "neuros"

    elif level == 2:
        # from ..module -> go up one level
        if current_module and '.' in current_module:
            parent = '.'.join(current_module.split('.')[:-1])
            if relative_path:
                absolute = f"{parent}.{relative_path}" if parent else relative_path
            else:
                absolute = parent if parent else "neuros"
        else:
            # Already at top level
            absolute = relative_path if relative_path else "neuros"

    else:
        # More than 2 levels, just convert to neuros.module
        absolute = relative_path if relative_path else "neuros"

    return f"{prefix}{absolute}{import_clause}"


def convert_file_imports(file_path: Path, packages_root: Path, dry_run: bool = False) -> Tuple[int, List[str]]:
    """
    Convert all relative imports in a file to absolute imports.

    Returns:
        (num_conversions, list_of_changes)
    """
    # Determine current module path
    current_module = detect_module_path(file_path, packages_root)

    # current_module already includes 'neuros' prefix, which is what we want

    # Read file
    try:
        content = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return 0, []

    lines = content.splitlines(keepends=True)
    new_lines = []
    conversions = 0
    changes = []

    for line_num, line in enumerate(lines, 1):
        new_line = convert_relative_import(line.rstrip('\n'), current_module)

        if new_line != line.rstrip('\n'):
            conversions += 1
            changes.append(f"  Line {line_num}: {line.rstrip()} -> {new_line}")
            new_lines.append(new_line + '\n')
        else:
            new_lines.append(line)

    # Write back if not dry run
    if not dry_run and conversions > 0:
        try:
            file_path.write_text(''.join(new_lines), encoding='utf-8')
        except Exception as e:
            print(f"Error writing {file_path}: {e}")
            return 0, []

    return conversions, changes


def process_package(package_path: Path, dry_run: bool = False) -> int:
    """
    Process all Python files in a package.

    Returns:
        Total number of conversions
    """
    python_files = list(package_path.rglob("*.py"))
    total_conversions = 0

    print(f"\n{'DRY RUN: ' if dry_run else ''}Processing {package_path.name}...")
    print(f"  Found {len(python_files)} Python files")

    for py_file in python_files:
        conversions, changes = convert_file_imports(py_file, package_path, dry_run)

        if conversions > 0:
            print(f"\n  {py_file.relative_to(package_path)}: {conversions} conversions")
            for change in changes[:5]:  # Show first 5 changes
                print(change)
            if len(changes) > 5:
                print(f"    ... and {len(changes) - 5} more")

        total_conversions += conversions

    return total_conversions


def main():
    # Get packages directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    packages_dir = project_root / "packages"

    if not packages_dir.exists():
        print(f"Error: {packages_dir} not found")
        sys.exit(1)

    # Check for --dry-run flag
    dry_run = "--dry-run" in sys.argv

    if dry_run:
        print("=" * 70)
        print("DRY RUN MODE - No files will be modified")
        print("=" * 70)

    # Process each package
    packages = [
        "neuros-core",
        "neuros-drivers",
        "neuros-models",
        "neuros-foundation",
        "neuros-ui",
        "neuros-cloud",
    ]

    total = 0
    for package_name in packages:
        package_path = packages_dir / package_name
        if package_path.exists():
            conversions = process_package(package_path, dry_run)
            total += conversions
            print(f"  ✓ {package_name}: {conversions} conversions")
        else:
            print(f"  ⚠ {package_name}: not found")

    print("\n" + "=" * 70)
    print(f"{'DRY RUN: ' if dry_run else ''}Total conversions: {total}")
    print("=" * 70)

    if dry_run:
        print("\nRun without --dry-run to apply changes")
    else:
        print("\n✓ Import conversion complete!")


if __name__ == "__main__":
    main()
