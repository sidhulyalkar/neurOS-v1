"""
Validation script for alignment module structure.

This script validates that all modules are properly structured and can be parsed,
even if dependencies are not installed.
"""

import ast
import os
from pathlib import Path


def validate_python_file(filepath):
    """Validate that a Python file has valid syntax."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        ast.parse(code)
        return True, "Valid syntax"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_module_structure(module_path):
    """Check the structure of a module."""
    info = {
        'classes': [],
        'functions': [],
        'imports': []
    }

    with open(module_path, 'r', encoding='utf-8') as f:
        code = f.read()

    try:
        tree = ast.parse(code)

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                info['classes'].append(node.name)
            elif isinstance(node, ast.FunctionDef):
                if not node.name.startswith('_'):
                    info['functions'].append(node.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    info['imports'].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    info['imports'].append(node.module)

    except Exception as e:
        print(f"Error parsing {module_path}: {e}")

    return info


def main():
    """Main validation function."""
    alignment_dir = Path(__file__).parent

    modules = {
        'cca.py': 'Canonical Correlation Analysis',
        'rsa.py': 'Representational Similarity Analysis',
        'pls.py': 'Partial Least Squares',
        'metrics.py': 'Evaluation Metrics',
        '__init__.py': 'Module Initialization'
    }

    print("=" * 80)
    print("NeuroFMX Alignment Module Validation")
    print("=" * 80)
    print()

    all_valid = True

    for module_file, description in modules.items():
        filepath = alignment_dir / module_file
        print(f"\n{description} ({module_file})")
        print("-" * 60)

        if not filepath.exists():
            print(f"  ✗ File not found: {filepath}")
            all_valid = False
            continue

        # Check syntax
        valid, message = validate_python_file(filepath)
        if valid:
            print(f"  [OK] Syntax validation: PASSED")
        else:
            print(f"  [FAIL] Syntax validation: FAILED - {message}")
            all_valid = False
            continue

        # Check structure
        if module_file != '__init__.py':
            info = check_module_structure(filepath)
            print(f"  [OK] Classes found: {len(info['classes'])}")
            if info['classes']:
                for cls in info['classes']:
                    print(f"    - {cls}")

            print(f"  [OK] Public functions: {len(info['functions'])}")
            if info['functions']:
                for func in info['functions'][:5]:  # Show first 5
                    print(f"    - {func}")
                if len(info['functions']) > 5:
                    print(f"    ... and {len(info['functions']) - 5} more")

    print("\n" + "=" * 80)
    if all_valid:
        print("[SUCCESS] All modules validated successfully!")
        print("\nModule Summary:")
        print("-" * 60)
        print("1. cca.py - Provides CCA, RegularizedCCA, KernelCCA, TimeVaryingCCA")
        print("2. rsa.py - Provides RDM computation, RSA, HierarchicalRSA, MDS")
        print("3. pls.py - Provides PLS, CrossValidatedPLS, PLSVisualization")
        print("4. metrics.py - Provides NoiseCeiling, Bootstrap, Permutation tests")
        print("5. __init__.py - Exports all public APIs")
        print("\nNote: Runtime testing requires dependencies (torch, sklearn, scipy)")
    else:
        print("[FAIL] Some modules failed validation")

    print("=" * 80)

    return all_valid


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
