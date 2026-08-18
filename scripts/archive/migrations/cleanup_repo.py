#!/usr/bin/env python3
"""
Repository cleanup script for neurOS.

Removes unnecessary progress/temporary files while keeping essential documentation.
"""

import os
import shutil
from pathlib import Path


def cleanup_repository(dry_run=True):
    """
    Clean up repository by removing unnecessary files.

    Keeps:
    - Essential docs (README, CONTRIBUTING, ROADMAP)
    - Package documentation
    - Tutorials
    - Core code and tests

    Removes:
    - Progress/planning docs (now consolidated)
    - Temporary files
    - Old evaluation files
    - Redundant documentation
    """

    project_root = Path(__file__).parent.parent

    # Files to remove (root level)
    files_to_remove = [
        'CAPABILITIES.md',  # Consolidated into docs
        'WEEK1_SPRINT_PLAN.md',  # Completed, now in docs
        'QUICKSTART.md',  # Redundant with README
        'AUDIT.md',  # Old planning doc
        'PRIORITIES_FROM_CHATGPT_EVAL.md',  # Consolidated into MODULARIZATION_PLAN
        'chatgpt-eval.txt',  # Old format, have PDF now
        'coverage_report.txt',  # Temporary file
    ]

    # Docs to consolidate/remove
    docs_to_remove = [
        'docs/white_paper.md',  # Redundant versions
        'docs/white_paper2.md',
        'docs/DINOv3_paper.md',  # Redundant
        'docs/dinov3_neuroscience_full_paper.md',  # Redundant
        'docs/dinov3_neuroscience_extended_paper.md',  # Keep only one version
        'docs/runbook_foundation_models_demo.md',  # Old planning doc
    ]

    # Keep these essential docs
    essential_docs = [
        'README.md',
        'CONTRIBUTING.md',
        'ROADMAP.md',
        'docs/MODULARIZATION_PLAN.md',
        'docs/MODULARIZATION_STATUS.md',
        'docs/SESSION_SUMMARY_PHASE2.md',
        'docs/getting-started/',
        'docs/index.md',
        'chatGPT-eval2.pdf',  # Latest evaluation
    ]

    removed_count = 0

    print("=" * 70)
    print(f"Repository Cleanup ({'DRY RUN' if dry_run else 'EXECUTING'})")
    print("=" * 70)

    # Remove root-level files
    print("\n📄 Removing unnecessary root files:")
    for filename in files_to_remove:
        filepath = project_root / filename
        if filepath.exists():
            print(f"  {'[DRY RUN] ' if dry_run else ''}Removing: {filename}")
            if not dry_run:
                filepath.unlink()
            removed_count += 1
        else:
            print(f"  [SKIP] Not found: {filename}")

    # Remove redundant docs
    print("\n📚 Removing redundant documentation:")
    for filepath in docs_to_remove:
        full_path = project_root / filepath
        if full_path.exists():
            print(f"  {'[DRY RUN] ' if dry_run else ''}Removing: {filepath}")
            if not dry_run:
                if full_path.is_dir():
                    shutil.rmtree(full_path)
                else:
                    full_path.unlink()
            removed_count += 1
        else:
            print(f"  [SKIP] Not found: {filepath}")

    # Clean up Python cache
    print("\n🧹 Cleaning Python cache files:")
    cache_patterns = ['__pycache__', '*.pyc', '*.pyo', '.pytest_cache', '.coverage']
    cache_removed = 0

    for pattern in cache_patterns:
        if pattern.startswith('.'):
            # Hidden files/dirs in root
            cache_path = project_root / pattern
            if cache_path.exists():
                print(f"  {'[DRY RUN] ' if dry_run else ''}Removing: {pattern}")
                if not dry_run:
                    if cache_path.is_dir():
                        shutil.rmtree(cache_path)
                    else:
                        cache_path.unlink()
                cache_removed += 1
        else:
            # Find all matching patterns
            for item in project_root.rglob(pattern):
                if '.git' not in str(item) and '.venv' not in str(item):
                    rel_path = item.relative_to(project_root)
                    if not dry_run:
                        if item.is_dir():
                            shutil.rmtree(item)
                        else:
                            item.unlink()
                    cache_removed += 1

    if cache_removed > 0:
        print(f"  {'[DRY RUN] ' if dry_run else ''}Removed {cache_removed} cache files/dirs")

    # Report
    print("\n" + "=" * 70)
    print(f"Cleanup Summary ({'DRY RUN' if dry_run else 'COMPLETE'})")
    print("=" * 70)
    print(f"Files removed: {removed_count}")
    print(f"Cache cleaned: {cache_removed}")
    print(f"\n✓ Repository is now professional and clean!")

    if dry_run:
        print("\nRun with dry_run=False to apply changes.")

    return removed_count


if __name__ == "__main__":
    import sys
    dry_run = '--dry-run' in sys.argv
    cleanup_repository(dry_run=dry_run)
