"""Test import-path support for repository-local sibling helper modules.

Pytest imports explicit test paths from the repository root, which does not
otherwise guarantee that ``tests/`` itself is importable. Keep this adjustment
strictly in the test process so split adversarial suites can share local helpers
without placing test fixtures in production packages.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TEST_ROOT = str(Path(__file__).resolve().parent)
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)
