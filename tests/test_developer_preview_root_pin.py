"""Score-free unit tests for the installed-wheel journey's root-pin boundary."""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_developer_preview_journey.py"
SPEC = importlib.util.spec_from_file_location("neuros_developer_preview_journey", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
journey = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = journey
SPEC.loader.exec_module(journey)

ROOT = "a" * 64


class QualificationRootTests(unittest.TestCase):
    def test_accepts_complete_qualification_root(self) -> None:
        self.assertEqual(
            journey._qualification_root({"status": "complete", "bundle_sha256": ROOT}),
            ROOT,
        )

    def test_rejects_missing_incomplete_or_malformed_roots(self) -> None:
        bad_results = (
            None,
            {},
            {"status": "failed", "bundle_sha256": ROOT},
            {"status": "complete"},
            {"status": "complete", "bundle_sha256": "abc"},
            {"status": "complete", "bundle_sha256": "A" * 64},
            {"status": "complete", "bundle_sha256": 7},
        )
        for result in bad_results:
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                journey._qualification_root(result)

    def test_accepts_explicitly_pinned_verified_reproduction(self) -> None:
        journey._require_pinned_reproduction(
            {
                "reproduced": True,
                "bundle_sha256": ROOT,
                "origin_authenticity": "externally_pinned",
            },
            ROOT,
        )

    def test_rejects_self_consistency_without_an_external_pin(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "externally pinned"):
            journey._require_pinned_reproduction(
                {"reproduced": True, "bundle_sha256": ROOT, "origin_authenticity": "not_established"},
                ROOT,
            )

    def test_rejects_mismatched_root_or_unsuccessful_replay(self) -> None:
        bad_results = (
            None,
            {},
            {"reproduced": False, "bundle_sha256": ROOT, "origin_authenticity": "externally_pinned"},
            {"reproduced": True, "bundle_sha256": "b" * 64, "origin_authenticity": "externally_pinned"},
        )
        for result in bad_results:
            with self.subTest(result=result), self.assertRaises(RuntimeError):
                journey._require_pinned_reproduction(result, ROOT)


if __name__ == "__main__":
    unittest.main()
