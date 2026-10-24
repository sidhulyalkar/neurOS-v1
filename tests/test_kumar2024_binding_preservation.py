"""Score-free tests of archival verification and write-once preservation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


MODULE = Path(__file__).resolve().parents[1] / "scripts" / "evidence" / "preserve_kumar2024_binding.py"
SPEC = importlib.util.spec_from_file_location("preserve_original_binding", MODULE)
assert SPEC is not None and SPEC.loader is not None
preserver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preserver)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_zip(path: Path, *, models_executed: bool = False) -> tuple[str, str]:
    manifest = {
        "source_revision": preserver.SOURCE_REVISION,
        "environment_authority_sha256": preserver.ENVIRONMENT_SHA256,
        "execution_plan_sha256": preserver.EXECUTION_PLAN_SHA256,
        "counts": preserver.COUNTS,
        "model_execution_performed": models_executed,
        "final_assessment_predictions_generated": False,
        "final_assessment_metrics_generated": False,
    }
    payloads = {
        "binding_manifest.json": json.dumps(manifest, sort_keys=True).encode(),
        "case_authorities.json": b"{}",
        "environment_lock.json": b"{}",
        "execution_plan.json": b"{}",
        "materialization.json": b"{}",
        "method_specs.json": b"{}",
    }
    files = {name: sha(blob) for name, blob in payloads.items()}
    bundle_root = sha(json.dumps({
        "schema": preserver.BUNDLE_SCHEMA, "payload": {"files": files},
    }, sort_keys=True, separators=(",", ":")).encode())
    index = {
        "schema_version": 1,
        "files": files,
        "bundle_sha256": bundle_root,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)
        archive.writestr("artifact_hashes.json", json.dumps(index))
    return sha(path.read_bytes()), bundle_root


class BindingPreservationTest(unittest.TestCase):
    def setUp(self) -> None:
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.source = self.root / "original.zip"
        zip_sha, bundle_root = make_zip(self.source)
        original_archive_sha = preserver.ARCHIVE_SHA256
        original_bundle_sha = preserver.BUNDLE_SHA256
        self.patch_archive = patch.object(preserver, "ARCHIVE_SHA256", zip_sha)
        self.patch_bundle = patch.object(preserver, "BUNDLE_SHA256", bundle_root)
        self.patch_archive.start()
        self.patch_bundle.start()
        self.addCleanup(self.patch_archive.stop)
        self.addCleanup(self.patch_bundle.stop)
        self.assertNotEqual(zip_sha, original_archive_sha)
        self.assertNotEqual(bundle_root, original_bundle_sha)

    def test_verification_has_no_model_result_claim(self) -> None:
        result = preserver.verify_archive(self.source)
        self.assertTrue(result["verified"])
        self.assertEqual(result["verified_managed_files"], 6)
        self.assertFalse(result["scientific_results_interpretable"])
        self.assertFalse(result["model_execution_performed"])

    def test_published_archive_is_byte_identical_and_write_once(self) -> None:
        dest = self.root / "durable.zip"
        result = preserver.preserve(self.source, dest)
        self.assertTrue(result["write_once"])
        self.assertEqual(self.source.read_bytes(), dest.read_bytes())
        with self.assertRaisesRegex(preserver.BindingArchiveError, "already exists"):
            preserver.preserve(self.source, dest)
        self.assertEqual(self.source.read_bytes(), dest.read_bytes())

    def test_verification_of_publication_detects_post_copy_corruption(self) -> None:
        destination = self.root / "verified.zip"
        real_verify = preserver.verify_archive

        def swapped_verify(path):
            if path == destination:
                raise preserver.BindingArchiveError("published link verification failed")
            return real_verify(path)

        with patch.object(preserver, "verify_archive", side_effect=swapped_verify):
            with self.assertRaisesRegex(
                preserver.BindingArchiveError, "published link verification failed"
            ):
                preserver.preserve(self.source, destination)

    def test_existing_destination_is_not_replaced(self) -> None:
        dest = self.root / "durable.zip"
        dest.write_bytes(b"independent-existing-evidence")
        with self.assertRaises(preserver.BindingArchiveError):
            preserver.preserve(self.source, dest)
        self.assertEqual(dest.read_bytes(), b"independent-existing-evidence")

    def test_refuses_wrong_source_sha_without_output(self) -> None:
        dest = self.root / "durable.zip"
        with self.source.open("ab") as stream:
            stream.write(b"unexpected")
        with self.assertRaisesRegex(preserver.BindingArchiveError, "ZIP SHA-256 mismatch"):
            preserver.preserve(self.source, dest)
        self.assertFalse(dest.exists())

    def test_refuses_executed_model_claim_even_when_hashes_match(self) -> None:
        changed_zip, changed_root = make_zip(self.source, models_executed=True)
        with patch.object(preserver, "ARCHIVE_SHA256", changed_zip):
            with patch.object(preserver, "BUNDLE_SHA256", changed_root):
                with self.assertRaisesRegex(preserver.BindingArchiveError, "model_execution_performed"):
                    preserver.verify_archive(self.source)

    def test_refuses_mismatch_inside_sealed_managed_file(self) -> None:
        # Repackage the ZIP with a changed file but retain the original index.
        bad = self.root / "tampered.zip"
        with zipfile.ZipFile(self.source, "r") as original:
            contents = {name: original.read(name) for name in original.namelist()}
        contents["method_specs.json"] = b'{"changed":true}'
        with zipfile.ZipFile(bad, "w") as archive:
            for name, value in contents.items():
                archive.writestr(name, value)
        with patch.object(preserver, "ARCHIVE_SHA256", sha(bad.read_bytes())):
            with self.assertRaisesRegex(preserver.BindingArchiveError, "member SHA-256"):
                preserver.verify_archive(bad)

    def test_refuses_extra_zip_member_even_with_pinned_zip(self) -> None:
        with zipfile.ZipFile(self.source, "a") as archive:
            archive.writestr("unexpected.txt", b"surprise")
        with patch.object(preserver, "ARCHIVE_SHA256", sha(self.source.read_bytes())):
            with self.assertRaisesRegex(preserver.BindingArchiveError, "inventory"):
                preserver.verify_archive(self.source)

    def test_refuses_symlink_as_input(self) -> None:
        alias = self.root / "alias.zip"
        try:
            alias.symlink_to(self.source)
        except OSError:
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(preserver.BindingArchiveError, "regular ZIP file"):
            preserver.verify_archive(alias)

    def test_refuses_same_source_destination(self) -> None:
        with self.assertRaisesRegex(preserver.BindingArchiveError, "must differ"):
            preserver.preserve(self.source, self.source)

    def test_refuses_new_directory_implicitly_created(self) -> None:
        with self.assertRaisesRegex(preserver.BindingArchiveError, "parent must already exist"):
            preserver.preserve(self.source, self.root / "absent" / "archive.zip")


if __name__ == "__main__":
    unittest.main()
