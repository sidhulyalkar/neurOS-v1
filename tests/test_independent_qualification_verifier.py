"""Adversarial tests for the independent public qualification verifier.

Tests construct a minimal sealed v1 fixture using only stdlib. They do not
depend on neurOS, NumPy, or the producer's bundle-verification helpers.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SOURCE = Path(__file__).resolve().parents[1] / "scripts" / "evidence" / "verify_public_qualification_bundle.py"
SPEC = importlib.util.spec_from_file_location("independent_public_verifier", SOURCE)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def make_fixture(root: Path) -> str:
    descriptor = {
        "stream_id": "eeg",
        "modality": "eeg",
        "sample_rate_hz": 250.0,
        "channel_names": ["Cz"],
        "channel_types": ["eeg"],
        "units": ["uV"],
        "device": None,
        "manufacturer": None,
        "clock_domain": "host_monotonic",
        "metadata": {},
    }
    fingerprint = sha_bytes(canonical({"schema": "neuros.stream_descriptor.v1", **descriptor}))
    descriptor["fingerprint_sha256"] = fingerprint
    stream_root = root / "session" / "streams" / "eeg"
    write_json(stream_root / "descriptor.json", descriptor)
    payload = b"synthetic-npy-payload"  # Hash integrity only; no NumPy parsing claim.
    data = stream_root / "frames" / "000000000000.npy"
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_bytes(payload)
    row = {
        "index": 0,
        "sequence_id": 0,
        "data": "frames/000000000000.npy",
        "data_sha256": sha_bytes(payload),
        "sample_rate_hz": 250.0,
        "host_receive_time_ns": 1,
        "device_time_ns": None,
        "synchronized_time_ns": None,
        "clock_domain": "host_monotonic",
        "quality": 0,
        "metadata": {},
    }
    (stream_root / "index.ndjson").write_text(json.dumps(row, sort_keys=True) + "\n", encoding="utf-8")
    write_json(
        root / "session" / "manifest.json",
        {
            "schema_version": 2,
            "status": "complete",
            "streams": {
                "eeg": {
                    "descriptor": descriptor,
                    "descriptor_fingerprint_sha256": fingerprint,
                    "frame_count": 1,
                }
            },
        },
    )
    (root / "config.yaml").write_text("schema_version: 1\n", encoding="utf-8")
    output = {"algorithm": "sha256-semantic-decoder-jsonl-v1", "count": 1, "sha256": "c" * 64}
    write_json(root / "decoder_outputs.json", {"record": output, "replay": output, "exact_match": True})
    q = {
        "runtime_state": "stopped",
        "runtime_failure": None,
        "node_failures": 0,
        "edge_items_dropped": 0,
    }
    write_json(root / "runtime.json", {"record_quality": q, "replay_quality": q})
    write_json(
        root / "manifest.json",
        {
            "schema_version": 1,
            "status": "complete",
            "qualification_scope": "runtime-record-replay",
            "archive": "session",
            "config_file_sha256": sha_bytes((root / "config.yaml").read_bytes()),
            "reproducibility": {
                "decoder_output_digest": output,
                "decoder_output_digest_exact": True,
            },
            "claim_boundary": {
                "runtime_record_replay_qualified": True,
                "real_dataset_qualified": False,
                "hardware_qualified": False,
                "closed_loop_qualified": False,
                "clinical_qualified": False,
            },
        },
    )
    return reindex(root)


def reindex(root: Path) -> str:
    rows = [
        {
            "path": item.relative_to(root).as_posix(),
            "bytes": item.stat().st_size,
            "sha256": sha_bytes(item.read_bytes()),
        }
        for item in sorted(root.rglob("*"))
        if item.is_file() and item.name != "artifact_hashes.json"
    ]
    digest = sha_bytes(canonical(rows))
    write_json(root / "artifact_hashes.json", {
        "algorithm": "sha256",
        "artifacts": rows,
        "bundle_sha256": digest,
    })
    return digest


class TestIndependentPublicVerifier(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "bundle"
        self.root.mkdir()
        self.expected = make_fixture(self.root)

    def test_verifies_pinned_bundle_without_reexecuting_decoder(self) -> None:
        report = verifier.verify(self.root, self.expected)
        self.assertEqual(report["status"], "verified")
        self.assertEqual(report["frame_count"], 1)
        self.assertEqual(report["root_sha256"], self.expected)
        self.assertTrue(report["externally_supplied_root_matched"])
        self.assertFalse(report["decoder_reexecuted"])
        self.assertFalse(report["source_authenticity_independently_established"])

    def test_rejects_ordinary_file_tampering(self) -> None:
        (self.root / "config.yaml").write_text("changed: true\n", encoding="utf-8")
        with self.assertRaisesRegex(verifier.VerificationError, "artifact byte or hash mismatch"):
            verifier.verify(self.root, self.expected)

    def test_rejects_resealed_bundle_with_stale_external_root(self) -> None:
        (self.root / "config.yaml").write_text("changed: true\n", encoding="utf-8")
        reindex(self.root)
        with self.assertRaises(verifier.VerificationError):
            verifier.verify(self.root, self.expected)

    def test_rejects_bad_inner_frame_hash_even_after_resealing(self) -> None:
        p = self.root / "session" / "streams" / "eeg" / "index.ndjson"
        row = json.loads(p.read_text(encoding="utf-8"))
        row["data_sha256"] = "b" * 64
        p.write_text(json.dumps(row) + "\n", encoding="utf-8")
        new_pin = reindex(self.root)
        with self.assertRaisesRegex(verifier.VerificationError, "inner frame checksum"):
            verifier.verify(self.root, new_pin)

    def test_rejects_wrong_descriptor_fingerprint_after_resealing(self) -> None:
        p = self.root / "session" / "streams" / "eeg" / "descriptor.json"
        descriptor = json.loads(p.read_text(encoding="utf-8"))
        descriptor["sample_rate_hz"] = 500.0
        write_json(p, descriptor)
        new_pin = reindex(self.root)
        with self.assertRaisesRegex(verifier.VerificationError, "descriptor fingerprint"):
            verifier.verify(self.root, new_pin)

    def test_rejects_claim_escalation_after_resealing(self) -> None:
        p = self.root / "manifest.json"
        manifest = json.loads(p.read_text(encoding="utf-8"))
        manifest["claim_boundary"]["hardware_qualified"] = True
        write_json(p, manifest)
        new_pin = reindex(self.root)
        with self.assertRaisesRegex(verifier.VerificationError, "unsupported qualification claim"):
            verifier.verify(self.root, new_pin)

    def test_rejects_decoder_digests_that_disagree_after_resealing(self) -> None:
        p = self.root / "decoder_outputs.json"
        evidence = json.loads(p.read_text(encoding="utf-8"))
        evidence["replay"]["sha256"] = "a" * 64
        write_json(p, evidence)
        new_pin = reindex(self.root)
        with self.assertRaisesRegex(verifier.VerificationError, "digest declaration mismatch"):
            verifier.verify(self.root, new_pin)

    def test_rejects_unsafe_inventory_paths(self) -> None:
        index_path = self.root / "artifact_hashes.json"
        index = json.loads(index_path.read_text(encoding="utf-8"))
        index["artifacts"][0]["path"] = "../outside"
        write_json(index_path, index)
        with self.assertRaisesRegex(verifier.VerificationError, "unsafe"):
            verifier.verify(self.root, self.expected)

    def test_rejects_duplicate_index_members(self) -> None:
        p = self.root / "artifact_hashes.json"
        index = json.loads(p.read_text(encoding="utf-8"))
        bad = '{"algorithm":"sha256","algorithm":"sha256","artifacts":' + json.dumps(index["artifacts"]) + ',"bundle_sha256":' + json.dumps(self.expected) + "}"
        p.write_text(bad, encoding="utf-8")
        with self.assertRaisesRegex(verifier.VerificationError, "duplicate JSON member"):
            verifier.verify(self.root, self.expected)

    def test_rejects_symlinked_bundle_file(self) -> None:
        p = self.root / "config.yaml"
        destination = Path(self.temp.name) / "elsewhere.yaml"
        destination.write_bytes(p.read_bytes())
        p.unlink()
        try:
            p.symlink_to(destination)
        except (OSError, NotImplementedError):
            self.skipTest("symlink creation unavailable")
        with self.assertRaisesRegex(verifier.VerificationError, "symbolic link"):
            verifier.verify(self.root, self.expected)

    def test_rejects_missing_or_malformed_pin(self) -> None:
        for pin in ("", "n/a", "A" * 64, "0" * 64):
            with self.subTest(pin=pin), self.assertRaises(verifier.VerificationError):
                verifier.verify(self.root, pin)


if __name__ == "__main__":
    unittest.main()
