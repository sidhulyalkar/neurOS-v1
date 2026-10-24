#!/usr/bin/env python3
"""Independent, dependency-free structural verifier for neurOS qualification v1.

No imports from neurOS or the producing verifier. This verifies a pinned artifact
root, sealed files, archive frame hashes, descriptors, and recorded metadata
consistency. It does NOT replay the decoder or authenticate the pin's publisher.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import re
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any


SHA256 = re.compile(r"[0-9a-f]{64}\Z")
INDEX = "artifact_hashes.json"


class VerificationError(ValueError):
    """An untrusted qualification artifact failed verification."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VerificationError(message)


def _sha(value: Any, label: str) -> str:
    _require(isinstance(value, str) and SHA256.fullmatch(value) is not None, f"{label}: invalid SHA-256")
    return value


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        _require(key not in value, f"duplicate JSON member: {key}")
        value[key] = item
    return value


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_pairs)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _digest(path: Path) -> tuple[int, str]:
    h = hashlib.sha256()
    length = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            length += len(block)
            h.update(block)
    return length, h.hexdigest()


def _relative(text: Any) -> str:
    _require(isinstance(text, str) and bool(text), "file path must be a nonempty string")
    path = PurePosixPath(text)
    _require(
        not path.is_absolute() and path.as_posix() == text
        and all(part not in ("", ".", "..") for part in text.split("/"))
        and "\\" not in text and "\x00" not in text,
        f"unsafe or noncanonical artifact path: {text!r}",
    )
    return text


def _inventory(root: Path) -> dict[str, Path]:
    _require(root.is_dir() and not root.is_symlink(), "bundle root must be a real directory")
    found: dict[str, Path] = {}
    for current, dirnames, filenames in os.walk(root, followlinks=False):
        for name in dirnames + filenames:
            item = Path(current) / name
            mode = item.lstat().st_mode
            _require(not stat.S_ISLNK(mode), f"symbolic link rejected: {item}")
            _require(stat.S_ISDIR(mode) or stat.S_ISREG(mode), f"special file rejected: {item}")
        for filename in filenames:
            path = Path(current) / filename
            rel = path.relative_to(root).as_posix()
            if rel != INDEX:
                found[rel] = path
    return found


def _descriptor_identity(raw: Any, stream_id: str) -> str:
    _require(isinstance(raw, dict), "stream descriptor must be an object")
    required = (
        "stream_id", "modality", "sample_rate_hz", "channel_names",
        "channel_types", "units", "device", "manufacturer", "clock_domain", "metadata",
    )
    _require(all(key in raw for key in required), "stream descriptor missing identity fields")
    _require(raw["stream_id"] == stream_id, "stream ID/descriptor mismatch")
    _require(type(raw["sample_rate_hz"]) in (int, float), "invalid descriptor sample rate")
    payload = {"schema": "neuros.stream_descriptor.v1"}
    for field in required:
        payload[field] = raw[field]
    # Canonical neurOS StreamDescriptor normalizes the declared rate to float.
    payload["sample_rate_hz"] = float(raw["sample_rate_hz"])
    return hashlib.sha256(_canonical(payload)).hexdigest()


def _archive(root: Path, digests: dict[str, str]) -> tuple[int, int]:
    archive = _read_json(root / "session" / "manifest.json")
    _require(isinstance(archive, dict), "invalid session manifest")
    _require(type(archive.get("schema_version")) is int and archive["schema_version"] == 2,
             "independent verifier supports session archive v2 only")
    _require(archive.get("status") == "complete", "session was not completed")
    streams = archive.get("streams")
    _require(isinstance(streams, dict) and bool(streams), "archive contains no streams")
    total = 0
    for stream_id, entry in streams.items():
        _require(_relative(stream_id) == stream_id and "/" not in stream_id, "invalid stream ID")
        _require(isinstance(entry, dict), "invalid stream manifest entry")
        count = entry.get("frame_count")
        _require(type(count) is int and count >= 0, "invalid stream frame_count")
        prefix = f"session/streams/{stream_id}/"
        descriptor_path = prefix + "descriptor.json"
        _require(descriptor_path in digests, "missing sealed stream descriptor")
        descriptor = _read_json(root / descriptor_path)
        identity = _descriptor_identity(descriptor, stream_id)
        _require(_sha(descriptor.get("fingerprint_sha256"), "descriptor fingerprint") == identity,
                 "descriptor fingerprint does not match its payload")
        _require(_sha(entry.get("descriptor_fingerprint_sha256"), "session descriptor fingerprint") == identity,
                 "session manifest descriptor fingerprint mismatch")
        _require(entry.get("descriptor") == descriptor, "session manifest descriptor differs from file")
        index_path = prefix + "index.ndjson"
        if count == 0:
            _require(index_path not in digests, "zero-frame stream has unexpected index")
            continue
        _require(index_path in digests, "missing sealed stream index")
        observed = 0
        with (root / index_path).open("r", encoding="utf-8") as handle:
            for line in handle:
                _require(bool(line.strip()), "empty line in frame index")
                row = json.loads(line, object_pairs_hook=_unique_pairs)
                _require(isinstance(row, dict), "frame index row is not an object")
                _require(type(row.get("index")) is int and row["index"] == observed,
                         "nonsequential frame row index")
                data = _relative(row.get("data"))
                _require(data == f"frames/{observed:012d}.npy", "unexpected frame path")
                frame_path = prefix + data
                _require(frame_path in digests, "frame missing from sealed artifact index")
                _require(
                    hmac.compare_digest(_sha(row.get("data_sha256"), "frame data SHA"), digests[frame_path]),
                    "inner frame checksum does not match sealed payload",
                )
                _require(type(row.get("sequence_id")) is int and row["sequence_id"] >= 0,
                         "invalid frame sequence")
                observed += 1
        _require(observed == count, "stream frame count differs from index")
        total += observed
    _require(total > 0, "archive contains no verified frames")
    return len(streams), total


def verify(bundle: Path, expected_sha256: str) -> dict[str, Any]:
    """Independently verify the v1 archive and its previously published SHA-256 root."""
    expected_sha256 = _sha(expected_sha256, "externally supplied expected root")
    files = _inventory(bundle)
    _require(INDEX not in files and "manifest.json" in files and "session/manifest.json" in files,
             "qualification bundle missing required manifests")
    index = _read_json(bundle / INDEX)
    _require(isinstance(index, dict) and index.get("algorithm") == "sha256",
             "invalid qualification artifact index")
    rows = index.get("artifacts")
    _require(isinstance(rows, list) and bool(rows), "artifact index has no rows")
    paths: list[str] = []
    digests: dict[str, str] = {}
    for row in rows:
        _require(isinstance(row, dict) and set(row) == {"path", "bytes", "sha256"},
                 "malformed artifact index row")
        name = _relative(row["path"])
        _require(name != INDEX, "artifact index cannot hash itself")
        _require(type(row["bytes"]) is int and row["bytes"] >= 0, "invalid artifact byte count")
        checksum = _sha(row["sha256"], "artifact checksum")
        _require(name not in digests, "duplicate artifact path")
        paths.append(name)
        digests[name] = checksum
    _require(paths == sorted(paths), "artifact index entries must be sorted")
    _require(set(paths) == set(files), "sealed artifact inventory does not match disk")
    for row in rows:
        length, actual_sha = _digest(files[row["path"]])
        _require(length == row["bytes"] and hmac.compare_digest(actual_sha, row["sha256"]),
                 f"artifact byte or hash mismatch: {row['path']}")
    root_sha = hashlib.sha256(_canonical(rows)).hexdigest()
    _require(hmac.compare_digest(root_sha, _sha(index.get("bundle_sha256"), "index root")),
             "index root does not match canonical artifacts")
    _require(hmac.compare_digest(root_sha, expected_sha256), "externally expected root mismatch")
    manifest = _read_json(bundle / "manifest.json")
    _require(isinstance(manifest, dict) and type(manifest.get("schema_version")) is int
             and manifest["schema_version"] == 1, "only qualification manifest v1 is supported")
    _require(manifest.get("status") == "complete"
             and manifest.get("qualification_scope") == "runtime-record-replay"
             and manifest.get("archive") == "session", "unsupported qualification scope/status")
    _require("config.yaml" in digests and
             manifest.get("config_file_sha256") == digests["config.yaml"],
             "config hash does not match sealed config")
    for claim in ("real_dataset_qualified", "hardware_qualified", "closed_loop_qualified",
                  "clinical_qualified"):
        _require(manifest.get("claim_boundary", {}).get(claim) is False,
                 f"unsupported qualification claim: {claim}")
    evidence = _read_json(bundle / "decoder_outputs.json")
    _require(isinstance(evidence, dict) and evidence.get("exact_match") is True
             and evidence.get("record") == evidence.get("replay"),
             "record/replay output digest declaration mismatch")
    digest = evidence["record"]
    _require(isinstance(digest, dict) and type(digest.get("count")) is int
             and digest["count"] > 0 and SHA256.fullmatch(str(digest.get("sha256", ""))),
             "missing or malformed decoder output digest")
    _require(manifest.get("reproducibility", {}).get("decoder_output_digest") == digest
             and manifest["reproducibility"].get("decoder_output_digest_exact") is True,
             "manifest decoder digest does not match sealed evidence")
    runtime = _read_json(bundle / "runtime.json")
    _require(isinstance(runtime, dict), "runtime evidence missing")
    for phase in ("record_quality", "replay_quality"):
        quality = runtime.get(phase)
        _require(isinstance(quality, dict) and quality.get("runtime_state") == "stopped"
                 and quality.get("runtime_failure") is None
                 and type(quality.get("node_failures")) is int and quality["node_failures"] == 0
                 and type(quality.get("edge_items_dropped")) is int
                 and quality["edge_items_dropped"] == 0,
                 f"runtime summary not qualified: {phase}")
    stream_count, frame_count = _archive(bundle, digests)
    return {
        "schema": "neuros.independent_qualification_verification.v1",
        "status": "verified",
        "root_sha256": root_sha,
        "artifact_count": len(rows),
        "stream_count": stream_count,
        "frame_count": frame_count,
        "externally_supplied_root_matched": True,
        "verification_scope": "structural_hashes_and_declared_runtime_evidence_only",
        "scientific_result_interpretable": False,
        "decoder_reexecuted": False,
        "source_authenticity_independently_established": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path, help="Path to the qualification bundle")
    parser.add_argument("--expected-sha256", required=True,
                        help="Previously published SHA-256 root obtained outside this bundle")
    args = parser.parse_args()
    try:
        result = verify(args.bundle, args.expected_sha256)
    except (VerificationError, OSError, UnicodeError, json.JSONDecodeError, OverflowError) as exc:
        print(json.dumps({"status": "failed", "reason": str(exc)}, sort_keys=True), file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
