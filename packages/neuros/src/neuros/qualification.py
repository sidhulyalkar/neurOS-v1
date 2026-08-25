"""Reproducible neurOS qualification bundles.

A qualification bundle is an evidence artifact, not a blanket product claim.
Version 1 proves a narrower property: a declared runtime configuration was run,
its exact input frames were recorded with integrity hashes, and the recorded
session reproduced the same canonical decoder-output digest through replay.

Hardware, real-dataset, closed-loop, and clinical qualification require
additional evidence and are explicitly *not* inferred from a successful v1
bundle.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import uuid
from dataclasses import fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from neuros.compatibility import compatibility_payload
from neuros.contracts import DecoderOutput
from neuros.quality import BenchmarkManifest
from neuros.recording import SessionArchiveReader, canonical_hash

from .cli.recording_commands import inspect_archive, record_config, replay_archive

QUALIFICATION_SCHEMA_VERSION = 1
_OUTPUT_DIGEST_ALGORITHM = "sha256-semantic-decoder-jsonl-v1"
_RELEVANT_PACKAGES = (
    "neuros",
    "neuros-core",
    "neuros-drivers",
    "neuros-models",
    "neuros-foundation",
    "neuros-sourceweigher",
    "neuros-mechint",
    "neuros-orion",
    "numpy",
    "scipy",
    "torch",
    "scikit-learn",
    "mne",
    "moabb",
    "braindecode",
    "brainflow",
    "pylsl",
    "pynwb",
    "zarr",
)


def _jsonable(value: Any) -> Any:
    """Convert evidence values without deepcopying immutable mapping proxies.

    ``dataclasses.asdict`` recursively deep-copies values. neurOS contracts use
    immutable ``MappingProxyType`` metadata by design, and deep-copying those
    objects fails. Evidence serialization instead walks dataclass fields and
    mappings directly so the digest reflects values without mutating or copying
    their authority wrappers.
    """

    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _jsonable(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        _jsonable(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    ).encode("utf-8")


def _canonical_output_payload(output: Any) -> Any:
    """Return the replay-stable semantic identity of one runtime output.

    ``DecoderOutput.inference_time_ns`` measures how long this particular
    execution took. It is intentionally expected to change between live and
    replay execution, so including it in decision identity would make every
    honest performance measurement look like a reproducibility failure.
    Latency remains preserved separately in runtime telemetry.
    """

    payload = _jsonable(output)
    if isinstance(output, DecoderOutput):
        payload = dict(payload)
        payload.pop("inference_time_ns", None)
    return payload


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_jsonable(value), indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in _RELEVANT_PACKAGES:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


def _load_raw_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("qualification config root must be a mapping")
    return raw


class _OutputDigest:
    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.count = 0

    async def __call__(self, output: Any) -> None:
        self._digest.update(_canonical_bytes(_canonical_output_payload(output)))
        self._digest.update(b"\n")
        self.count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": _OUTPUT_DIGEST_ALGORITHM,
            "count": self.count,
            "sha256": self._digest.hexdigest(),
        }


def _runtime_quality(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    nodes = snapshot.get("nodes", {})
    edges = snapshot.get("edges", {})
    dropped = sum(int(item.get("dropped", 0)) for item in edges.values())
    accepted = sum(int(item.get("accepted", 0)) for item in edges.values())
    failed = sum(int(item.get("failed", 0)) for item in nodes.values())
    p99_values = [float(item.get("p99_latency_ms", 0.0)) for item in nodes.values()]
    return {
        "runtime_state": snapshot.get("state"),
        "runtime_seconds": float(snapshot.get("runtime_seconds", 0.0)),
        "node_failures": failed,
        "edge_items_accepted": accepted,
        "edge_items_dropped": dropped,
        "max_node_p99_latency_ms": max(p99_values, default=0.0),
        "runtime_failure": snapshot.get("failure"),
    }


def _stream_evidence(reader: SessionArchiveReader) -> tuple[dict[str, Any], dict[str, Any]]:
    devices: dict[str, Any] = {}
    clocks: dict[str, Any] = {}
    for stream_id in reader.stream_ids:
        descriptor = reader.descriptor(stream_id)
        devices[stream_id] = {
            "modality": descriptor.modality,
            "sample_rate_hz": descriptor.sample_rate_hz,
            "channel_names": list(descriptor.channel_names),
            "channel_types": list(descriptor.channel_types),
            "units": list(descriptor.units),
            "device": descriptor.device,
            "manufacturer": descriptor.manufacturer,
            "descriptor_metadata": dict(descriptor.metadata),
        }
        clocks[stream_id] = {
            "clock_domain": descriptor.clock_domain.value,
            "timing_semantics": descriptor.metadata.get("timing_semantics"),
            "clock_uncertainty_ns": descriptor.metadata.get("clock_uncertainty_ns"),
            "clock_drift_ppm": descriptor.metadata.get("clock_drift_ppm"),
        }
    return devices, clocks


def _used_integrations(config: Mapping[str, Any]) -> list[str]:
    integration_ids: set[str] = set()
    for stream in config.get("streams", []):
        if not isinstance(stream, Mapping):
            continue
        source = stream.get("source", {})
        if isinstance(source, Mapping):
            plugin = str(source.get("plugin", "")).lower()
            if plugin in {"brainflow", "lsl"}:
                integration_ids.add(plugin)
    decoder = config.get("decoder", {})
    if isinstance(decoder, Mapping):
        plugin = str(decoder.get("plugin", "")).lower()
        if plugin == "braindecode":
            integration_ids.add("braindecode")
    return sorted(integration_ids)


def _environment_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    benchmark = BenchmarkManifest.capture(
        "qualification-runtime-replay-v1",
        config=config,
        metadata={"qualification_schema_version": QUALIFICATION_SCHEMA_VERSION},
    )
    return {
        "created_at": benchmark.created_at,
        "git_sha": benchmark.git_sha,
        "python": sys.version,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": _installed_versions(),
    }


def _artifact_index(root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "artifact_hashes.json":
            continue
        relative = path.relative_to(root).as_posix()
        rows.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return {
        "algorithm": "sha256",
        "artifacts": rows,
        "bundle_sha256": hashlib.sha256(_canonical_bytes(rows)).hexdigest(),
    }


def _safe_replace_directory(staging: Path, destination: Path, *, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Qualification output already exists: {destination}")
        if destination.is_symlink() or not destination.is_dir():
            raise ValueError("--overwrite only supports an existing directory")
        resolved = destination.resolve()
        if resolved == Path(resolved.anchor) or resolved == Path.home().resolve():
            raise ValueError("Refusing to overwrite a filesystem root or home directory")
        shutil.rmtree(destination)
    staging.replace(destination)


async def qualify_config(
    config_path: str | Path,
    output: str | Path,
    *,
    session_id: str = "qualification",
    duration_s: float = 1.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run, record, replay, and seal one immutable qualification bundle."""

    if duration_s <= 0:
        raise ValueError("qualification duration must be positive")
    source_config = Path(config_path).resolve()
    if not source_config.is_file():
        raise FileNotFoundError(source_config)
    raw_config = _load_raw_config(source_config)

    destination = Path(output).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Qualification output already exists: {destination}")
    staging = destination.parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)

    live_outputs = _OutputDigest()
    replay_outputs = _OutputDigest()
    try:
        bundled_config = staging / "config.yaml"
        shutil.copyfile(source_config, bundled_config)
        _write_json(staging / "config.json", raw_config)

        session_root = staging / "session"
        record_summary = await record_config(
            source_config,
            session_root,
            session_id=session_id,
            duration_s=duration_s,
            overwrite=False,
            on_output=live_outputs,
        )
        # ``record_config`` correctly reports the path it wrote to, but this run
        # occurs inside a temporary staging directory that is renamed once the
        # bundle has verified. Persist only portable bundle-relative references.
        record_summary = dict(record_summary)
        record_summary["archive"] = "session"
        record_summary["exports"] = {}

        archive_summary = inspect_archive(session_root, verify_hashes=True)
        replay_snapshot = await replay_archive(
            session_root,
            bundled_config,
            realtime=False,
            speed=1.0,
            duration_s=None,
            on_output=replay_outputs,
        )

        live_digest = live_outputs.to_dict()
        replay_digest = replay_outputs.to_dict()
        if live_digest != replay_digest:
            raise RuntimeError(
                "Qualification replay output mismatch: the recorded session did not "
                "reproduce the canonical decoder-output digest"
            )

        reader = SessionArchiveReader(session_root, verify_hashes=True)
        devices, clocks = _stream_evidence(reader)
        session_manifest = reader.manifest
        environment = _environment_payload(raw_config)
        integrations = _used_integrations(raw_config)
        compatibility = {
            "used_integrations": integrations,
            "records": [
                record
                for integration_id in integrations
                for record in compatibility_payload(integration_id)
            ],
        }
        model_artifacts = list(session_manifest.get("model_artifacts", []))
        decoder_config = raw_config.get("decoder", {})
        model_identity = {
            "decoder_config": decoder_config,
            "artifact_bound": bool(model_artifacts),
            "model_artifacts": model_artifacts,
            "limitation": None
            if model_artifacts
            else (
                "No promoted ModelArtifactManifest was bound to this session. The decoder "
                "configuration is recorded, but learned-weight identity is not claimed."
            ),
        }
        record_runtime = session_manifest.get("runtime_metrics", {})
        replay_quality = _runtime_quality(replay_snapshot)
        record_quality = _runtime_quality(record_runtime)

        _write_json(staging / "environment.json", environment)
        _write_json(staging / "compatibility.json", compatibility)
        _write_json(staging / "devices.json", devices)
        _write_json(staging / "clocks.json", clocks)
        _write_json(staging / "model.json", model_identity)
        _write_json(
            staging / "runtime.json",
            {
                "record": record_runtime,
                "replay": replay_snapshot,
                "record_quality": record_quality,
                "replay_quality": replay_quality,
            },
        )
        _write_json(
            staging / "decoder_outputs.json",
            {"record": live_digest, "replay": replay_digest, "exact_match": True},
        )

        created_at = datetime.now(timezone.utc).isoformat()
        manifest = {
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "bundle_id": f"qualification:{session_id}:{uuid.uuid4().hex}",
            "created_at": created_at,
            "status": "complete",
            "qualification_scope": "runtime-record-replay",
            "evidence_tier": "integration",
            "session_id": session_id,
            "git_sha": environment.get("git_sha"),
            "config_file_sha256": _sha256_file(bundled_config),
            "config_semantic_hash": canonical_hash(raw_config),
            "archive_config_hash": session_manifest.get("config_hash"),
            "archive": "session",
            "record_summary": record_summary,
            "archive_summary": archive_summary,
            "reproducibility": {
                "archive_integrity_verified": archive_summary.get("integrity") == "verified",
                "replay_completed": replay_snapshot.get("state") == "stopped",
                "decoder_output_digest_exact": True,
                "decoder_output_digest": live_digest,
            },
            "claim_boundary": {
                "runtime_record_replay_qualified": True,
                "real_dataset_qualified": False,
                "hardware_qualified": False,
                "closed_loop_qualified": False,
                "clinical_qualified": False,
                "statement": (
                    "This bundle qualifies the recorded software/runtime replay boundary only. "
                    "It does not establish real-dataset utility, physical-device timing or "
                    "reliability, closed-loop safety, or clinical validity."
                ),
            },
        }
        _write_json(staging / "manifest.json", manifest)
        artifact_index = _artifact_index(staging)
        _write_json(staging / "artifact_hashes.json", artifact_index)

        verification = verify_qualification_bundle(staging)
        _safe_replace_directory(staging, destination, overwrite=overwrite)
        return {
            "bundle": str(destination),
            "bundle_id": manifest["bundle_id"],
            "schema_version": QUALIFICATION_SCHEMA_VERSION,
            "status": "complete",
            "evidence_tier": "integration",
            "session_id": session_id,
            "decoder_outputs": live_digest,
            "bundle_sha256": verification["bundle_sha256"],
            "claim_boundary": manifest["claim_boundary"],
        }
    except BaseException:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        raise


def verify_qualification_bundle(path: str | Path) -> dict[str, Any]:
    """Verify the sealed artifact index and embedded session archive."""

    root = Path(path).resolve()
    manifest_path = root / "manifest.json"
    hashes_path = root / "artifact_hashes.json"
    if not manifest_path.is_file() or not hashes_path.is_file():
        raise IOError("Not a complete neurOS qualification bundle")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if int(manifest.get("schema_version", -1)) != QUALIFICATION_SCHEMA_VERSION:
        raise ValueError("Unsupported neurOS qualification schema")
    if manifest.get("status") != "complete":
        raise IOError("Qualification bundle is not marked complete")

    index = json.loads(hashes_path.read_text(encoding="utf-8"))
    rows = index.get("artifacts")
    if not isinstance(rows, list):
        raise IOError("Qualification artifact index is malformed")
    expected_paths = {str(row["path"]) for row in rows}
    actual_paths = {
        item.relative_to(root).as_posix()
        for item in root.rglob("*")
        if item.is_file() and item.name != "artifact_hashes.json"
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        unexpected = sorted(actual_paths - expected_paths)
        raise IOError(
            f"Qualification artifact set mismatch; missing={missing}, unexpected={unexpected}"
        )

    for row in rows:
        artifact = root / str(row["path"])
        if artifact.stat().st_size != int(row["bytes"]):
            raise IOError(f"Qualification artifact size mismatch: {row['path']}")
        if _sha256_file(artifact) != str(row["sha256"]):
            raise IOError(f"Qualification artifact hash mismatch: {row['path']}")

    bundle_sha256 = hashlib.sha256(_canonical_bytes(rows)).hexdigest()
    if bundle_sha256 != index.get("bundle_sha256"):
        raise IOError("Qualification bundle digest mismatch")

    reader = SessionArchiveReader(root / str(manifest.get("archive", "session")), verify_hashes=True)
    frame_count = 0
    for stream_id in reader.stream_ids:
        frame_count += sum(1 for _ in reader.iter_frames(stream_id))

    return {
        "bundle": str(root),
        "bundle_id": manifest.get("bundle_id"),
        "schema_version": QUALIFICATION_SCHEMA_VERSION,
        "integrity": "verified",
        "artifact_count": len(rows),
        "frame_count": frame_count,
        "bundle_sha256": bundle_sha256,
        "claim_boundary": manifest.get("claim_boundary", {}),
    }


async def reproduce_qualification(path: str | Path) -> dict[str, Any]:
    """Verify a bundle and deterministically replay its recorded computational path."""

    verification = verify_qualification_bundle(path)
    root = Path(path).resolve()
    expected = json.loads((root / "decoder_outputs.json").read_text(encoding="utf-8"))["record"]
    capture = _OutputDigest()
    snapshot = await replay_archive(
        root / "session",
        root / "config.yaml",
        realtime=False,
        speed=1.0,
        duration_s=None,
        on_output=capture,
    )
    observed = capture.to_dict()
    if observed != expected:
        raise RuntimeError(
            "Qualification reproduction failed: decoder-output digest differs from sealed bundle"
        )
    return {
        **verification,
        "reproduced": True,
        "runtime_state": snapshot.get("state"),
        "decoder_outputs": observed,
    }
