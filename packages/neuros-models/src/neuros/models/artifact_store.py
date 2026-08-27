"""Content-addressed storage and atomic rollback references for Model Artifact v1."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Mapping

from neuros.models.artifact import (
    ArtifactBackedDecoder,
    ModelArtifactManifest,
    load_model_artifact,
    verify_model_artifact,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _sha256(value: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError("artifact SHA must be a 64-character lowercase SHA-256 digest")
    return value


def _ref_name(value: str) -> str:
    if not isinstance(value, str) or not _REF_RE.fullmatch(value):
        raise ValueError(
            "artifact ref must start with an alphanumeric character and contain only "
            "letters, digits, '.', '_', or '-'"
        )
    if _SHA256_RE.fullmatch(value):
        raise ValueError("artifact refs cannot be SHA-shaped because resolution would be ambiguous")
    return value


def _strict_ref_payload(path: Path) -> Mapping[str, Any]:
    def pairs(values: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in values:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in artifact ref")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("artifact ref is not valid UTF-8 JSON") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("artifact ref root must be a JSON object")
    if set(payload) != {"schema_version", "ref", "artifact_sha256"}:
        raise ValueError("artifact ref contains missing or unknown fields")
    if payload["schema_version"] != 1 or isinstance(payload["schema_version"], bool):
        raise ValueError("artifact ref schema_version must be 1")
    return payload


class ModelArtifactStore:
    """Immutable artifact objects with atomic mutable references.

    Artifacts are stored under ``artifacts/<artifact_sha256>`` and never
    overwritten. A reference such as ``active`` is a tiny JSON pointer that can
    move between existing artifacts. Rollback therefore mutates only the
    pointer, never model bytes or provenance.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.artifacts_dir = self.root / "artifacts"
        self.refs_dir = self.root / "refs"
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.refs_dir.mkdir(parents=True, exist_ok=True)

    def artifact_path(self, artifact_sha256: str) -> Path:
        return self.artifacts_dir / _sha256(artifact_sha256)

    def publish(self, source: str | Path) -> ModelArtifactManifest:
        """Copy a verified artifact into content-addressed immutable storage."""

        source_path = Path(source)
        manifest = verify_model_artifact(source_path)
        destination = self.artifact_path(manifest.artifact_sha256)
        if destination.exists():
            existing = verify_model_artifact(destination)
            if existing.artifact_sha256 != manifest.artifact_sha256:
                raise ValueError("content-addressed artifact destination contains a different object")
            return existing

        temporary: Path | None = Path(
            tempfile.mkdtemp(prefix=".publish-", dir=str(self.artifacts_dir))
        )
        try:
            shutil.copy2(source_path / "manifest.json", temporary / "manifest.json")
            shutil.copy2(
                source_path / "weights.safetensors",
                temporary / "weights.safetensors",
            )
            staged = verify_model_artifact(temporary)
            if staged.artifact_sha256 != manifest.artifact_sha256:
                raise ValueError("staged artifact identity changed during publication")
            try:
                os.replace(temporary, destination)
                temporary = None
            except OSError:
                # A concurrent publisher may have won the same content address.
                if not destination.exists():
                    raise
                existing = verify_model_artifact(destination)
                if existing.artifact_sha256 != manifest.artifact_sha256:
                    raise ValueError(
                        "content-addressed artifact destination was populated by different bytes"
                    )
            return manifest
        finally:
            if temporary is not None and temporary.exists():
                shutil.rmtree(temporary)

    def activate(self, ref: str, artifact_sha256: str) -> str:
        """Atomically point ``ref`` at an already published artifact."""

        name = _ref_name(ref)
        sha = _sha256(artifact_sha256)
        manifest = verify_model_artifact(self.artifact_path(sha))
        if manifest.artifact_sha256 != sha:
            raise ValueError("artifact directory identity does not match requested SHA")

        target = self.refs_dir / f"{name}.json"
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{name}.", suffix=".tmp", dir=str(self.refs_dir)
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump(
                    {
                        "schema_version": 1,
                        "ref": name,
                        "artifact_sha256": sha,
                    },
                    stream,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, target)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)
        return sha

    def rollback(self, ref: str, artifact_sha256: str) -> str:
        """Move a reference back to a previously published immutable artifact."""

        return self.activate(ref, artifact_sha256)

    def resolve(self, ref_or_sha256: str) -> tuple[Path, ModelArtifactManifest]:
        """Resolve either a full artifact SHA or an atomic named reference."""

        if isinstance(ref_or_sha256, str) and _SHA256_RE.fullmatch(ref_or_sha256):
            sha = _sha256(ref_or_sha256)
        else:
            name = _ref_name(ref_or_sha256)
            ref_path = self.refs_dir / f"{name}.json"
            if not ref_path.is_file():
                raise FileNotFoundError(f"artifact ref not found: {name}")
            payload = _strict_ref_payload(ref_path)
            if payload["ref"] != name:
                raise ValueError("artifact ref file does not match the requested ref name")
            sha = _sha256(payload["artifact_sha256"])

        path = self.artifact_path(sha)
        manifest = verify_model_artifact(path)
        if manifest.artifact_sha256 != sha:
            raise ValueError("resolved artifact content does not match its content address")
        return path, manifest

    def active_sha256(self, ref: str) -> str:
        return self.resolve(ref)[1].artifact_sha256

    def load(self, ref_or_sha256: str, *, device: str = "cpu") -> ArtifactBackedDecoder:
        path, _manifest = self.resolve(ref_or_sha256)
        return load_model_artifact(path, device=device)

    def list_artifacts(self) -> tuple[str, ...]:
        values: list[str] = []
        for entry in self.artifacts_dir.iterdir():
            if entry.is_dir() and _SHA256_RE.fullmatch(entry.name):
                manifest = verify_model_artifact(entry)
                if manifest.artifact_sha256 != entry.name:
                    raise ValueError("artifact store contains a misaddressed artifact directory")
                values.append(entry.name)
        return tuple(sorted(values))
