"""Reproducibility manifests for neurOS benchmarks and scientific gates."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Mapping


def content_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_git_sha() -> str | None:
    if os.environ.get("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def installed_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("neuros", "neuros-core", "neuros-drivers", "neuros-models", "neuros-orion"):
        try:
            result[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return result


@dataclass(frozen=True, slots=True)
class BenchmarkManifest:
    benchmark_id: str
    created_at: str
    git_sha: str | None
    config_hash: str | None
    data_hash: str | None
    artifact_ids: tuple[str, ...] = ()
    seed: int | None = None
    packages: Mapping[str, str] = field(default_factory=dict)
    host: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def capture(
        cls,
        benchmark_id: str,
        *,
        config: Any | None = None,
        data_fingerprint: Any | None = None,
        artifact_ids: tuple[str, ...] = (),
        seed: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "BenchmarkManifest":
        return cls(
            benchmark_id=benchmark_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            git_sha=current_git_sha(),
            config_hash=content_hash(config) if config is not None else None,
            data_hash=content_hash(data_fingerprint) if data_fingerprint is not None else None,
            artifact_ids=artifact_ids,
            seed=seed,
            packages=installed_versions(),
            host={
                "python": platform.python_version(),
                "platform": platform.platform(),
                "machine": platform.machine(),
            },
            metadata=dict(metadata or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
