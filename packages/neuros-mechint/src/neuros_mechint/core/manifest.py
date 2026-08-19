"""Reproducible mechanistic experiment manifests and full-content hashing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields, is_dataclass, replace
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch
from neuros.quality import BenchmarkManifest

from .evidence import EvidenceTier


def _update_hash(hasher: Any, value: Any) -> None:
    """Hash nested values including tensor/array contents, dtype, and shape."""

    if value is None:
        hasher.update(b"none")
    elif isinstance(value, bool):
        hasher.update(b"bool:1" if value else b"bool:0")
    elif isinstance(value, int):
        hasher.update(f"int:{value}".encode())
    elif isinstance(value, float):
        hasher.update(f"float:{value.hex()}".encode())
    elif isinstance(value, str):
        encoded = value.encode("utf-8")
        hasher.update(f"str:{len(encoded)}:".encode())
        hasher.update(encoded)
    elif isinstance(value, bytes):
        hasher.update(f"bytes:{len(value)}:".encode())
        hasher.update(value)
    elif isinstance(value, Path):
        _update_hash(hasher, str(value))
    elif isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        hasher.update(b"torch:")
        _update_hash(hasher, str(tensor.dtype))
        _update_hash(hasher, tuple(tensor.shape))
        hasher.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("object-dtype arrays cannot be hashed deterministically")
        hasher.update(b"numpy:")
        _update_hash(hasher, str(array.dtype))
        _update_hash(hasher, tuple(array.shape))
        hasher.update(array.tobytes(order="C"))
    elif isinstance(value, EvidenceTier):
        _update_hash(hasher, int(value))
    elif is_dataclass(value):
        hasher.update(f"dataclass:{type(value).__module__}.{type(value).__qualname__}:".encode())
        for item in fields(value):
            _update_hash(hasher, item.name)
            _update_hash(hasher, getattr(value, item.name))
    elif isinstance(value, Mapping):
        hasher.update(b"mapping:")
        for key in sorted(value, key=lambda item: str(item)):
            _update_hash(hasher, key)
            _update_hash(hasher, value[key])
    elif isinstance(value, (set, frozenset)):
        hasher.update(b"set:")
        _update_hash(hasher, sorted(stable_hash(item) for item in value))
    elif isinstance(value, Sequence):
        hasher.update(b"sequence:")
        hasher.update(str(len(value)).encode())
        for item in value:
            _update_hash(hasher, item)
    else:
        raise TypeError(f"unsupported value for stable hashing: {type(value)!r}")


def stable_hash(value: Any) -> str:
    """Return a SHA-256 digest over the full nested content of ``value``."""

    hasher = hashlib.sha256()
    _update_hash(hasher, value)
    return hasher.hexdigest()


def stable_hash_or_none(value: Any) -> str | None:
    """Best-effort strong hash for arbitrary research inputs."""

    try:
        return stable_hash(value)
    except (TypeError, ValueError):
        return None


def _research_packages() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in ("neuros-mechint", "neuros-orion"):
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue
    return versions


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    """Versioned provenance record for a mechanistic experiment.

    ``scientific_fingerprint`` identifies the frozen scientific design and excludes
    host/time/runtime metadata. ``run_hash`` identifies the concrete execution and
    therefore includes the captured :class:`BenchmarkManifest`.
    """

    experiment_name: str
    method: str
    model_id: str
    dataset_id: str = "in_memory"
    model_revision: str | None = None
    model_hash: str | None = None
    dataset_hash: str | None = None
    method_version: str = "1"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    seed: int = 0
    evidence_tier: EvidenceTier = EvidenceTier.UNIT
    git_sha: str | None = None
    benchmark: BenchmarkManifest | None = None
    schema_version: str = "3"

    def __post_init__(self) -> None:
        tier = EvidenceTier.coerce(self.evidence_tier)
        object.__setattr__(self, "evidence_tier", tier)
        object.__setattr__(self, "parameters", dict(self.parameters))

        if self.benchmark is None:
            config = {
                "experiment_name": self.experiment_name,
                "method": self.method,
                "method_version": self.method_version,
                "model_id": self.model_id,
                "model_revision": self.model_revision,
                "dataset_id": self.dataset_id,
                "parameters": dict(self.parameters),
                "evidence_tier": tier.label,
            }
            captured = BenchmarkManifest.capture(
                f"mechint:{self.experiment_name}",
                config=config,
                seed=self.seed,
                metadata={
                    "source": "neuros-mechint",
                    "schema_version": self.schema_version,
                    "evidence_tier": tier.label,
                },
            )
            packages = dict(captured.packages)
            packages.update(_research_packages())
            captured = replace(
                captured,
                git_sha=self.git_sha or captured.git_sha,
                config_hash=stable_hash(config),
                data_hash=self.dataset_hash,
                packages=packages,
            )
            object.__setattr__(self, "benchmark", captured)

    def scientific_identity_dict(self) -> dict[str, Any]:
        """Return the deterministic design identity used for independent reproduction."""

        return {
            "experiment_name": self.experiment_name,
            "method": self.method,
            "method_version": self.method_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_hash": self.model_hash,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "evidence_tier": {
                "level": int(self.evidence_tier),
                "label": self.evidence_tier.label,
            },
        }

    @property
    def scientific_fingerprint(self) -> str:
        return stable_hash(self.scientific_identity_dict())

    def to_dict(self) -> dict[str, Any]:
        assert self.benchmark is not None
        return {
            "schema_version": self.schema_version,
            "experiment_name": self.experiment_name,
            "method": self.method,
            "method_version": self.method_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "model_hash": self.model_hash,
            "dataset_id": self.dataset_id,
            "dataset_hash": self.dataset_hash,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "evidence_tier": {
                "level": int(self.evidence_tier),
                "label": self.evidence_tier.label,
            },
            "scientific_identity": self.scientific_identity_dict(),
            "scientific_fingerprint": self.scientific_fingerprint,
            "benchmark": self.benchmark.to_dict(),
        }

    @property
    def run_hash(self) -> str:
        """Hash the complete execution record, including host/time provenance."""

        return stable_hash(self.to_dict())

    @property
    def content_hash(self) -> str:
        """Compatibility alias for the execution-specific ``run_hash``."""

        return self.run_hash

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, default=str)
