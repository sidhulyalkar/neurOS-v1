"""Versioned artifact metadata for reproducible neurOS models."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class ModelArtifactManifest:
    """Portable metadata that accompanies a serialized model artifact."""

    model_id: str
    model_version: str
    architecture: str
    input_schema_version: str
    artifact_hash: str
    created_at: str
    git_sha: str | None = None
    subject_scope: str | None = None
    training_dataset_hashes: tuple[str, ...] = ()
    metrics: Mapping[str, float] = field(default_factory=dict)
    dependencies: Mapping[str, str] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.model_id or not self.model_version:
            raise ValueError("model_id and model_version must be non-empty")
        if not self.artifact_hash:
            raise ValueError("artifact_hash must be non-empty")
        object.__setattr__(self, "metrics", MappingProxyType(dict(self.metrics)))
        object.__setattr__(self, "dependencies", MappingProxyType(dict(self.dependencies)))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
