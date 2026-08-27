"""Compatibility surface for the hardened Model Artifact v1 implementation."""

from neuros.models.artifact_v1 import (
    ArtifactBackedDecoder,
    ModelArtifactManifest,
    ModelInputContract,
    ModelOutputContract,
    export_model_artifact,
    load_model_artifact,
    verify_model_artifact,
)

__all__ = [
    "ArtifactBackedDecoder",
    "ModelArtifactManifest",
    "ModelInputContract",
    "ModelOutputContract",
    "export_model_artifact",
    "load_model_artifact",
    "verify_model_artifact",
]
