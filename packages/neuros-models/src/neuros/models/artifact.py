"""Public Model Artifact v1 authority facade.

Canonical serialization, runtime execution policy, and content-addressed storage
are intentionally separate modules. This facade composes the first two without
letting envelope verification silently become execution authorization.
"""

from __future__ import annotations

import shutil
from functools import wraps
from pathlib import Path
from typing import Any, Mapping

from neuros.models.artifact_policy import (
    preflight_bundle_size,
    validate_backend_runtime,
    validate_declared_environment,
    validate_output_contract,
    validate_runtime_authority,
)
from neuros.models.artifact_v1 import (
    ArtifactBackedDecoder,
    ModelArtifactManifest,
    ModelInputContract,
    ModelOutputContract,
    export_model_artifact as _export_model_artifact,
    load_model_artifact as _load_model_artifact,
    verify_model_artifact as _verify_model_artifact,
)


@wraps(_verify_model_artifact)
def verify_model_artifact(path: str | Path) -> ModelArtifactManifest:
    """Verify bounded envelope/content identity without approving execution."""

    preflight_bundle_size(path)
    return _verify_model_artifact(path)


@wraps(_export_model_artifact)
def export_model_artifact(*args: Any, **kwargs: Any) -> ModelArtifactManifest:
    """Promote a trained built-in decoder under the strict v1 runtime policy."""

    package_versions = kwargs.get("package_versions")
    if package_versions is not None:
        if not isinstance(package_versions, Mapping):
            raise TypeError("package_versions must be a mapping")
        validate_declared_environment(package_versions)

    output_contract = kwargs.get("output_contract")
    if output_contract is not None:
        validate_output_contract(output_contract)

    manifest = _export_model_artifact(*args, **kwargs)
    try:
        validate_runtime_authority(manifest)
        validate_backend_runtime(manifest)
        return manifest
    except Exception:
        # A serializer-valid bundle that fails the public promotion policy must
        # not remain at the requested destination looking promoted.
        output_dir = kwargs.get("output_dir")
        if output_dir is None and len(args) >= 2:
            output_dir = args[1]
        if output_dir is not None:
            destination = Path(output_dir)
            if destination.is_dir() and not destination.is_symlink():
                shutil.rmtree(destination)
        raise


@wraps(_load_model_artifact)
def load_model_artifact(path: str | Path, *, device: str = "cpu") -> ArtifactBackedDecoder:
    """Verify envelope then authorize runtime before trusted model allocation."""

    manifest = verify_model_artifact(path)
    validate_runtime_authority(manifest)
    validate_backend_runtime(manifest)
    return _load_model_artifact(path, device=device)


__all__ = [
    "ArtifactBackedDecoder",
    "ModelArtifactManifest",
    "ModelInputContract",
    "ModelOutputContract",
    "export_model_artifact",
    "load_model_artifact",
    "verify_model_artifact",
]
