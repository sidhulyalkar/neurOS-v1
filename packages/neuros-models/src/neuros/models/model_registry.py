"""Legacy local model registry for neurOS.

This registry predates Model Artifact v1 and serializes complete Python objects
with pickle. It is retained for backward compatibility with trusted local files,
but it is **not** a promoted deployment or scientific persistence boundary.
Pickle can execute arbitrary code while loading. New deployment/evidence paths
must use ``export_model_artifact`` / ``load_model_artifact`` instead.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from neuros.models.base_model import BaseModel


class LegacyModelRegistryWarning(UserWarning):
    """Warning emitted when the legacy pickle persistence path is used."""


def _warn_legacy_pickle(operation: str) -> None:
    warnings.warn(
        f"ModelRegistry.{operation}() uses legacy pickle persistence and must only be used "
        "with trusted local files. It is not a promoted neurOS deployment artifact. "
        "Use export_model_artifact()/load_model_artifact() from neuros.models for "
        "content-addressed, tensor-only Model Artifact v1 persistence.",
        LegacyModelRegistryWarning,
        stacklevel=3,
    )


@dataclass
class ModelMetadata:
    """Metadata for a legacy pickle-backed saved model."""

    name: str
    version: str
    model_type: str
    created_at: str
    metrics: Dict[str, float]
    hyperparameters: Dict[str, Any]
    training_info: Dict[str, Any]
    tags: List[str]
    checksum: str
    file_path: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelMetadata":
        return cls(**data)


class ModelRegistry:
    """Backward-compatible registry for trusted local pickle files.

    .. warning::
       This class is legacy persistence. ``load()`` invokes ``pickle.load`` and
       can execute arbitrary code from an untrusted file. It must never be used
       as the promoted deployment, evidence, exchange, or rollback boundary.
       Use Model Artifact v1 for those purposes.
    """

    def __init__(self, registry_dir: Optional[str | Path] = None):
        if registry_dir is None:
            registry_dir = Path.home() / ".neuros" / "models"
        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _compute_checksum(self, file_path: Path) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _generate_version(self) -> str:
        return datetime.now().strftime("%Y%m%d_%H%M%S")

    def save(
        self,
        model: BaseModel,
        name: str,
        *,
        version: Optional[str] = None,
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        training_info: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> ModelMetadata:
        """Save a trusted local model using the legacy pickle format."""

        _warn_legacy_pickle("save")
        name = name.strip().replace(" ", "_")
        if version is None:
            version = self._generate_version()

        model_filename = f"{name}_v{version}.pkl"
        metadata_filename = f"{name}_v{version}.json"
        model_path = self.models_dir / model_filename
        metadata_path = self.metadata_dir / metadata_filename

        if not overwrite and (model_path.exists() or metadata_path.exists()):
            raise FileExistsError(
                f"Model {name} v{version} already exists. Use overwrite=True to replace it."
            )

        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        checksum = self._compute_checksum(model_path)
        metadata = ModelMetadata(
            name=name,
            version=version,
            model_type=model.__class__.__name__,
            created_at=datetime.now().isoformat(),
            metrics=metrics or {},
            hyperparameters=hyperparameters or {},
            training_info=training_info or {},
            tags=tags or [],
            checksum=checksum,
            file_path=str(model_path.relative_to(self.registry_dir)),
        )
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)
        return metadata

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        verify_checksum: bool = True,
    ) -> BaseModel:
        """Load a trusted local legacy pickle.

        Never call this on downloaded, shared, user-supplied, or otherwise
        untrusted model files. Model Artifact v1 is the safe promoted boundary.
        """

        _warn_legacy_pickle("load")
        if version is None:
            metadata = self.get_latest(name)
            if metadata is None:
                raise FileNotFoundError(f"No models found with name '{name}'")
        else:
            metadata_filename = f"{name}_v{version}.json"
            metadata_path = self.metadata_dir / metadata_filename
            if not metadata_path.exists():
                raise FileNotFoundError(f"Model {name} v{version} not found in registry")
            with open(metadata_path, "r") as f:
                metadata = ModelMetadata.from_dict(json.load(f))

        model_path = self.registry_dir / metadata.file_path
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if verify_checksum:
            checksum = self._compute_checksum(model_path)
            if checksum != metadata.checksum:
                raise ValueError(
                    f"Checksum mismatch for {name} v{metadata.version}. File may be corrupted."
                )

        # Deliberately isolated legacy boundary. No promoted artifact path calls pickle.load.
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model

    def get_metadata(self, name: str, version: str) -> Optional[ModelMetadata]:
        metadata_filename = f"{name}_v{version}.json"
        metadata_path = self.metadata_dir / metadata_filename
        if not metadata_path.exists():
            return None
        with open(metadata_path, "r") as f:
            return ModelMetadata.from_dict(json.load(f))

    def get_latest(self, name: str) -> Optional[ModelMetadata]:
        versions = []
        for metadata_file in self.metadata_dir.glob(f"{name}_v*.json"):
            with open(metadata_file, "r") as f:
                versions.append(ModelMetadata.from_dict(json.load(f)))
        if not versions:
            return None
        versions.sort(key=lambda m: m.created_at, reverse=True)
        return versions[0]

    def list_models(self, name_filter: Optional[str] = None) -> List[ModelMetadata]:
        models = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            with open(metadata_file, "r") as f:
                meta = ModelMetadata.from_dict(json.load(f))
                if name_filter and name_filter not in meta.name:
                    continue
                models.append(meta)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    def search(
        self,
        *,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_accuracy: Optional[float] = None,
    ) -> List[ModelMetadata]:
        models = self.list_models()
        results = []
        for meta in models:
            if model_type and meta.model_type != model_type:
                continue
            if tags and not any(tag in meta.tags for tag in tags):
                continue
            if min_accuracy is not None:
                accuracy = meta.metrics.get("accuracy", 0.0)
                if accuracy < min_accuracy:
                    continue
            results.append(meta)
        return results

    def delete(self, name: str, version: str) -> bool:
        model_filename = f"{name}_v{version}.pkl"
        metadata_filename = f"{name}_v{version}.json"
        model_path = self.models_dir / model_filename
        metadata_path = self.metadata_dir / metadata_filename
        deleted = False
        if model_path.exists():
            model_path.unlink()
            deleted = True
        if metadata_path.exists():
            metadata_path.unlink()
            deleted = True
        return deleted

    def export_metadata(self, output_path: str | Path) -> None:
        models = self.list_models()
        export_data = {
            "registry_dir": str(self.registry_dir),
            "export_time": datetime.now().isoformat(),
            "models": [m.to_dict() for m in models],
        }
        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
