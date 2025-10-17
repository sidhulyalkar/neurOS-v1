"""
Model registry for neurOS.

This module provides a centralized system for saving, loading, and managing
trained models with metadata. It supports versioning, tagging, and searching
models by various criteria.
"""

from __future__ import annotations

import json
import pickle
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np

from neuros.models.base_model import BaseModel


@dataclass
class ModelMetadata:
    """Metadata for a saved model.

    Parameters
    ----------
    name : str
        Human-readable model name.
    version : str
        Model version (e.g., "1.0.0" or timestamp-based).
    model_type : str
        Class name of the model (e.g., "EEGNetModel").
    created_at : str
        ISO format timestamp of when model was saved.
    metrics : dict
        Performance metrics (accuracy, loss, etc.).
    hyperparameters : dict
        Model hyperparameters used during training.
    training_info : dict
        Additional training information (dataset, duration, etc.).
    tags : list[str]
        User-defined tags for organization.
    checksum : str
        SHA-256 hash of the model file for integrity checking.
    file_path : str
        Relative path to the model file.
    """

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
        """Convert metadata to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> ModelMetadata:
        """Create metadata from dictionary."""
        return cls(**data)


class ModelRegistry:
    """Registry for managing trained models.

    The registry stores models with metadata in a structured directory:

    ```
    registry_dir/
    ├── models/
    │   ├── motor_imagery_v1.pkl
    │   ├── motor_imagery_v2.pkl
    │   └── p300_speller_v1.pkl
    └── metadata/
        ├── motor_imagery_v1.json
        ├── motor_imagery_v2.json
        └── p300_speller_v1.json
    ```

    Parameters
    ----------
    registry_dir : str or Path
        Directory to store models and metadata. Defaults to ~/.neuros/models

    Examples
    --------
    >>> from neuros.models import SimpleClassifier, ModelRegistry
    >>> import numpy as np

    >>> # Train a model
    >>> model = SimpleClassifier()
    >>> X = np.random.randn(100, 40)
    >>> y = np.random.randint(0, 2, 100)
    >>> model.train(X, y)

    >>> # Save with metadata
    >>> registry = ModelRegistry()
    >>> registry.save(
    ...     model,
    ...     name="motor_imagery_classifier",
    ...     version="1.0.0",
    ...     metrics={"accuracy": 0.92, "f1_score": 0.89},
    ...     tags=["motor-imagery", "production"],
    ... )

    >>> # Load model
    >>> loaded_model = registry.load("motor_imagery_classifier", version="1.0.0")

    >>> # List all models
    >>> models = registry.list_models()
    >>> for meta in models:
    ...     print(f"{meta.name} v{meta.version}: {meta.metrics}")

    >>> # Search by tags
    >>> prod_models = registry.search(tags=["production"])
    """

    def __init__(self, registry_dir: Optional[str | Path] = None):
        """Initialize model registry.

        Parameters
        ----------
        registry_dir : str or Path, optional
            Directory to store models. If None, uses ~/.neuros/models
        """
        if registry_dir is None:
            registry_dir = Path.home() / ".neuros" / "models"

        self.registry_dir = Path(registry_dir)
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"

        # Create directories if they don't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _generate_version(self) -> str:
        """Generate timestamp-based version if not provided."""
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
        """Save a model with metadata.

        Parameters
        ----------
        model : BaseModel
            Trained model to save.
        name : str
            Model name (alphanumeric, hyphens, underscores).
        version : str, optional
            Version string. If None, generates timestamp-based version.
        metrics : dict, optional
            Performance metrics (e.g., {"accuracy": 0.95}).
        hyperparameters : dict, optional
            Model hyperparameters.
        training_info : dict, optional
            Additional training information.
        tags : list[str], optional
            Tags for organization.
        overwrite : bool, optional
            If True, overwrite existing model. Default is False.

        Returns
        -------
        ModelMetadata
            Metadata for the saved model.

        Raises
        ------
        ValueError
            If model with same name/version exists and overwrite=False.
        FileExistsError
            If model file already exists and overwrite=False.
        """
        # Validate and normalize name
        name = name.strip().replace(" ", "_")

        # Generate version if not provided
        if version is None:
            version = self._generate_version()

        # Create file paths
        model_filename = f"{name}_v{version}.pkl"
        metadata_filename = f"{name}_v{version}.json"

        model_path = self.models_dir / model_filename
        metadata_path = self.metadata_dir / metadata_filename

        # Check for existing files
        if not overwrite and (model_path.exists() or metadata_path.exists()):
            raise FileExistsError(
                f"Model {name} v{version} already exists. "
                "Use overwrite=True to replace it."
            )

        # Save model using pickle
        with open(model_path, "wb") as f:
            pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

        # Compute checksum
        checksum = self._compute_checksum(model_path)

        # Create metadata
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

        # Save metadata
        with open(metadata_path, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        return metadata

    def load(
        self,
        name: str,
        version: Optional[str] = None,
        verify_checksum: bool = True,
    ) -> BaseModel:
        """Load a model from the registry.

        Parameters
        ----------
        name : str
            Model name.
        version : str, optional
            Version to load. If None, loads the latest version.
        verify_checksum : bool, optional
            If True, verify file integrity before loading. Default is True.

        Returns
        -------
        BaseModel
            Loaded model.

        Raises
        ------
        FileNotFoundError
            If model or metadata not found.
        ValueError
            If checksum verification fails.
        """
        # Find metadata
        if version is None:
            # Load latest version
            metadata = self.get_latest(name)
            if metadata is None:
                raise FileNotFoundError(f"No models found with name '{name}'")
        else:
            metadata_filename = f"{name}_v{version}.json"
            metadata_path = self.metadata_dir / metadata_filename

            if not metadata_path.exists():
                raise FileNotFoundError(
                    f"Model {name} v{version} not found in registry"
                )

            with open(metadata_path, "r") as f:
                metadata = ModelMetadata.from_dict(json.load(f))

        # Load model file
        model_path = self.registry_dir / metadata.file_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Verify checksum
        if verify_checksum:
            checksum = self._compute_checksum(model_path)
            if checksum != metadata.checksum:
                raise ValueError(
                    f"Checksum mismatch for {name} v{metadata.version}. "
                    f"File may be corrupted."
                )

        # Load model
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        return model

    def get_metadata(self, name: str, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a specific model version.

        Parameters
        ----------
        name : str
            Model name.
        version : str
            Model version.

        Returns
        -------
        ModelMetadata or None
            Metadata if found, None otherwise.
        """
        metadata_filename = f"{name}_v{version}.json"
        metadata_path = self.metadata_dir / metadata_filename

        if not metadata_path.exists():
            return None

        with open(metadata_path, "r") as f:
            return ModelMetadata.from_dict(json.load(f))

    def get_latest(self, name: str) -> Optional[ModelMetadata]:
        """Get metadata for the latest version of a model.

        Parameters
        ----------
        name : str
            Model name.

        Returns
        -------
        ModelMetadata or None
            Latest version metadata if found, None otherwise.
        """
        versions = []
        for metadata_file in self.metadata_dir.glob(f"{name}_v*.json"):
            with open(metadata_file, "r") as f:
                meta = ModelMetadata.from_dict(json.load(f))
                versions.append(meta)

        if not versions:
            return None

        # Sort by creation time
        versions.sort(key=lambda m: m.created_at, reverse=True)
        return versions[0]

    def list_models(
        self,
        name_filter: Optional[str] = None,
    ) -> List[ModelMetadata]:
        """List all models in the registry.

        Parameters
        ----------
        name_filter : str, optional
            Filter models by name (substring match).

        Returns
        -------
        list[ModelMetadata]
            List of model metadata, sorted by creation time (newest first).
        """
        models = []
        for metadata_file in self.metadata_dir.glob("*.json"):
            with open(metadata_file, "r") as f:
                meta = ModelMetadata.from_dict(json.load(f))

                # Apply filter
                if name_filter and name_filter not in meta.name:
                    continue

                models.append(meta)

        # Sort by creation time (newest first)
        models.sort(key=lambda m: m.created_at, reverse=True)
        return models

    def search(
        self,
        *,
        model_type: Optional[str] = None,
        tags: Optional[List[str]] = None,
        min_accuracy: Optional[float] = None,
    ) -> List[ModelMetadata]:
        """Search for models by criteria.

        Parameters
        ----------
        model_type : str, optional
            Filter by model type (e.g., "EEGNetModel").
        tags : list[str], optional
            Filter by tags (returns models with ANY of the tags).
        min_accuracy : float, optional
            Minimum accuracy threshold.

        Returns
        -------
        list[ModelMetadata]
            Matching models, sorted by creation time.
        """
        models = self.list_models()
        results = []

        for meta in models:
            # Filter by model type
            if model_type and meta.model_type != model_type:
                continue

            # Filter by tags
            if tags and not any(tag in meta.tags for tag in tags):
                continue

            # Filter by accuracy
            if min_accuracy is not None:
                accuracy = meta.metrics.get("accuracy", 0.0)
                if accuracy < min_accuracy:
                    continue

            results.append(meta)

        return results

    def delete(self, name: str, version: str) -> bool:
        """Delete a model and its metadata.

        Parameters
        ----------
        name : str
            Model name.
        version : str
            Model version.

        Returns
        -------
        bool
            True if deleted, False if not found.
        """
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
        """Export all metadata to a single JSON file.

        Parameters
        ----------
        output_path : str or Path
            Path to save the metadata export.
        """
        models = self.list_models()
        export_data = {
            "registry_dir": str(self.registry_dir),
            "export_time": datetime.now().isoformat(),
            "models": [m.to_dict() for m in models],
        }

        with open(output_path, "w") as f:
            json.dump(export_data, f, indent=2)
