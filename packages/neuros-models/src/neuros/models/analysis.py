"""Stable model-analysis contracts shared by neurOS and mechanistic tooling.

The classes in this module are intentionally dependency-light.  ``neuros-models``
owns the description of a decoder's inspectable surfaces; ``neuros-mechint`` owns
the experiments performed on those surfaces.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, runtime_checkable


class AnalysisCapability(str, Enum):
    """Mechanistic operations a model surface can support."""

    ACTIVATION_CAPTURE = "activation_capture"
    ACTIVATION_REPLACEMENT = "activation_replacement"
    GRADIENT_ATTRIBUTION = "gradient_attribution"
    REPRESENTATIONS = "representations"
    ATTENTION = "attention"
    FEATURE_IMPORTANCE = "feature_importance"
    MODALITY_GATING = "modality_gating"


@dataclass(frozen=True, slots=True)
class AnalysisSurface:
    """One semantically named point in a model's computation graph.

    ``path`` follows ``torch.nn.Module.named_modules()`` semantics for PyTorch
    models.  Axis names are descriptive rather than executable shape promises,
    which lets the same surface remain stable across batch/window lengths.
    """

    path: str
    role: str
    axes: tuple[str, ...] = ()
    description: str = ""
    capabilities: tuple[AnalysisCapability, ...] = (
        AnalysisCapability.ACTIVATION_CAPTURE,
        AnalysisCapability.ACTIVATION_REPLACEMENT,
    )
    recommended_methods: tuple[str, ...] = ()
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["capabilities"] = [cap.value for cap in self.capabilities]
        value["axes"] = list(self.axes)
        value["recommended_methods"] = list(self.recommended_methods)
        return value


@dataclass(frozen=True, slots=True)
class InterpretabilityManifest:
    """Reproducible declaration of a model's mechanistic analysis contract."""

    model_type: str
    architecture: str
    backend: str
    input_axes: tuple[str, ...]
    output_semantics: str
    surfaces: tuple[AnalysisSurface, ...] = ()
    capabilities: tuple[AnalysisCapability, ...] = ()
    method_notes: Mapping[str, str] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    schema_version: str = "1"

    @classmethod
    def opaque(cls, model_type: str) -> "InterpretabilityManifest":
        return cls(
            model_type=model_type,
            architecture="opaque",
            backend="unknown",
            input_axes=("batch", "features"),
            output_semantics="decoder prediction",
            limitations=(
                "No stable internal activation surface is declared. Treat the model as a black-box decoder.",
            ),
        )

    @property
    def mechint_ready(self) -> bool:
        return bool(self.surfaces) and AnalysisCapability.ACTIVATION_CAPTURE in self.capabilities

    @property
    def surface_paths(self) -> tuple[str, ...]:
        return tuple(surface.path for surface in self.surfaces)

    @property
    def recommended_paths(self) -> tuple[str, ...]:
        """Tensor-output surfaces safe for the generic capture/replacement adapter."""

        return tuple(
            surface.path
            for surface in self.surfaces
            if AnalysisCapability.ACTIVATION_CAPTURE in surface.capabilities
        )

    def surface(self, path: str) -> AnalysisSurface:
        for item in self.surfaces:
            if item.path == path:
                return item
        raise KeyError(f"Unknown analysis surface: {path}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_type": self.model_type,
            "architecture": self.architecture,
            "backend": self.backend,
            "input_axes": list(self.input_axes),
            "output_semantics": self.output_semantics,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "capabilities": [cap.value for cap in self.capabilities],
            "method_notes": dict(self.method_notes),
            "limitations": list(self.limitations),
        }

    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@runtime_checkable
class MechanisticallyInspectable(Protocol):
    """Duck-typed boundary consumed by ``neuros-mechint``."""

    def analysis_manifest(self) -> InterpretabilityManifest:
        ...

    def analysis_model(self) -> Any:
        ...


def validate_manifest_paths(model: Any, manifest: InterpretabilityManifest) -> tuple[str, ...]:
    """Return manifest paths missing from a PyTorch-like ``named_modules`` graph."""

    if not hasattr(model, "named_modules"):
        return manifest.recommended_paths
    paths = {name for name, _ in model.named_modules()}
    return tuple(path for path in manifest.surface_paths if path not in paths)
