"""Stable model-analysis contracts shared by neurOS and mechanistic tooling.

The classes in this module are intentionally dependency-light. ``neuros-models``
owns the description of a decoder's inspectable surfaces; ``neuros-mechint`` owns
the experiments performed on those surfaces.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from types import MappingProxyType
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


def _nonempty(name: str, value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _string_tuple(name: str, values: Any) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a sequence of strings") from exc
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError(f"{name} must contain only non-empty strings")
    return tuple(value.strip() for value in result)


def _capability_tuple(name: str, values: Any) -> tuple[AnalysisCapability, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must contain AnalysisCapability values")
    try:
        result = tuple(values)
    except TypeError as exc:
        raise TypeError(f"{name} must contain AnalysisCapability values") from exc
    if any(not isinstance(value, AnalysisCapability) for value in result):
        raise TypeError(f"{name} must contain AnalysisCapability values")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} cannot contain duplicate capabilities")
    return result


@dataclass(frozen=True, slots=True)
class AnalysisSurface:
    """One semantically named point in a model's computation graph.

    ``path`` follows ``torch.nn.Module.named_modules()`` semantics for PyTorch
    models. Axis names are descriptive rather than executable shape promises,
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

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _nonempty("path", self.path))
        object.__setattr__(self, "role", _nonempty("role", self.role))
        object.__setattr__(self, "axes", _string_tuple("axes", self.axes))
        object.__setattr__(
            self,
            "capabilities",
            _capability_tuple("capabilities", self.capabilities),
        )
        object.__setattr__(
            self,
            "recommended_methods",
            _string_tuple("recommended_methods", self.recommended_methods),
        )
        if not isinstance(self.description, str) or not isinstance(self.notes, str):
            raise TypeError("description and notes must be strings")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["capabilities"] = [cap.value for cap in self.capabilities]
        value["axes"] = list(self.axes)
        value["recommended_methods"] = list(self.recommended_methods)
        return value


@dataclass(frozen=True, slots=True)
class InterpretabilityManifest:
    """Reproducible declaration of a model's mechanistic analysis contract.

    ``sha256()`` is the durable identity. ``fingerprint()`` remains a short
    display-only prefix for backward compatibility and must not be used as an
    artifact or scientific authority identifier.
    """

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

    def __post_init__(self) -> None:
        for name in ("model_type", "architecture", "backend", "output_semantics", "schema_version"):
            object.__setattr__(self, name, _nonempty(name, getattr(self, name)))
        input_axes = _string_tuple("input_axes", self.input_axes)
        if not input_axes:
            raise ValueError("input_axes must be non-empty")
        if len(set(input_axes)) != len(input_axes):
            raise ValueError("input_axes cannot contain duplicate axis names")
        object.__setattr__(self, "input_axes", input_axes)

        try:
            surfaces = tuple(self.surfaces)
        except TypeError as exc:
            raise TypeError("surfaces must contain AnalysisSurface values") from exc
        if any(not isinstance(surface, AnalysisSurface) for surface in surfaces):
            raise TypeError("surfaces must contain AnalysisSurface values")
        paths = [surface.path for surface in surfaces]
        if len(set(paths)) != len(paths):
            raise ValueError("surfaces cannot repeat a module path")
        object.__setattr__(self, "surfaces", surfaces)
        object.__setattr__(
            self,
            "capabilities",
            _capability_tuple("capabilities", self.capabilities),
        )
        object.__setattr__(self, "limitations", _string_tuple("limitations", self.limitations))

        if not isinstance(self.method_notes, Mapping):
            raise TypeError("method_notes must be a mapping")
        notes: dict[str, str] = {}
        for raw_key, raw_value in self.method_notes.items():
            key = _nonempty("method_notes key", raw_key)
            value = _nonempty(f"method_notes[{key!r}]", raw_value)
            if key in notes:
                raise ValueError("method_notes keys cannot duplicate after normalization")
            notes[key] = value
        object.__setattr__(self, "method_notes", MappingProxyType(dict(sorted(notes.items()))))

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

    def sha256(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def fingerprint(self) -> str:
        """Display-only 16-character prefix retained for compatibility."""

        return self.sha256()[:16]


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
