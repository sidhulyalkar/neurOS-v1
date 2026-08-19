"""Typed metadata contracts for the neural foundation-model landscape.

The public catalog deliberately separates *what a model claims/supports* from
*whether neurOS can execute it locally*. This makes discovery useful even for
closed, gated, or research-only models without pretending they are runnable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable


class NeuralModality(str, Enum):
    EEG = "eeg"
    MEG = "meg"
    ECOG = "ecog"
    SEEG = "seeg"
    SPIKES = "spikes"
    LFP = "lfp"
    CALCIUM = "calcium"
    FMRI = "fmri"
    BEHAVIOR = "behavior"
    VIDEO = "video"
    TEXT = "text"
    MULTIMODAL = "multimodal"
    GENERIC_TIME_SERIES = "generic_time_series"


class ModelTask(str, Enum):
    REPRESENTATION = "representation"
    DECODING = "decoding"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    FORECASTING = "forecasting"
    RECONSTRUCTION = "reconstruction"
    DENOISING = "denoising"
    IMPUTATION = "imputation"
    UPSAMPLING = "upsampling"
    GENERATION = "generation"
    BRAIN_LANGUAGE = "brain_language"
    MULTIMODAL_ALIGNMENT = "multimodal_alignment"


class AccessLevel(str, Enum):
    OPEN_WEIGHTS = "open_weights"
    OPEN_CODE = "open_code"
    GATED = "gated"
    CLOUD = "cloud"
    CLOSED = "closed"
    ANNOUNCED = "announced"
    UNKNOWN = "unknown"


class IntegrationLevel(str, Enum):
    NATIVE = "native"
    ADAPTER = "adapter"
    CATALOG = "catalog"
    LEGACY = "legacy"
    EXPERIMENTAL = "experimental"


class ModelStatus(str, Enum):
    RELEASED = "released"
    RESEARCH = "research"
    COMMERCIAL = "commercial"
    ANNOUNCED = "announced"
    LEGACY = "legacy"


@dataclass(frozen=True, slots=True)
class FoundationModelCard:
    """Machine-readable model card used by the neurOS foundation registry.

    Unknown values remain ``None`` instead of being inferred from secondary
    sources. ``source_url`` points to the primary release/publication used to
    populate the entry.
    """

    id: str
    name: str
    organization: str
    year: int
    modalities: tuple[NeuralModality, ...]
    tasks: tuple[ModelTask, ...]
    architecture: str
    pretraining_objective: str
    input_geometry: str
    transfer_regimes: tuple[str, ...] = ()
    parameters: int | None = None
    pretraining_scale: str | None = None
    access: AccessLevel = AccessLevel.UNKNOWN
    integration: IntegrationLevel = IntegrationLevel.CATALOG
    status: ModelStatus = ModelStatus.RESEARCH
    python_package: str | None = None
    install_extra: str | None = None
    license: str | None = None
    paper_url: str | None = None
    code_url: str | None = None
    weights_url: str | None = None
    source_url: str | None = None
    notes: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    verified_on: str = "2026-08-19"

    def supports_modality(self, modality: NeuralModality | str) -> bool:
        value = NeuralModality(modality)
        return value in self.modalities or NeuralModality.MULTIMODAL in self.modalities

    def supports_task(self, task: ModelTask | str) -> bool:
        return ModelTask(task) in self.tasks

    def has_tags(self, tags: Iterable[str]) -> bool:
        current = {tag.lower() for tag in self.tags}
        return all(tag.lower() in current for tag in tags)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["modalities"] = [value.value for value in self.modalities]
        data["tasks"] = [value.value for value in self.tasks]
        data["access"] = self.access.value
        data["integration"] = self.integration.value
        data["status"] = self.status.value
        data["transfer_regimes"] = list(self.transfer_regimes)
        data["notes"] = list(self.notes)
        data["tags"] = list(self.tags)
        return data


@dataclass(frozen=True, slots=True)
class AdapterAvailability:
    model_id: str
    available: bool
    reason: str
    package: str | None = None
    capabilities: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
