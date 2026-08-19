"""Typed results for internal and input-level causal experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from .manifest import ExperimentManifest


@dataclass(frozen=True, slots=True)
class InterventionEffect:
    name: str
    component: str
    clean_metric: float
    corrupted_metric: float
    intervened_metric: float
    effect: float
    recovered_fraction: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metrics(
        cls,
        *,
        name: str,
        component: str,
        clean_metric: float,
        corrupted_metric: float,
        intervened_metric: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> InterventionEffect:
        total_effect = clean_metric - corrupted_metric
        recovered = None
        if abs(total_effect) > 1e-12:
            recovered = (intervened_metric - corrupted_metric) / total_effect
        return cls(
            name=name,
            component=component,
            clean_metric=clean_metric,
            corrupted_metric=corrupted_metric,
            intervened_metric=intervened_metric,
            effect=intervened_metric - corrupted_metric,
            recovered_fraction=recovered,
            metadata=dict(metadata or {}),
        )


@dataclass(frozen=True, slots=True)
class InputInterventionEffect:
    """Effect of editing an experiment input relative to one fixed baseline."""

    name: str
    target: str
    baseline_metric: float
    intervened_metric: float
    effect: float
    relative_effect: float | None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metrics(
        cls,
        *,
        name: str,
        target: str,
        baseline_metric: float,
        intervened_metric: float,
        metadata: Mapping[str, Any] | None = None,
    ) -> InputInterventionEffect:
        relative = None
        if abs(baseline_metric) > 1e-12:
            relative = (intervened_metric - baseline_metric) / abs(baseline_metric)
        return cls(
            name=name,
            target=target,
            baseline_metric=baseline_metric,
            intervened_metric=intervened_metric,
            effect=intervened_metric - baseline_metric,
            relative_effect=relative,
            metadata=dict(metadata or {}),
        )


@dataclass(slots=True)
class ExperimentResult:
    manifest: ExperimentManifest
    metric_name: str
    clean_metric: float
    corrupted_metric: float
    effects: list[InterventionEffect] = field(default_factory=list)
    controls: list[InterventionEffect] = field(default_factory=list)

    @property
    def total_effect(self) -> float:
        return self.clean_metric - self.corrupted_metric

    @property
    def specificity_gap(self) -> float | None:
        if not self.effects or not self.controls:
            return None
        signal = max(abs(item.effect) for item in self.effects)
        control = max(abs(item.effect) for item in self.controls)
        return signal - control

    def top_effects(self, k: int = 10) -> list[InterventionEffect]:
        return sorted(self.effects, key=lambda item: abs(item.effect), reverse=True)[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_hash": self.manifest.content_hash,
            "metric_name": self.metric_name,
            "clean_metric": self.clean_metric,
            "corrupted_metric": self.corrupted_metric,
            "total_effect": self.total_effect,
            "specificity_gap": self.specificity_gap,
            "effects": [asdict(effect) for effect in self.effects],
            "controls": [asdict(control) for control in self.controls],
        }


@dataclass(slots=True)
class InputExperimentResult:
    manifest: ExperimentManifest
    metric_name: str
    baseline_metric: float
    effects: list[InputInterventionEffect] = field(default_factory=list)
    controls: list[InputInterventionEffect] = field(default_factory=list)

    @property
    def specificity_gap(self) -> float | None:
        if not self.effects or not self.controls:
            return None
        signal = max(abs(item.effect) for item in self.effects)
        control = max(abs(item.effect) for item in self.controls)
        return signal - control

    def top_effects(self, k: int = 10) -> list[InputInterventionEffect]:
        return sorted(self.effects, key=lambda item: abs(item.effect), reverse=True)[:k]

    def to_dict(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.to_dict(),
            "manifest_hash": self.manifest.content_hash,
            "metric_name": self.metric_name,
            "baseline_metric": self.baseline_metric,
            "specificity_gap": self.specificity_gap,
            "effects": [asdict(effect) for effect in self.effects],
            "controls": [asdict(control) for control in self.controls],
        }
