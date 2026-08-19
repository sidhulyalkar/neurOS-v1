"""ModelAdapter integration for held-out causal feature correspondence."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from neuros_mechint.adapters.base import ModelAdapter
from neuros_mechint.benchmarks.correspondence import (
    CausalSubstitutionMetrics,
    CorrespondenceSplit,
    FactorialCorrespondenceOrigin,
    FeatureCorrespondenceResult,
    FeatureCorrespondenceSpec,
    FeaturePairExample,
    FeatureSpaceIdentity,
    run_feature_correspondence_study,
)
from neuros_mechint.benchmarks.factorial import FactorialMechanismReport
from neuros_mechint.core.manifest import stable_hash
from neuros_mechint.core.metrics import ScalarMetric


@dataclass(frozen=True, slots=True)
class TensorFeatureProjector:
    """Reduce and inject a declared tensor feature axis.

    The default reduction averages every non-feature axis. Event-preserving or
    token-position-specific studies should provide another projector object with
    the same ``vector`` and ``replace`` methods rather than treating all hidden
    tensor layouts as interchangeable.
    """

    feature_axis: int = -1

    def _axis(self, ndim: int) -> int:
        axis = self.feature_axis if self.feature_axis >= 0 else ndim + self.feature_axis
        if not 0 <= axis < ndim:
            raise ValueError("feature_axis is out of bounds for activation tensor")
        return axis

    def vector(self, activation: torch.Tensor) -> np.ndarray:
        if not isinstance(activation, torch.Tensor) or activation.ndim < 1:
            raise TypeError("feature projector requires a tensor with at least one dimension")
        axis = self._axis(activation.ndim)
        moved = torch.movedim(activation.detach(), axis, -1)
        if moved.ndim == 1:
            reduced = moved
        else:
            reduced = moved.mean(dim=tuple(range(moved.ndim - 1)))
        return reduced.cpu().to(dtype=torch.float64).numpy()

    def replace(
        self,
        activation: torch.Tensor,
        indices: Sequence[int],
        values: Sequence[float],
    ) -> torch.Tensor:
        if not isinstance(activation, torch.Tensor) or activation.ndim < 1:
            raise TypeError("feature projector requires a tensor with at least one dimension")
        axis = self._axis(activation.ndim)
        moved = torch.movedim(activation.detach().clone(), axis, -1)
        indices = tuple(int(index) for index in indices)
        values_array = np.asarray(values, dtype=np.float64).reshape(-1)
        if len(indices) != values_array.size:
            raise ValueError("replacement value count must match selected feature count")
        if any(index < 0 or index >= moved.shape[-1] for index in indices):
            raise IndexError("selected feature index is out of bounds")
        tensor_values = torch.as_tensor(
            values_array,
            device=moved.device,
            dtype=moved.dtype,
        )
        moved[..., list(indices)] = tensor_values
        return torch.movedim(moved, -1, axis)

    def ablate(self, activation: torch.Tensor, indices: Sequence[int]) -> torch.Tensor:
        indices = tuple(indices)
        return self.replace(activation, indices, np.zeros(len(indices), dtype=np.float64))


@dataclass(slots=True)
class AdapterFeatureSpaceView:
    """One ModelAdapter feature surface plus paired-example inputs."""

    identity: FeatureSpaceIdentity
    adapter: ModelAdapter
    path: str
    metric: ScalarMetric
    inputs: Mapping[str, Any]
    projector: TensorFeatureProjector = field(default_factory=TensorFeatureProjector)

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("adapter feature-space path must be non-empty")
        self.inputs = MappingProxyType(dict(self.inputs))

    def activation(self, example_id: str) -> torch.Tensor:
        try:
            inputs = self.inputs[example_id]
        except KeyError as exc:
            raise KeyError(f"missing adapter input for example {example_id!r}") from exc
        activation = self.adapter.capture_outputs(inputs, (self.path,))[self.path]
        vector = self.projector.vector(activation)
        if vector.size != len(self.identity.feature_names):
            raise ValueError(
                f"projected feature count at {self.path!r} is {vector.size}, "
                f"expected {len(self.identity.feature_names)}"
            )
        return activation

    def vector(self, example_id: str) -> np.ndarray:
        return self.projector.vector(self.activation(example_id))


@dataclass(frozen=True, slots=True)
class AdapterPairedExampleSpec:
    example_id: str
    semantic_trial_id: str
    split: CorrespondenceSplit | str
    partition_id: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.example_id or not self.semantic_trial_id or not self.partition_id:
            raise ValueError("paired example IDs and partition_id must be non-empty")
        object.__setattr__(self, "split", CorrespondenceSplit(self.split))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


def build_adapter_feature_pair_examples(
    source: AdapterFeatureSpaceView,
    target: AdapterFeatureSpaceView,
    examples: Sequence[AdapterPairedExampleSpec],
) -> tuple[FeaturePairExample, ...]:
    """Capture paired feature vectors without serializing raw model inputs."""

    return tuple(
        FeaturePairExample(
            example_id=item.example_id,
            semantic_trial_id=item.semantic_trial_id,
            split=item.split,
            partition_id=item.partition_id,
            source_activation=source.vector(item.example_id),
            target_activation=target.vector(item.example_id),
            metadata=dict(item.metadata),
        )
        for item in examples
    )


class AdapterCausalSubstitutionEvaluator:
    """Execute source ablation and target substitution through ModelAdapter."""

    def __init__(
        self,
        source: AdapterFeatureSpaceView,
        target: AdapterFeatureSpaceView,
    ) -> None:
        self.source = source
        self.target = target
        self._activation_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._clean_metric_cache: dict[tuple[str, str], float] = {}

    @staticmethod
    def _indices(identity: FeatureSpaceIdentity, features: Sequence[str]) -> tuple[int, ...]:
        lookup = {name: index for index, name in enumerate(identity.feature_names)}
        missing = sorted(set(features) - set(lookup))
        if missing:
            raise ValueError(f"unknown selected feature(s): {missing}")
        return tuple(lookup[name] for name in features)

    def _activation(
        self,
        side: str,
        view: AdapterFeatureSpaceView,
        example_id: str,
    ) -> torch.Tensor:
        key = (side, example_id)
        if key not in self._activation_cache:
            self._activation_cache[key] = view.activation(example_id).detach().clone()
        return self._activation_cache[key].detach().clone()

    def _clean_metric(
        self,
        side: str,
        view: AdapterFeatureSpaceView,
        example_id: str,
    ) -> float:
        key = (side, example_id)
        if key not in self._clean_metric_cache:
            output = view.adapter.forward(view.inputs[example_id])
            self._clean_metric_cache[key] = float(view.metric(output))
        return self._clean_metric_cache[key]

    @staticmethod
    def _metric_with_replacement(
        view: AdapterFeatureSpaceView,
        example_id: str,
        replacement: torch.Tensor,
    ) -> float:
        output = view.adapter.forward_with_replacements(
            view.inputs[example_id],
            {view.path: replacement},
        )
        return float(view.metric(output))

    def __call__(
        self,
        *,
        target_example_id: str,
        source_example_id: str,
        source_features: tuple[str, ...],
        target_features: tuple[str, ...],
        replacement_values: np.ndarray,
    ) -> CausalSubstitutionMetrics:
        source_activation = self._activation("source", self.source, source_example_id)
        target_activation = self._activation("target", self.target, target_example_id)
        source_indices = self._indices(self.source.identity, source_features)
        target_indices = self._indices(self.target.identity, target_features)
        source_ablated = self.source.projector.ablate(source_activation, source_indices)
        target_ablated = self.target.projector.ablate(target_activation, target_indices)
        target_substituted = self.target.projector.replace(
            target_activation,
            target_indices,
            replacement_values,
        )
        return CausalSubstitutionMetrics(
            source_clean_metric=self._clean_metric("source", self.source, source_example_id),
            source_ablated_metric=self._metric_with_replacement(
                self.source,
                source_example_id,
                source_ablated,
            ),
            target_clean_metric=self._clean_metric("target", self.target, target_example_id),
            target_ablated_metric=self._metric_with_replacement(
                self.target,
                target_example_id,
                target_ablated,
            ),
            target_substituted_metric=self._metric_with_replacement(
                self.target,
                target_example_id,
                target_substituted,
            ),
        )


def _model_fingerprint(adapter: ModelAdapter) -> str | None:
    payload = adapter.model_fingerprint_payload()
    return None if payload is None else stable_hash(payload)


def run_adapter_feature_correspondence_study(
    spec: FeatureCorrespondenceSpec,
    *,
    source: AdapterFeatureSpaceView,
    target: AdapterFeatureSpaceView,
    examples: Sequence[AdapterPairedExampleSpec],
) -> FeatureCorrespondenceResult:
    """Run v0.8 correspondence with ModelAdapter mutation guards."""

    if source.identity != spec.source_space or target.identity != spec.target_space:
        raise ValueError("adapter feature-space identities must match the correspondence spec")
    source_before = _model_fingerprint(source.adapter)
    target_before = _model_fingerprint(target.adapter)
    paired = build_adapter_feature_pair_examples(source, target, examples)
    evaluator = AdapterCausalSubstitutionEvaluator(source, target)
    result = run_feature_correspondence_study(spec, paired, evaluator=evaluator)
    source_after = _model_fingerprint(source.adapter)
    target_after = _model_fingerprint(target.adapter)
    if source_before is not None and source_before != source_after:
        raise RuntimeError("source model mutated during correspondence study")
    if target_before is not None and target_before != target_after:
        raise RuntimeError("target model mutated during correspondence study")
    return result


def factorial_origin_from_report(
    report: FactorialMechanismReport,
    contrast_id: str,
) -> FactorialCorrespondenceOrigin:
    """Create a typed v0.7 provenance link only from an estimable contrast."""

    matches = [item for item in report.contrasts if item.contrast_id == contrast_id]
    if len(matches) != 1:
        raise ValueError(f"factorial report does not contain unique contrast {contrast_id!r}")
    contrast = matches[0]
    if not contrast.estimable:
        raise ValueError(
            f"cannot launch correspondence from non-estimable contrast {contrast_id!r}: "
            f"{list(contrast.reasons)}"
        )
    return FactorialCorrespondenceOrigin(
        factorial_study_fingerprint=report.study_fingerprint,
        contrast_id=contrast.contrast_id,
        cell_ids=contrast.cell_ids,
    )
