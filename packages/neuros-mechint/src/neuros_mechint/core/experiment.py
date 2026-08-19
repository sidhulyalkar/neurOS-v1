"""Scientific experiment kernel for mechanistic interpretability."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from neuros_mechint.adapters.base import ModelAdapter

from .evidence import EvidenceTier
from .interventions import Intervention
from .manifest import ExperimentManifest, stable_hash_or_none
from .metrics import ScalarMetric
from .results import ExperimentResult, InterventionEffect


@dataclass(frozen=True, slots=True)
class CounterfactualPair:
    """Clean and corrupted inputs defining the behavior to explain."""

    clean: Any
    corrupted: Any
    metadata: Mapping[str, Any] = field(default_factory=dict)


class MechanisticExperiment:
    """Run independent internal interventions against a fixed counterfactual pair."""

    def __init__(
        self,
        *,
        adapter: ModelAdapter,
        pair: CounterfactualPair,
        metric: ScalarMetric,
        experiment_name: str,
        model_id: str,
        dataset_id: str = "in_memory",
        seed: int = 0,
        evidence_tier: EvidenceTier = EvidenceTier.UNIT,
        git_sha: str | None = None,
    ) -> None:
        self.adapter = adapter
        self.pair = pair
        self.metric = metric
        self.experiment_name = experiment_name
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.seed = seed
        self.evidence_tier = EvidenceTier.coerce(evidence_tier)
        self.git_sha = git_sha

    def run(
        self,
        interventions: Iterable[Intervention],
        *,
        controls: Iterable[Intervention] = (),
    ) -> ExperimentResult:
        torch.manual_seed(self.seed)
        clean_output = self.adapter.forward(self.pair.clean)
        corrupted_output = self.adapter.forward(self.pair.corrupted)
        clean_metric = self.metric(clean_output)
        corrupted_metric = self.metric(corrupted_output)

        all_interventions = list(interventions)
        all_controls = list(controls)
        required_paths = sorted(
            {item.component.path for item in [*all_interventions, *all_controls]}
        )
        clean_cache = self.adapter.capture_outputs(self.pair.clean, required_paths)
        corrupted_cache = self.adapter.capture_outputs(self.pair.corrupted, required_paths)

        def _evaluate(intervention: Intervention) -> InterventionEffect:
            path = intervention.component.path
            replacement = intervention.replacement(
                clean_value=clean_cache[path],
                corrupted_value=corrupted_cache[path],
            )
            output = self.adapter.forward_with_replacements(
                self.pair.corrupted,
                {path: replacement},
            )
            return InterventionEffect.from_metrics(
                name=intervention.name,
                component=intervention.component.label,
                clean_metric=clean_metric,
                corrupted_metric=corrupted_metric,
                intervened_metric=self.metric(output),
                metadata=intervention.metadata(),
            )

        effects = [_evaluate(intervention) for intervention in all_interventions]
        control_effects = [_evaluate(intervention) for intervention in all_controls]
        dataset_hash = stable_hash_or_none(
            {
                "clean": self.pair.clean,
                "corrupted": self.pair.corrupted,
                "metadata": dict(self.pair.metadata),
            }
        )
        model_hash = stable_hash_or_none(self.adapter.model_fingerprint_payload())

        manifest = ExperimentManifest(
            experiment_name=self.experiment_name,
            method="mechanistic_experiment",
            model_id=self.model_id,
            model_hash=model_hash,
            dataset_id=self.dataset_id,
            dataset_hash=dataset_hash,
            parameters={
                "metric": self.metric.name,
                "pair_metadata": dict(self.pair.metadata),
                "interventions": [effect.name for effect in effects],
                "controls": [effect.name for effect in control_effects],
            },
            seed=self.seed,
            evidence_tier=self.evidence_tier,
            git_sha=self.git_sha,
        )
        return ExperimentResult(
            manifest=manifest,
            metric_name=self.metric.name,
            clean_metric=clean_metric,
            corrupted_metric=corrupted_metric,
            effects=effects,
            controls=control_effects,
        )
