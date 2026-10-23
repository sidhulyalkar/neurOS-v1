"""Scientific experiment kernel for mechanistic interpretability."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field, is_dataclass
import math
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
        strict_execution: bool = False,
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
        self.strict_execution = bool(strict_execution)

    def run(
        self,
        interventions: Iterable[Intervention],
        *,
        controls: Iterable[Intervention] = (),
    ) -> ExperimentResult:
        """Run without consuming caller RNG state, including when evaluation fails."""
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(self.seed)
            return self._run_seeded(interventions, controls=controls)

    def _run_seeded(
        self,
        interventions: Iterable[Intervention],
        *,
        controls: Iterable[Intervention] = (),
    ) -> ExperimentResult:
        all_interventions = list(interventions)
        all_controls = list(controls)
        if self.strict_execution:
            model = getattr(self.adapter, "model", None)
            if not isinstance(model, torch.nn.Module):
                raise TypeError("strict_execution currently requires a PyTorch-backed adapter")
            if model.training:
                raise ValueError("strict_execution requires model.eval() before auditing")
            if any(not is_dataclass(item) for item in [*all_interventions, *all_controls]):
                raise TypeError("strict_execution requires dataclass intervention specifications")

        before_state_hash = (
            stable_hash_or_none(self.adapter.model_fingerprint_payload())
            if self.strict_execution else None
        )

        def _metric(output: torch.Tensor) -> float:
            score = float(self.metric(output))
            if self.strict_execution and not math.isfinite(score):
                raise ValueError("strict_execution rejects non-finite task metrics")
            return score

        clean_output = self.adapter.forward(self.pair.clean)
        corrupted_output = self.adapter.forward(self.pair.corrupted)
        clean_metric = _metric(clean_output)
        corrupted_metric = _metric(corrupted_output)
        if self.strict_execution:
            replay = self.adapter.forward(self.pair.corrupted)
            if not torch.equal(corrupted_output.detach(), replay.detach()):
                raise ValueError("strict_execution detected a non-deterministic baseline")

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
            if self.strict_execution:
                replay = self.adapter.forward_with_replacements(
                    self.pair.corrupted,
                    {path: replacement},
                )
                if not torch.equal(output.detach(), replay.detach()):
                    raise ValueError(
                        f"strict_execution detected a non-deterministic intervention: {intervention.name}"
                    )
            return InterventionEffect.from_metrics(
                name=intervention.name,
                component=intervention.component.label,
                clean_metric=clean_metric,
                corrupted_metric=corrupted_metric,
                intervened_metric=_metric(output),
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
        if self.strict_execution:
            if dataset_hash is None or model_hash is None:
                raise ValueError("strict_execution requires content-hashable data and model")
            if before_state_hash != model_hash:
                raise ValueError("strict_execution detected model-state mutation")

        parameters: dict[str, Any] = {
            "metric": self.metric.name,
            "pair_metadata": dict(self.pair.metadata),
            "interventions": [effect.name for effect in effects],
            "controls": [effect.name for effect in control_effects],
        }
        if self.strict_execution:
            def _identity(item: Intervention) -> dict[str, Any]:
                return {
                    "type": f"{type(item).__module__}.{type(item).__qualname__}",
                    "parameters": asdict(item),
                }

            parameters["strict_execution"] = True
            parameters["intervention_protocol"] = [_identity(i) for i in all_interventions]
            parameters["control_protocol"] = [_identity(i) for i in all_controls]

        manifest = ExperimentManifest(
            experiment_name=self.experiment_name,
            method="mechanistic_experiment",
            model_id=self.model_id,
            model_hash=model_hash,
            dataset_id=self.dataset_id,
            dataset_hash=dataset_hash,
            method_version="2" if self.strict_execution else "1",
            parameters=parameters,
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
