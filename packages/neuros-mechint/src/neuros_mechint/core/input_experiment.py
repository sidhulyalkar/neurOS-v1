"""Framework-agnostic causal audits that intervene on model inputs."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
import torch

from .evidence import EvidenceTier
from .manifest import ExperimentManifest, stable_hash_or_none
from .results import InputExperimentResult, InputInterventionEffect


class InputIntervention(Protocol):
    """An independent edit applied to one immutable reference input."""

    name: str
    target: str

    def apply(self, reference: Any) -> Any:
        ...

    def metadata(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class InputMetric:
    """Wrap a callable that maps an input directly to a scalar score."""

    fn: Callable[[Any], Any]
    name: str = "input_metric"

    def __call__(self, value: Any) -> float:
        score = self.fn(value)
        if isinstance(score, torch.Tensor):
            if score.numel() != 1:
                raise ValueError(
                    f"metric {self.name!r} must return a scalar, got {tuple(score.shape)}"
                )
            score = score.detach().cpu().item()
        elif isinstance(score, np.ndarray):
            if score.size != 1:
                raise ValueError(
                    f"metric {self.name!r} must return a scalar, got {score.shape}"
                )
            score = score.item()
        return float(score)


class InputCausalExperiment:
    """Measure effects of independent edits to token, signal, or representation inputs.

    This complements module-level tracing. It is especially useful for ORION:
    token windows, token types, or side features can be intervened on before a
    neural encoder/decoder without pretending those inputs are internal model
    components.
    """

    def __init__(
        self,
        *,
        reference: Any,
        metric: InputMetric,
        experiment_name: str,
        model_id: str,
        dataset_id: str = "in_memory",
        seed: int = 0,
        evidence_tier: EvidenceTier = EvidenceTier.UNIT,
        git_sha: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.reference = reference
        self.metric = metric
        self.experiment_name = experiment_name
        self.model_id = model_id
        self.dataset_id = dataset_id
        self.seed = seed
        self.evidence_tier = EvidenceTier.coerce(evidence_tier)
        self.git_sha = git_sha
        self.metadata = dict(metadata or {})

    def run(
        self,
        interventions: Iterable[InputIntervention],
        *,
        controls: Iterable[InputIntervention] = (),
    ) -> InputExperimentResult:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        baseline_metric = self.metric(self.reference)
        interventions = list(interventions)
        controls = list(controls)

        def _evaluate(intervention: InputIntervention) -> InputInterventionEffect:
            edited = intervention.apply(self.reference)
            return InputInterventionEffect.from_metrics(
                name=intervention.name,
                target=intervention.target,
                baseline_metric=baseline_metric,
                intervened_metric=self.metric(edited),
                metadata=intervention.metadata(),
            )

        effects = [_evaluate(item) for item in interventions]
        control_effects = [_evaluate(item) for item in controls]
        dataset_hash = stable_hash_or_none(self.reference)
        manifest = ExperimentManifest(
            experiment_name=self.experiment_name,
            method="input_causal_audit",
            model_id=self.model_id,
            dataset_id=self.dataset_id,
            dataset_hash=dataset_hash,
            parameters={
                "metric": self.metric.name,
                "metadata": self.metadata,
                "interventions": [item.name for item in interventions],
                "controls": [item.name for item in controls],
            },
            seed=self.seed,
            evidence_tier=self.evidence_tier,
            git_sha=self.git_sha,
        )
        return InputExperimentResult(
            manifest=manifest,
            metric_name=self.metric.name,
            baseline_metric=baseline_metric,
            effects=effects,
            controls=control_effects,
        )
