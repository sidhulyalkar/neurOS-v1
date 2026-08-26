"""Metamorphic conformance and adversarial counterexample search.

Synthetic physiology need not be a perfect human model to verify many systems
invariants. Arena therefore supports paired-world properties such as:

- increasing transport drop probability cannot increase packets delivered;
- increasing display drop probability cannot reduce the set of dropped frames
  when the same deterministic random stream is used;
- application-defined authority should fail closed under declared degradations.

When a property fails, the resolved manifests are retained as portable
counterexamples rather than summarized away into a single score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np

from .manifest import ArenaManifest
from .population import ParameterDistribution, PopulationSpec, _replace_manifest_value
from .runner import ArenaRun, run_scenario


@dataclass(frozen=True)
class MetamorphicResult:
    name: str
    passed: bool
    base_value: float
    mutated_value: float
    relation: str
    detail: str
    base_manifest: dict
    mutated_manifest: dict

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "passed": self.passed,
            "base_value": self.base_value,
            "mutated_value": self.mutated_value,
            "relation": self.relation,
            "detail": self.detail,
            "base_manifest": self.base_manifest,
            "mutated_manifest": self.mutated_manifest,
        }


@dataclass(frozen=True)
class Counterexample:
    rank: int
    objective: float
    sampled: dict[str, float]
    manifest: dict
    metrics: dict[str, float]


@dataclass(frozen=True)
class AdversarialSearchResult:
    objective_name: str
    minimize: bool
    evaluated: int
    counterexamples: tuple[Counterexample, ...]

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.synthetic_bci_arena.counterexamples.v1",
            "objective_name": self.objective_name,
            "minimize": self.minimize,
            "evaluated": self.evaluated,
            "counterexamples": [asdict(item) for item in self.counterexamples],
            "evidence_boundary": "Counterexamples are failures within the declared synthetic envelope, not estimates of human failure prevalence.",
        }


def _run(manifest: ArenaManifest) -> ArenaRun:
    manifest.validate()
    return run_scenario(
        manifest.scenario,
        manifest.participant,
        manifest.device,
        manifest.display,
        manifest.transport,
        manifest.world_model,
    )


def _paired_result(
    name: str,
    base: ArenaManifest,
    mutated: ArenaManifest,
    base_value: float,
    mutated_value: float,
    *,
    relation: str,
    passed: bool,
    detail: str,
) -> MetamorphicResult:
    return MetamorphicResult(
        name=name,
        passed=bool(passed),
        base_value=float(base_value),
        mutated_value=float(mutated_value),
        relation=relation,
        detail=detail,
        base_manifest=base.to_dict(),
        mutated_manifest=mutated.to_dict(),
    )


def check_transport_drop_monotonicity(
    manifest: ArenaManifest,
    *,
    higher_drop_probability: float,
) -> MetamorphicResult:
    """Verify that a nested deterministic drop mask cannot deliver more packets."""
    base_p = manifest.transport.drop_probability
    if higher_drop_probability < base_p or higher_drop_probability >= 1.0:
        raise ValueError("higher_drop_probability must be >= base and < 1")
    mutated = replace(
        manifest,
        transport=replace(manifest.transport, drop_probability=float(higher_drop_probability)),
    )
    first = _run(manifest)
    second = _run(mutated)
    a = float(first.report["metrics"]["transport"]["packets_delivered"])
    b = float(second.report["metrics"]["transport"]["packets_delivered"])
    return _paired_result(
        "transport_drop_monotonicity",
        manifest,
        mutated,
        a,
        b,
        relation="mutated <= base",
        passed=b <= a,
        detail="Higher drop probability uses the same deterministic random stream, so delivered packet count must not increase.",
    )


def check_display_drop_monotonicity(
    manifest: ArenaManifest,
    *,
    higher_drop_probability: float,
) -> MetamorphicResult:
    """Verify deterministic nested frame-drop masks at the display layer."""
    base_p = manifest.display.frame_drop_probability
    if higher_drop_probability < base_p or higher_drop_probability >= 1.0:
        raise ValueError("higher_drop_probability must be >= base and < 1")
    mutated = replace(
        manifest,
        display=replace(manifest.display, frame_drop_probability=float(higher_drop_probability)),
    )
    first = _run(manifest)
    second = _run(mutated)
    a = max((trace.frame_drop_fraction for trace in first.stimulus_traces), default=0.0)
    b = max((trace.frame_drop_fraction for trace in second.stimulus_traces), default=0.0)
    return _paired_result(
        "display_drop_monotonicity",
        manifest,
        mutated,
        a,
        b,
        relation="mutated >= base",
        passed=b + 1e-12 >= a,
        detail="Higher frame-drop probability uses the same per-stage seed, so the dropped-frame set must be a superset.",
    )


def check_fail_closed_degradation(
    manifest: ArenaManifest,
    mutated: ArenaManifest,
    evaluator: Callable[[ArenaRun], float],
    *,
    name: str = "application_fail_closed",
    tolerance: float = 0.0,
) -> MetamorphicResult:
    """Check an application-defined authority metric under a declared degradation.

    ``evaluator`` should return a metric where *larger means more gameplay or
    control authority*. The property asserts that the degraded world does not
    gain authority beyond the configured tolerance. This deliberately leaves
    task semantics outside Arena.
    """
    base_run = _run(manifest)
    mutated_run = _run(mutated)
    base_value = float(evaluator(base_run))
    mutated_value = float(evaluator(mutated_run))
    return _paired_result(
        name,
        manifest,
        mutated,
        base_value,
        mutated_value,
        relation=f"mutated <= base + {float(tolerance):g}",
        passed=mutated_value <= base_value + float(tolerance),
        detail="Application-defined authority should not increase when the caller declares the second world to be a degradation.",
    )


def search_counterexamples(
    manifest: ArenaManifest,
    spec: PopulationSpec,
    evaluator: Callable[[ArenaRun], tuple[float, dict[str, float]]],
    *,
    objective_name: str,
    minimize: bool = True,
    top_k: int = 10,
) -> AdversarialSearchResult:
    """Search a parameter envelope and preserve the worst resolved worlds.

    The sampler is intentionally deterministic and simple in v0.1. More advanced
    optimizers can be added later without changing the counterexample artifact.
    ``evaluator`` returns ``(objective, metrics)`` for one complete run.
    """
    manifest.validate()
    spec.validate()
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    rng = np.random.default_rng(spec.seed)
    candidates: list[tuple[float, dict[str, float], ArenaManifest, dict[str, float]]] = []
    for index in range(spec.size):
        world = manifest
        sampled: dict[str, float] = {}
        for distribution in spec.parameters:
            value = distribution.sample(rng)
            sampled[distribution.path] = value
            world = _replace_manifest_value(world, distribution.path, value)
        world = replace(
            world,
            participant=replace(
                world.participant,
                seed=world.participant.seed + spec.seed + index * 7919,
            ),
        )
        objective, metrics = evaluator(_run(world))
        if not np.isfinite(objective):
            continue
        candidates.append((float(objective), sampled, world, {k: float(v) for k, v in metrics.items()}))
    candidates.sort(key=lambda item: item[0], reverse=not minimize)
    selected = candidates[: min(top_k, len(candidates))]
    counterexamples = tuple(
        Counterexample(
            rank=rank + 1,
            objective=objective,
            sampled=sampled,
            manifest=world.to_dict(),
            metrics=metrics,
        )
        for rank, (objective, sampled, world, metrics) in enumerate(selected)
    )
    return AdversarialSearchResult(
        objective_name=objective_name,
        minimize=bool(minimize),
        evaluated=len(candidates),
        counterexamples=counterexamples,
    )
