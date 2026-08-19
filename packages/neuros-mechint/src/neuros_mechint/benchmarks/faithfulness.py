"""Quantitative circuit faithfulness with equal-cardinality random controls."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from types import MappingProxyType
from typing import Any

import numpy as np
import torch

from neuros_mechint.adapters.base import ModelAdapter
from neuros_mechint.core.metrics import ScalarMetric


@dataclass(frozen=True, slots=True)
class CircuitCandidate:
    """A proposed mechanism over named intervention targets."""

    name: str
    targets: tuple[str, ...]
    scores: Mapping[str, float] = field(default_factory=dict)
    source: str = "manual"

    def __post_init__(self) -> None:
        targets = tuple(dict.fromkeys(str(item) for item in self.targets))
        if not self.name:
            raise ValueError("candidate name must be non-empty")
        if not targets:
            raise ValueError("candidate targets must not be empty")
        object.__setattr__(self, "targets", targets)
        object.__setattr__(
            self,
            "scores",
            MappingProxyType({str(key): float(value) for key, value in self.scores.items()}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "targets": list(self.targets),
            "scores": dict(self.scores),
            "source": self.source,
        }


@dataclass(frozen=True, slots=True)
class FaithfulnessPolicy:
    """Explicit thresholds for promoting a circuit-faithfulness result."""

    min_sufficiency_fraction: float = 0.80
    min_necessity_fraction: float = 0.50
    min_random_percentile: float = 0.95

    def __post_init__(self) -> None:
        for name in (
            "min_sufficiency_fraction",
            "min_necessity_fraction",
            "min_random_percentile",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class RandomCircuitControl:
    targets: tuple[str, ...]
    sufficiency_fraction: float
    necessity_fraction: float

    @property
    def joint_faithfulness(self) -> float:
        return min(self.sufficiency_fraction, self.necessity_fraction)

    def to_dict(self) -> dict[str, Any]:
        return {
            "joint_faithfulness": self.joint_faithfulness,
            "necessity_fraction": self.necessity_fraction,
            "sufficiency_fraction": self.sufficiency_fraction,
            "targets": list(self.targets),
        }


@dataclass(frozen=True, slots=True)
class CircuitFaithfulnessReport:
    """Necessity/sufficiency evidence for one proposed circuit."""

    candidate: CircuitCandidate
    all_targets: tuple[str, ...]
    baseline_metric: float
    null_metric: float
    circuit_metric: float
    complement_metric: float
    sufficiency_fraction: float
    necessity_fraction: float
    random_controls: tuple[RandomCircuitControl, ...]
    sufficiency_random_percentile: float | None
    necessity_random_percentile: float | None
    joint_random_percentile: float | None
    policy: FaithfulnessPolicy
    higher_is_better: bool
    seed: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def joint_faithfulness(self) -> float:
        return min(self.sufficiency_fraction, self.necessity_fraction)

    @property
    def passed(self) -> bool:
        if self.joint_random_percentile is None:
            return False
        return (
            self.sufficiency_fraction >= self.policy.min_sufficiency_fraction
            and self.necessity_fraction >= self.policy.min_necessity_fraction
            and self.joint_random_percentile >= self.policy.min_random_percentile
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "all_targets": list(self.all_targets),
            "baseline_metric": self.baseline_metric,
            "candidate": self.candidate.to_dict(),
            "circuit_metric": self.circuit_metric,
            "complement_metric": self.complement_metric,
            "higher_is_better": self.higher_is_better,
            "joint_faithfulness": self.joint_faithfulness,
            "joint_random_percentile": self.joint_random_percentile,
            "metadata": dict(self.metadata),
            "necessity_fraction": self.necessity_fraction,
            "necessity_random_percentile": self.necessity_random_percentile,
            "null_metric": self.null_metric,
            "passed": self.passed,
            "policy": self.policy.to_dict(),
            "random_controls": [item.to_dict() for item in self.random_controls],
            "seed": self.seed,
            "sufficiency_fraction": self.sufficiency_fraction,
            "sufficiency_random_percentile": self.sufficiency_random_percentile,
        }


def _orient(value: float, *, higher_is_better: bool) -> float:
    return float(value) if higher_is_better else -float(value)


def _normalized_scores(
    *,
    baseline: float,
    null: float,
    subset: float,
    complement: float,
    higher_is_better: bool,
) -> tuple[float, float]:
    base = _orient(baseline, higher_is_better=higher_is_better)
    empty = _orient(null, higher_is_better=higher_is_better)
    kept = _orient(subset, higher_is_better=higher_is_better)
    removed = _orient(complement, higher_is_better=higher_is_better)
    span = base - empty
    if abs(span) <= 1e-12:
        raise ValueError(
            "all-target and null metrics are indistinguishable; faithfulness cannot be normalized"
        )
    if span < 0.0:
        raise ValueError(
            "null intervention outperforms the all-target baseline under the selected metric direction"
        )
    sufficiency = (kept - empty) / span
    necessity = (base - removed) / span
    return float(sufficiency), float(necessity)


def _random_target_sets(
    all_targets: tuple[str, ...],
    candidate_targets: tuple[str, ...],
    *,
    random_trials: int,
    seed: int,
) -> tuple[tuple[str, ...], ...]:
    if random_trials <= 0:
        raise ValueError("random_trials must be positive")
    size = len(candidate_targets)
    if size > len(all_targets):
        raise ValueError("candidate cannot contain more targets than the target universe")
    candidate_set = frozenset(candidate_targets)
    total_combinations = 1
    for numerator, denominator in zip(
        range(len(all_targets) - size + 1, len(all_targets) + 1),
        range(1, size + 1),
        strict=True,
    ):
        total_combinations = total_combinations * numerator // denominator

    if total_combinations <= random_trials + 1:
        return tuple(
            tuple(items)
            for items in combinations(all_targets, size)
            if frozenset(items) != candidate_set
        )

    rng = np.random.default_rng(seed)
    sampled: set[tuple[str, ...]] = set()
    indices = np.arange(len(all_targets))
    max_attempts = max(100, random_trials * 50)
    attempts = 0
    while len(sampled) < random_trials and attempts < max_attempts:
        attempts += 1
        chosen = tuple(sorted(int(i) for i in rng.choice(indices, size=size, replace=False)))
        targets = tuple(all_targets[index] for index in chosen)
        if frozenset(targets) != candidate_set:
            sampled.add(targets)
    return tuple(sorted(sampled))


def _strict_percentile(candidate: float, controls: Sequence[float]) -> float | None:
    """Fraction of controls strictly worse than the candidate."""

    if not controls:
        return None
    values = np.asarray(controls, dtype=np.float64)
    return float(np.mean(values < float(candidate)))


def evaluate_circuit_faithfulness(
    *,
    all_targets: Sequence[str],
    candidate: CircuitCandidate,
    subset_metric: Callable[[tuple[str, ...]], float],
    random_trials: int = 100,
    seed: int = 0,
    higher_is_better: bool = True,
    policy: FaithfulnessPolicy | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> CircuitFaithfulnessReport:
    """Measure circuit sufficiency and necessity against random same-size circuits.

    ``subset_metric(targets)`` returns the metric obtained when only the named
    targets are retained. Subset metrics are memoized by canonical target tuple
    so large-model evaluations do not repeat identical forward passes.
    """

    universe = tuple(dict.fromkeys(str(item) for item in all_targets))
    if not universe:
        raise ValueError("all_targets must not be empty")
    universe_set = set(universe)
    candidate_set = set(candidate.targets)
    missing = [target for target in candidate.targets if target not in universe_set]
    if missing:
        raise ValueError(f"candidate targets are not in all_targets: {missing}")

    metric_cache: dict[tuple[str, ...], float] = {}

    def _evaluate(retained: Sequence[str]) -> float:
        retained_set = set(retained)
        canonical = tuple(target for target in universe if target in retained_set)
        if canonical not in metric_cache:
            metric_cache[canonical] = float(subset_metric(canonical))
        return metric_cache[canonical]

    candidate_targets = tuple(target for target in universe if target in candidate_set)
    complement_targets = tuple(target for target in universe if target not in candidate_set)
    baseline_metric = _evaluate(universe)
    null_metric = _evaluate(())
    circuit_metric = _evaluate(candidate_targets)
    complement_metric = _evaluate(complement_targets)
    sufficiency, necessity = _normalized_scores(
        baseline=baseline_metric,
        null=null_metric,
        subset=circuit_metric,
        complement=complement_metric,
        higher_is_better=higher_is_better,
    )

    random_controls = []
    for targets in _random_target_sets(
        universe,
        candidate_targets,
        random_trials=random_trials,
        seed=seed,
    ):
        random_set = set(targets)
        random_complement = tuple(target for target in universe if target not in random_set)
        random_metric = _evaluate(targets)
        random_complement_metric = _evaluate(random_complement)
        random_sufficiency, random_necessity = _normalized_scores(
            baseline=baseline_metric,
            null=null_metric,
            subset=random_metric,
            complement=random_complement_metric,
            higher_is_better=higher_is_better,
        )
        random_controls.append(
            RandomCircuitControl(
                targets=targets,
                sufficiency_fraction=random_sufficiency,
                necessity_fraction=random_necessity,
            )
        )

    policy = policy or FaithfulnessPolicy()
    joint = min(sufficiency, necessity)
    return CircuitFaithfulnessReport(
        candidate=candidate,
        all_targets=universe,
        baseline_metric=baseline_metric,
        null_metric=null_metric,
        circuit_metric=circuit_metric,
        complement_metric=complement_metric,
        sufficiency_fraction=sufficiency,
        necessity_fraction=necessity,
        random_controls=tuple(random_controls),
        sufficiency_random_percentile=_strict_percentile(
            sufficiency,
            [item.sufficiency_fraction for item in random_controls],
        ),
        necessity_random_percentile=_strict_percentile(
            necessity,
            [item.necessity_fraction for item in random_controls],
        ),
        joint_random_percentile=_strict_percentile(
            joint,
            [item.joint_faithfulness for item in random_controls],
        ),
        policy=policy,
        higher_is_better=higher_is_better,
        seed=seed,
        metadata=dict(metadata or {}),
    )


def _ablation_value(
    value: torch.Tensor,
    mode: str,
    *,
    reference: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "zero":
        return torch.zeros_like(value)
    if mode != "mean":
        raise ValueError("ablation_mode must be 'zero' or 'mean'")

    if reference is None:
        scalar = value.detach().mean()
    else:
        if not isinstance(reference, torch.Tensor) or reference.numel() != 1:
            raise ValueError("mean ablation reference must be a scalar tensor")
        scalar = reference.detach().reshape(())
    scalar = scalar.to(device=value.device, dtype=value.dtype)
    return torch.ones_like(value) * scalar


def evaluate_adapter_circuit_faithfulness(
    *,
    adapter: ModelAdapter,
    inputs: Any,
    metric: ScalarMetric,
    all_targets: Sequence[str],
    candidate: CircuitCandidate,
    ablation_mode: str = "zero",
    ablation_references: Mapping[str, torch.Tensor] | None = None,
    random_trials: int = 100,
    seed: int = 0,
    higher_is_better: bool = True,
    policy: FaithfulnessPolicy | None = None,
) -> CircuitFaithfulnessReport:
    """Evaluate a circuit over any ``ModelAdapter`` using output interventions.

    ``mean`` replacement can use scalar per-target references fitted on a
    discovery corpus. Without explicit references the current example's global
    activation mean is used. Evidence-pack studies should pass frozen discovery
    references so held-out examples do not define their own intervention donor.
    """

    universe = tuple(dict.fromkeys(str(item) for item in all_targets))
    captured = adapter.capture_outputs(inputs, universe)
    references = dict(ablation_references or {})
    if ablation_mode == "mean":
        missing = [target for target in references if target not in universe]
        if missing:
            raise ValueError(f"ablation reference target(s) outside universe: {missing}")
    ablations = {
        target: _ablation_value(
            captured[target],
            ablation_mode,
            reference=references.get(target),
        )
        for target in universe
    }

    def _metric_for_subset(retained: tuple[str, ...]) -> float:
        retained_set = set(retained)
        replacements = {
            target: replacement
            for target, replacement in ablations.items()
            if target not in retained_set
        }
        if replacements:
            output = adapter.forward_with_replacements(inputs, replacements)
        else:
            output = adapter.forward(inputs)
        return float(metric(output))

    return evaluate_circuit_faithfulness(
        all_targets=universe,
        candidate=candidate,
        subset_metric=_metric_for_subset,
        random_trials=random_trials,
        seed=seed,
        higher_is_better=higher_is_better,
        policy=policy,
        metadata={
            "ablation_mode": ablation_mode,
            "adapter": type(adapter).__qualname__,
            "metric": metric.name,
            "uses_external_mean_reference": bool(ablation_references),
        },
    )
