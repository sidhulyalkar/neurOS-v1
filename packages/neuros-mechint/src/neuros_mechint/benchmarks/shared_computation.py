"""Cross-context causal-map comparison for neural and artificial models.

The module is representation agnostic. Integrations produce named causal effect
maps; this layer compares them under explicit model/data context metadata and
turns sufficiently controlled patterns into falsifiable candidate hypotheses.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from itertools import combinations
from statistics import median
from types import MappingProxyType
from typing import Any

import numpy as np

from .stability import EffectMapStability, compare_effect_maps


@dataclass(frozen=True, slots=True)
class MechanismContext:
    """Identity of one model/data context in a comparative mechanism study."""

    context_id: str
    architecture: str
    dataset_id: str
    session_id: str
    subject_id: str | None = None
    checkpoint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("context_id", "architecture", "dataset_id", "session_id"):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def changed_axes(self, other: MechanismContext) -> tuple[str, ...]:
        axes = []
        for name in ("architecture", "dataset_id", "session_id", "subject_id", "checkpoint"):
            if getattr(self, name) != getattr(other, name):
                axes.append(name.removesuffix("_id"))
        return tuple(axes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "architecture": self.architecture,
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "checkpoint": self.checkpoint,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class CausalEffectRecord:
    """Causal intervention map, matched controls, and baseline for one context."""

    context: MechanismContext
    baseline_metric: float
    effect_map: Mapping[str, float]
    control_map: Mapping[str, float] = field(default_factory=dict)
    metric_name: str = "score"

    def __post_init__(self) -> None:
        effects = {str(key): float(value) for key, value in self.effect_map.items()}
        controls = {str(key): float(value) for key, value in self.control_map.items()}
        if not effects:
            raise ValueError("effect_map must not be empty")
        if not np.isfinite(np.asarray(list(effects.values()), dtype=np.float64)).all():
            raise ValueError("effect_map contains non-finite values")
        if controls and not np.isfinite(
            np.asarray(list(controls.values()), dtype=np.float64)
        ).all():
            raise ValueError("control_map contains non-finite values")
        object.__setattr__(self, "effect_map", MappingProxyType(effects))
        object.__setattr__(self, "control_map", MappingProxyType(controls))
        object.__setattr__(self, "baseline_metric", float(self.baseline_metric))

    @property
    def mean_absolute_effect(self) -> float:
        return float(np.mean(np.abs(np.asarray(list(self.effect_map.values())))))

    @property
    def mean_absolute_control(self) -> float | None:
        if not self.control_map:
            return None
        return float(np.mean(np.abs(np.asarray(list(self.control_map.values())))))

    @property
    def control_to_effect_ratio(self) -> float | None:
        control = self.mean_absolute_control
        effect = self.mean_absolute_effect
        if control is None or effect <= 1e-12:
            return None
        return control / effect

    def top_k_concentration(self, k: int) -> float:
        if k <= 0:
            raise ValueError("k must be positive")
        values = np.abs(np.asarray(list(self.effect_map.values()), dtype=np.float64))
        values = np.sort(values)[::-1]
        total = float(values.sum())
        if total <= 1e-12:
            return 0.0
        return float(values[: min(k, len(values))].sum() / total)

    def to_dict(self, *, top_k: int = 5) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "baseline_metric": self.baseline_metric,
            "metric_name": self.metric_name,
            "effect_map": dict(self.effect_map),
            "control_map": dict(self.control_map),
            "mean_absolute_effect": self.mean_absolute_effect,
            "mean_absolute_control": self.mean_absolute_control,
            "control_to_effect_ratio": self.control_to_effect_ratio,
            "top_k_concentration": self.top_k_concentration(top_k),
        }


@dataclass(frozen=True, slots=True)
class PairwiseMechanismComparison:
    left_context_id: str
    right_context_id: str
    left_architecture: str
    right_architecture: str
    axes_changed: tuple[str, ...]
    baseline_delta: float
    stability: EffectMapStability

    @property
    def same_architecture(self) -> bool:
        return self.left_architecture == self.right_architecture

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["axes_changed"] = list(self.axes_changed)
        payload["same_architecture"] = self.same_architecture
        payload["stability"] = self.stability.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class StabilityAggregate:
    pair_count: int
    median_pearson_r: float | None
    median_spearman_r: float | None
    median_sign_agreement: float
    median_top_k_jaccard: float
    median_shared_target_fraction: float
    median_mean_absolute_delta: float

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ArchitectureMechanismSummary:
    architecture: str
    context_count: int
    mean_baseline_metric: float
    baseline_metric_std: float
    mean_absolute_effect: float
    mean_control_to_effect_ratio: float | None
    mean_top_k_concentration: float
    within_architecture_stability: StabilityAggregate | None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["within_architecture_stability"] = (
            None
            if self.within_architecture_stability is None
            else self.within_architecture_stability.to_dict()
        )
        return payload


@dataclass(frozen=True, slots=True)
class ArchitectureComparisonReport:
    """Descriptive and one-factor-at-a-time causal stability summaries."""

    context_count: int
    architectures: tuple[str, ...]
    pairwise: tuple[PairwiseMechanismComparison, ...]
    architecture_summaries: Mapping[str, ArchitectureMechanismSummary]
    axis_stability: Mapping[str, StabilityAggregate]
    isolated_axis_stability: Mapping[str, StabilityAggregate]
    architecture_pair_stability: Mapping[str, StabilityAggregate]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_count": self.context_count,
            "architectures": list(self.architectures),
            "pairwise": [item.to_dict() for item in self.pairwise],
            "architecture_summaries": {
                name: item.to_dict() for name, item in self.architecture_summaries.items()
            },
            "axis_stability": {
                name: item.to_dict() for name, item in self.axis_stability.items()
            },
            "isolated_axis_stability": {
                name: item.to_dict() for name, item in self.isolated_axis_stability.items()
            },
            "architecture_pair_stability": {
                name: item.to_dict() for name, item in self.architecture_pair_stability.items()
            },
        }


@dataclass(frozen=True, slots=True)
class HypothesisPolicy:
    """Transparent thresholds for candidate-hypothesis prioritization."""

    min_pair_count: int = 2
    min_shared_targets: int = 3
    min_shared_target_fraction: float = 0.75
    stable_spearman: float = 0.70
    stable_sign_agreement: float = 0.75
    stable_top_k_jaccard: float = 0.50
    divergent_spearman: float = 0.30
    high_control_ratio: float = 0.50
    sparse_top_k_concentration: float = 0.70

    def __post_init__(self) -> None:
        if self.min_pair_count <= 0 or self.min_shared_targets <= 0:
            raise ValueError("minimum counts must be positive")
        bounded = (
            "min_shared_target_fraction",
            "stable_spearman",
            "stable_sign_agreement",
            "stable_top_k_jaccard",
            "divergent_spearman",
            "high_control_ratio",
            "sparse_top_k_concentration",
        )
        for name in bounded:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MechanisticHypothesis:
    """Falsifiable candidate interpretation, never an automatic conclusion."""

    hypothesis_id: str
    kind: str
    statement: str
    priority: str
    supporting_metrics: Mapping[str, float | int | None]
    context_ids: tuple[str, ...]
    falsification_tests: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "supporting_metrics",
            MappingProxyType(dict(self.supporting_metrics)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "kind": self.kind,
            "statement": self.statement,
            "priority": self.priority,
            "supporting_metrics": dict(self.supporting_metrics),
            "context_ids": list(self.context_ids),
            "falsification_tests": list(self.falsification_tests),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class SharedComputationAnalysis:
    comparison: ArchitectureComparisonReport
    hypotheses: tuple[MechanisticHypothesis, ...]
    policy: HypothesisPolicy

    def to_dict(self) -> dict[str, Any]:
        return {
            "comparison": self.comparison.to_dict(),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "policy": self.policy.to_dict(),
        }


def _median_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(median(finite))


def _mean_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(np.mean(finite))


def _aggregate(items: Sequence[PairwiseMechanismComparison]) -> StabilityAggregate:
    if not items:
        raise ValueError("cannot aggregate an empty comparison set")
    reports = [item.stability for item in items]
    return StabilityAggregate(
        pair_count=len(reports),
        median_pearson_r=_median_optional([item.pearson_r for item in reports]),
        median_spearman_r=_median_optional([item.spearman_r for item in reports]),
        median_sign_agreement=float(median(item.sign_agreement for item in reports)),
        median_top_k_jaccard=float(median(item.top_k_jaccard for item in reports)),
        median_shared_target_fraction=float(
            median(item.shared_target_fraction for item in reports)
        ),
        median_mean_absolute_delta=float(median(item.mean_absolute_delta for item in reports)),
    )


def _group_aggregate(
    groups: Mapping[str, Sequence[PairwiseMechanismComparison]],
) -> Mapping[str, StabilityAggregate]:
    return MappingProxyType(
        {name: _aggregate(items) for name, items in groups.items() if items}
    )


def compare_causal_records(
    records: Sequence[CausalEffectRecord],
    *,
    top_k: int = 5,
) -> ArchitectureComparisonReport:
    """Compare causal effect maps across all supplied contexts."""

    records = tuple(records)
    if len(records) < 2:
        raise ValueError("at least two causal-effect records are required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    context_ids = [item.context.context_id for item in records]
    if len(context_ids) != len(set(context_ids)):
        raise ValueError("context_id values must be unique")

    pairwise = []
    for left, right in combinations(records, 2):
        pairwise.append(
            PairwiseMechanismComparison(
                left_context_id=left.context.context_id,
                right_context_id=right.context.context_id,
                left_architecture=left.context.architecture,
                right_architecture=right.context.architecture,
                axes_changed=left.context.changed_axes(right.context),
                baseline_delta=right.baseline_metric - left.baseline_metric,
                stability=compare_effect_maps(left.effect_map, right.effect_map, top_k=top_k),
            )
        )

    architectures = sorted({item.context.architecture for item in records})
    architecture_summaries = {}
    for architecture in architectures:
        members = [item for item in records if item.context.architecture == architecture]
        within = [
            item
            for item in pairwise
            if item.left_architecture == architecture and item.right_architecture == architecture
        ]
        architecture_summaries[architecture] = ArchitectureMechanismSummary(
            architecture=architecture,
            context_count=len(members),
            mean_baseline_metric=float(np.mean([item.baseline_metric for item in members])),
            baseline_metric_std=float(np.std([item.baseline_metric for item in members])),
            mean_absolute_effect=float(np.mean([item.mean_absolute_effect for item in members])),
            mean_control_to_effect_ratio=_mean_optional(
                [item.control_to_effect_ratio for item in members]
            ),
            mean_top_k_concentration=float(
                np.mean([item.top_k_concentration(top_k) for item in members])
            ),
            within_architecture_stability=None if not within else _aggregate(within),
        )

    axes = ("architecture", "dataset", "session", "subject", "checkpoint")
    axis_groups: dict[str, list[PairwiseMechanismComparison]] = {
        "same_architecture": [item for item in pairwise if item.same_architecture],
        "cross_architecture": [item for item in pairwise if not item.same_architecture],
    }
    for axis in axes:
        axis_groups[axis] = [item for item in pairwise if axis in item.axes_changed]

    isolated_groups = {
        axis: [item for item in pairwise if item.axes_changed == (axis,)] for axis in axes
    }

    architecture_pair_groups: dict[str, list[PairwiseMechanismComparison]] = {}
    for item in pairwise:
        if item.same_architecture:
            continue
        left, right = sorted((item.left_architecture, item.right_architecture))
        architecture_pair_groups.setdefault(f"{left}<->{right}", []).append(item)

    return ArchitectureComparisonReport(
        context_count=len(records),
        architectures=tuple(architectures),
        pairwise=tuple(pairwise),
        architecture_summaries=MappingProxyType(architecture_summaries),
        axis_stability=_group_aggregate(axis_groups),
        isolated_axis_stability=_group_aggregate(isolated_groups),
        architecture_pair_stability=_group_aggregate(architecture_pair_groups),
    )


def _is_stable(item: StabilityAggregate, policy: HypothesisPolicy) -> bool:
    return (
        item.pair_count >= policy.min_pair_count
        and item.median_spearman_r is not None
        and item.median_spearman_r >= policy.stable_spearman
        and item.median_sign_agreement >= policy.stable_sign_agreement
        and item.median_top_k_jaccard >= policy.stable_top_k_jaccard
        and item.median_shared_target_fraction >= policy.min_shared_target_fraction
    )


def _all_context_ids(report: ArchitectureComparisonReport) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                context_id
                for item in report.pairwise
                for context_id in (item.left_context_id, item.right_context_id)
            }
        )
    )


def generate_mechanistic_hypotheses(
    records: Sequence[CausalEffectRecord],
    report: ArchitectureComparisonReport,
    *,
    policy: HypothesisPolicy | None = None,
) -> tuple[MechanisticHypothesis, ...]:
    """Prioritize falsifiable hypotheses from matched causal-map comparisons."""

    policy = policy or HypothesisPolicy()
    hypotheses = []
    context_ids = _all_context_ids(report)
    limitations = (
        "Ablation effects establish intervention sensitivity/necessity, not biological identity.",
        "Hypotheses require held-out validation and alternative intervention families.",
        "Task performance and metric definitions should be matched before mechanism comparison.",
    )

    architecture = report.isolated_axis_stability.get("architecture")
    if architecture is not None and _is_stable(architecture, policy):
        hypotheses.append(
            MechanisticHypothesis(
                hypothesis_id="shared-causal-temporal-structure",
                kind="architecture_invariant",
                statement=(
                    "Matched intervention targets preserve their causal-effect ordering when only "
                    "model architecture changes."
                ),
                priority="high",
                supporting_metrics={
                    "pair_count": architecture.pair_count,
                    "median_spearman_r": architecture.median_spearman_r,
                    "median_sign_agreement": architecture.median_sign_agreement,
                    "median_top_k_jaccard": architecture.median_top_k_jaccard,
                    "median_shared_target_fraction": architecture.median_shared_target_fraction,
                },
                context_ids=context_ids,
                falsification_tests=(
                    "Repeat on held-out matched sessions/datasets.",
                    "Match downstream task performance and training stage.",
                    "Repeat with mean, donor-patching, or conditional-resampling interventions.",
                    "Compare against shuffled or random-window causal maps.",
                ),
                limitations=limitations,
            )
        )

    session = report.isolated_axis_stability.get("session")
    if session is not None and architecture is not None and _is_stable(session, policy):
        cross_rank = architecture.median_spearman_r
        if cross_rank is not None and cross_rank <= policy.divergent_spearman:
            hypotheses.append(
                MechanisticHypothesis(
                    hypothesis_id="architecture-specific-implementation",
                    kind="architecture_specific",
                    statement=(
                        "Causal maps are stable when only session changes but diverge when only "
                        "architecture changes, suggesting architecture-specific task implementation."
                    ),
                    priority="high",
                    supporting_metrics={
                        "within_session_control_spearman": session.median_spearman_r,
                        "architecture_change_spearman": architecture.median_spearman_r,
                        "session_top_k_jaccard": session.median_top_k_jaccard,
                        "architecture_top_k_jaccard": architecture.median_top_k_jaccard,
                    },
                    context_ids=context_ids,
                    falsification_tests=(
                        "Repeat across additional architecture seeds.",
                        "Fit a separate latent alignment and repeat the causal comparison.",
                        "Compare architectures at matched performance and checkpoint maturity.",
                    ),
                    limitations=limitations,
                )
            )

    for axis in ("session", "dataset", "subject", "checkpoint"):
        aggregate = report.isolated_axis_stability.get(axis)
        if aggregate is None or not _is_stable(aggregate, policy):
            continue
        hypotheses.append(
            MechanisticHypothesis(
                hypothesis_id=f"stable-across-{axis}",
                kind="context_invariant",
                statement=(
                    f"Causal-effect maps remain stable when only {axis} changes in matched pairs."
                ),
                priority="medium",
                supporting_metrics={
                    "pair_count": aggregate.pair_count,
                    "median_spearman_r": aggregate.median_spearman_r,
                    "median_sign_agreement": aggregate.median_sign_agreement,
                    "median_top_k_jaccard": aggregate.median_top_k_jaccard,
                    "median_shared_target_fraction": aggregate.median_shared_target_fraction,
                },
                context_ids=context_ids,
                falsification_tests=(
                    f"Replicate in a held-out cohort that changes only {axis}.",
                    "Repeat with a second intervention family and held-out target metric.",
                    "Bootstrap trials/episodes to quantify uncertainty.",
                ),
                limitations=limitations,
            )
        )

    control_ratio = _mean_optional([item.control_to_effect_ratio for item in records])
    if control_ratio is not None and control_ratio >= policy.high_control_ratio:
        hypotheses.append(
            MechanisticHypothesis(
                hypothesis_id="distribution-shift-sensitive-effects",
                kind="diagnostic_warning",
                statement=(
                    "Matched control perturbations are large relative to targeted effects; some "
                    "apparent causal signal may reflect perturbation-induced distribution shift."
                ),
                priority="diagnostic",
                supporting_metrics={"mean_control_to_effect_ratio": control_ratio},
                context_ids=context_ids,
                falsification_tests=(
                    "Use in-distribution donor patching or conditional resampling.",
                    "Test intervention dose-response rather than one perturbation magnitude.",
                    "Check whether a perturbation detector predicts measured effect size.",
                ),
                limitations=limitations,
            )
        )

    k = report.pairwise[0].stability.top_k
    concentration = float(np.mean([item.top_k_concentration(k) for item in records]))
    if concentration >= policy.sparse_top_k_concentration:
        hypotheses.append(
            MechanisticHypothesis(
                hypothesis_id="concentrated-causal-support",
                kind="causal_sparsity",
                statement=(
                    "A small set of intervention targets accounts for most measured causal-effect "
                    "magnitude in the compared contexts."
                ),
                priority="medium",
                supporting_metrics={"mean_top_k_concentration": concentration},
                context_ids=context_ids,
                falsification_tests=(
                    "Test sufficiency by retaining only top causal targets.",
                    "Compare with equal-cardinality random target sets.",
                    "Vary temporal-window width to test discretization sensitivity.",
                ),
                limitations=limitations,
            )
        )

    return tuple(hypotheses)


def analyze_shared_computation(
    records: Sequence[CausalEffectRecord],
    *,
    top_k: int = 5,
    policy: HypothesisPolicy | None = None,
) -> SharedComputationAnalysis:
    """Compare records and produce policy-gated candidate hypotheses."""

    records = tuple(records)
    policy = policy or HypothesisPolicy()
    report = compare_causal_records(records, top_k=top_k)
    for item in report.pairwise:
        if item.stability.shared_targets < policy.min_shared_targets:
            raise ValueError(
                "insufficient shared intervention targets for mechanistic comparison: "
                f"{item.left_context_id!r} vs {item.right_context_id!r} has "
                f"{item.stability.shared_targets}, requires {policy.min_shared_targets}"
            )
    return SharedComputationAnalysis(
        comparison=report,
        hypotheses=generate_mechanistic_hypotheses(records, report, policy=policy),
        policy=policy,
    )
