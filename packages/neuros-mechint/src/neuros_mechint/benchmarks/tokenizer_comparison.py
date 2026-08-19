"""Compare causal intervention profiles across neural tokenization schemes."""

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
class TokenizerMechanismContext:
    """Identity of one tokenizer/downstream-model/data condition."""

    context_id: str
    tokenizer_id: str
    downstream_model_id: str
    dataset_id: str
    session_id: str
    subject_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in (
            "context_id",
            "tokenizer_id",
            "downstream_model_id",
            "dataset_id",
            "session_id",
        ):
            if not getattr(self, name):
                raise ValueError(f"{name} must be non-empty")
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def changed_axes(self, other: TokenizerMechanismContext) -> tuple[str, ...]:
        axes = []
        for field_name, axis in (
            ("tokenizer_id", "tokenizer"),
            ("downstream_model_id", "downstream_model"),
            ("dataset_id", "dataset"),
            ("session_id", "session"),
            ("subject_id", "subject"),
        ):
            if getattr(self, field_name) != getattr(other, field_name):
                axes.append(axis)
        return tuple(axes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_id": self.context_id,
            "tokenizer_id": self.tokenizer_id,
            "downstream_model_id": self.downstream_model_id,
            "dataset_id": self.dataset_id,
            "session_id": self.session_id,
            "subject_id": self.subject_id,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class TokenizerEffectRecord:
    context: TokenizerMechanismContext
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
    def control_to_effect_ratio(self) -> float | None:
        if not self.control_map or self.mean_absolute_effect <= 1e-12:
            return None
        control = float(np.mean(np.abs(np.asarray(list(self.control_map.values())))))
        return control / self.mean_absolute_effect

    def to_dict(self) -> dict[str, Any]:
        return {
            "context": self.context.to_dict(),
            "baseline_metric": self.baseline_metric,
            "metric_name": self.metric_name,
            "effect_map": dict(self.effect_map),
            "control_map": dict(self.control_map),
            "mean_absolute_effect": self.mean_absolute_effect,
            "control_to_effect_ratio": self.control_to_effect_ratio,
        }


@dataclass(frozen=True, slots=True)
class TokenizerPairComparison:
    left_context_id: str
    right_context_id: str
    left_tokenizer_id: str
    right_tokenizer_id: str
    axes_changed: tuple[str, ...]
    baseline_delta: float
    stability: EffectMapStability

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["axes_changed"] = list(self.axes_changed)
        payload["stability"] = self.stability.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class TokenizerStabilityAggregate:
    pair_count: int
    median_spearman_r: float | None
    median_sign_agreement: float
    median_top_k_jaccard: float
    median_shared_target_fraction: float
    median_mean_absolute_delta: float

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class TokenizerMechanismHypothesis:
    hypothesis_id: str
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
            "statement": self.statement,
            "priority": self.priority,
            "supporting_metrics": dict(self.supporting_metrics),
            "context_ids": list(self.context_ids),
            "falsification_tests": list(self.falsification_tests),
            "limitations": list(self.limitations),
        }


@dataclass(frozen=True, slots=True)
class TokenizerComparisonReport:
    tokenizer_ids: tuple[str, ...]
    pairwise: tuple[TokenizerPairComparison, ...]
    isolated_tokenizer_stability: TokenizerStabilityAggregate | None
    hypotheses: tuple[TokenizerMechanismHypothesis, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_ids": list(self.tokenizer_ids),
            "pairwise": [item.to_dict() for item in self.pairwise],
            "isolated_tokenizer_stability": (
                None
                if self.isolated_tokenizer_stability is None
                else self.isolated_tokenizer_stability.to_dict()
            ),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
        }


def _median_optional(values: Sequence[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return None if not finite else float(median(finite))


def _aggregate(items: Sequence[TokenizerPairComparison]) -> TokenizerStabilityAggregate:
    if not items:
        raise ValueError("cannot aggregate an empty comparison set")
    return TokenizerStabilityAggregate(
        pair_count=len(items),
        median_spearman_r=_median_optional(
            [item.stability.spearman_r for item in items]
        ),
        median_sign_agreement=float(
            median(item.stability.sign_agreement for item in items)
        ),
        median_top_k_jaccard=float(
            median(item.stability.top_k_jaccard for item in items)
        ),
        median_shared_target_fraction=float(
            median(item.stability.shared_target_fraction for item in items)
        ),
        median_mean_absolute_delta=float(
            median(item.stability.mean_absolute_delta for item in items)
        ),
    )


def compare_tokenizer_mechanisms(
    records: Sequence[TokenizerEffectRecord],
    *,
    top_k: int = 5,
    stable_spearman: float = 0.7,
    divergent_spearman: float = 0.3,
    min_shared_target_fraction: float = 0.75,
) -> TokenizerComparisonReport:
    """Compare tokenizers using only matched one-factor tokenizer contrasts."""

    records = tuple(records)
    if len(records) < 2:
        raise ValueError("at least two tokenizer effect records are required")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if not 0.0 <= stable_spearman <= 1.0:
        raise ValueError("stable_spearman must lie in [0, 1]")
    if not 0.0 <= divergent_spearman <= 1.0:
        raise ValueError("divergent_spearman must lie in [0, 1]")
    if not 0.0 < min_shared_target_fraction <= 1.0:
        raise ValueError("min_shared_target_fraction must lie in (0, 1]")

    ids = [item.context.context_id for item in records]
    if len(ids) != len(set(ids)):
        raise ValueError("context_id values must be unique")

    pairwise = []
    for left, right in combinations(records, 2):
        pairwise.append(
            TokenizerPairComparison(
                left_context_id=left.context.context_id,
                right_context_id=right.context.context_id,
                left_tokenizer_id=left.context.tokenizer_id,
                right_tokenizer_id=right.context.tokenizer_id,
                axes_changed=left.context.changed_axes(right.context),
                baseline_delta=right.baseline_metric - left.baseline_metric,
                stability=compare_effect_maps(
                    left.effect_map,
                    right.effect_map,
                    top_k=top_k,
                ),
            )
        )

    isolated = [item for item in pairwise if item.axes_changed == ("tokenizer",)]
    aggregate = None if not isolated else _aggregate(isolated)

    contexts = tuple(sorted(ids))
    hypotheses = []
    limitations = (
        "Tokenizers must represent the same semantic time intervals and task target.",
        "A downstream scorer can adapt to tokenizer-specific artifacts.",
        "Causal-map agreement does not imply identical internal model computation.",
    )
    if (
        aggregate is not None
        and aggregate.median_shared_target_fraction >= min_shared_target_fraction
        and aggregate.median_spearman_r is not None
        and aggregate.median_spearman_r >= stable_spearman
    ):
        hypotheses.append(
            TokenizerMechanismHypothesis(
                hypothesis_id="tokenization-invariant-causal-profile",
                statement=(
                    "Matched temporal interventions preserve their causal ordering "
                    "when only the neural tokenization scheme changes."
                ),
                priority="high",
                supporting_metrics={
                    "pair_count": aggregate.pair_count,
                    "median_spearman_r": aggregate.median_spearman_r,
                    "median_sign_agreement": aggregate.median_sign_agreement,
                    "median_top_k_jaccard": aggregate.median_top_k_jaccard,
                    "median_shared_target_fraction": aggregate.median_shared_target_fraction,
                },
                context_ids=contexts,
                falsification_tests=(
                    "Repeat with a newly trained downstream model per tokenizer.",
                    "Repeat on held-out sessions and subjects.",
                    "Match token budget and temporal resolution.",
                    "Repeat with token-type and side-feature interventions.",
                ),
                limitations=limitations,
            )
        )
    elif (
        aggregate is not None
        and aggregate.median_shared_target_fraction >= min_shared_target_fraction
        and aggregate.median_spearman_r is not None
        and aggregate.median_spearman_r <= divergent_spearman
    ):
        hypotheses.append(
            TokenizerMechanismHypothesis(
                hypothesis_id="tokenizer-dependent-causal-profile",
                statement=(
                    "Matched temporal causal profiles diverge when only tokenization changes, "
                    "making tokenizer choice a candidate source of computational bias."
                ),
                priority="high",
                supporting_metrics={
                    "pair_count": aggregate.pair_count,
                    "median_spearman_r": aggregate.median_spearman_r,
                    "median_shared_target_fraction": aggregate.median_shared_target_fraction,
                },
                context_ids=contexts,
                falsification_tests=(
                    "Equalize token budget, time resolution, and downstream model capacity.",
                    "Retrain downstream models from multiple seeds for each tokenizer.",
                    "Test whether divergence survives an aligned latent representation.",
                ),
                limitations=limitations,
            )
        )

    return TokenizerComparisonReport(
        tokenizer_ids=tuple(sorted({item.context.tokenizer_id for item in records})),
        pairwise=tuple(pairwise),
        isolated_tokenizer_stability=aggregate,
        hypotheses=tuple(hypotheses),
    )
