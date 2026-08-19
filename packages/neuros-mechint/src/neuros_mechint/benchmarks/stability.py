"""Quantify stability of causal effect maps across contexts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.stats import rankdata


@dataclass(frozen=True, slots=True)
class EffectMapStability:
    """Agreement between two causal effect maps over shared targets."""

    shared_targets: int
    left_targets: int
    right_targets: int
    union_targets: int
    shared_target_fraction: float
    pearson_r: float | None
    spearman_r: float | None
    sign_agreement: float
    top_k: int
    top_k_jaccard: float
    mean_absolute_delta: float

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2:
        return None
    if np.isclose(left.std(), 0.0) or np.isclose(right.std(), 0.0):
        return None
    return float(np.corrcoef(left, right)[0, 1])


def compare_effect_maps(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    top_k: int = 10,
) -> EffectMapStability:
    """Compare causal maps using magnitude, ranking, sign, and top-k overlap.

    Only targets present in both maps are scored. Missing targets remain visible
    through ``shared_targets`` and ``shared_target_fraction`` rather than being
    silently converted into zero-effect interventions.
    """

    if top_k <= 0:
        raise ValueError("top_k must be positive")
    left_keys = set(left)
    right_keys = set(right)
    shared = sorted(left_keys & right_keys)
    if not shared:
        raise ValueError("effect maps have no shared targets")

    left_values = np.asarray([left[key] for key in shared], dtype=np.float64)
    right_values = np.asarray([right[key] for key in shared], dtype=np.float64)
    k = min(top_k, len(shared))

    left_top = set(sorted(shared, key=lambda key: abs(left[key]), reverse=True)[:k])
    right_top = set(sorted(shared, key=lambda key: abs(right[key]), reverse=True)[:k])
    top_union = left_top | right_top
    all_union = left_keys | right_keys

    return EffectMapStability(
        shared_targets=len(shared),
        left_targets=len(left_keys),
        right_targets=len(right_keys),
        union_targets=len(all_union),
        shared_target_fraction=len(shared) / len(all_union),
        pearson_r=_correlation(left_values, right_values),
        spearman_r=_correlation(rankdata(left_values), rankdata(right_values)),
        sign_agreement=float(np.mean(np.sign(left_values) == np.sign(right_values))),
        top_k=k,
        top_k_jaccard=len(left_top & right_top) / len(top_union),
        mean_absolute_delta=float(np.mean(np.abs(left_values - right_values))),
    )


def _extract(items: list[Any]) -> dict[str, float]:
    effect_map: dict[str, float] = {}
    for item in items:
        target = getattr(item, "component", None)
        if target is None:
            target = getattr(item, "target", None)
        if target is None:
            raise TypeError("result effect has neither component nor target")
        key = str(target)
        if key in effect_map:
            raise ValueError(f"duplicate effect target: {key}")
        effect_map[key] = float(item.effect)
    return effect_map


def extract_effect_map(result: Any, *, include_controls: bool = False) -> dict[str, float]:
    """Extract ``target/component -> effect`` from a typed mech-int result.

    ``include_controls`` is retained for compatibility. Prefer
    :func:`extract_control_map` when interventions and matched controls share
    target names, which is the normal case for causal audits.
    """

    items = list(result.effects)
    if include_controls:
        items.extend(result.controls)
    return _extract(items)


def extract_control_map(result: Any) -> dict[str, float]:
    """Extract matched-control effects without mixing them with interventions."""

    return _extract(list(result.controls))
