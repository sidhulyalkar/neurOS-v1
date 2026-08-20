"""Diagnostics for source-mixture stability and drift."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def effective_sample_size(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if w.ndim != 1 or w.size == 0 or np.any(w < 0) or not np.isfinite(w).all():
        raise ValueError("weights must be a finite non-negative 1-D vector")
    total = float(w.sum())
    if total <= 0:
        raise ValueError("weights must have positive mass")
    p = w / total
    return float(1.0 / np.sum(p**2))


def jensen_shannon_weight_shift(a: np.ndarray, b: np.ndarray) -> float:
    """Jensen-Shannon divergence between two source-weight distributions."""
    p = np.asarray(a, dtype=float)
    q = np.asarray(b, dtype=float)
    if p.shape != q.shape or p.ndim != 1:
        raise ValueError("weight vectors must be 1-D with equal shape")
    if np.any(p < 0) or np.any(q < 0) or p.sum() <= 0 or q.sum() <= 0:
        raise ValueError("weight vectors must be non-negative with positive mass")
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(x: np.ndarray, y: np.ndarray) -> float:
        mask = x > 0
        return float(np.sum(x[mask] * np.log(x[mask] / y[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


@dataclass(frozen=True, slots=True)
class SourceStabilityReport:
    base_weights: np.ndarray
    l1_change_when_removed: np.ndarray
    residual_change_when_removed: np.ndarray
    most_influential_source: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "base_weights": self.base_weights.tolist(),
            "l1_change_when_removed": self.l1_change_when_removed.tolist(),
            "residual_change_when_removed": self.residual_change_when_removed.tolist(),
            "most_influential_source": self.most_influential_source,
        }


def leave_one_source_out_stability(
    estimator: Any,
    source_moments: np.ndarray,
    target_moments: np.ndarray,
    **kwargs: Any,
) -> SourceStabilityReport:
    """Measure how fragile the mixture is to removing each source."""
    source = np.asarray(source_moments, dtype=float)
    target = np.asarray(target_moments, dtype=float)
    if source.ndim != 2 or source.shape[0] < 2:
        raise ValueError("leave-one-source-out stability requires at least two sources")
    base = estimator.estimate(source, target, **kwargs)
    n = source.shape[0]
    l1 = np.zeros(n)
    residual_delta = np.zeros(n)

    source_ids = kwargs.get("source_ids")
    prior = kwargs.get("prior")
    quality = kwargs.get("quality_scores")
    for removed in range(n):
        keep = np.arange(n) != removed
        child_kwargs = dict(kwargs)
        if source_ids is not None:
            child_kwargs["source_ids"] = [source_ids[i] for i in range(n) if keep[i]]
        if prior is not None:
            child_kwargs["prior"] = np.asarray(prior)[keep]
        if quality is not None:
            child_kwargs["quality_scores"] = np.asarray(quality)[keep]
        child = estimator.estimate(source[keep], target, **child_kwargs)
        expanded = np.zeros(n)
        expanded[keep] = child.weights
        l1[removed] = np.linalg.norm(base.weights - expanded, ord=1)
        residual_delta[removed] = child.residual - base.residual

    most = int(np.argmax(l1))
    return SourceStabilityReport(base.weights.copy(), l1, residual_delta, most)


@dataclass(frozen=True, slots=True)
class PerturbationReport:
    mean_weights: np.ndarray
    std_weights: np.ndarray
    max_std: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "mean_weights": self.mean_weights.tolist(),
            "std_weights": self.std_weights.tolist(),
            "max_std": self.max_std,
        }


def target_perturbation_sensitivity(
    estimator: Any,
    source_moments: np.ndarray,
    target_moments: np.ndarray,
    *,
    noise_scale: float = 0.01,
    n_repeats: int = 100,
    seed: int = 0,
    **kwargs: Any,
) -> PerturbationReport:
    """Stress-test weight stability to small target-summary perturbations."""
    if noise_scale < 0 or n_repeats <= 0:
        raise ValueError("noise_scale must be non-negative and n_repeats positive")
    source = np.asarray(source_moments, dtype=float)
    target = np.asarray(target_moments, dtype=float)
    active = source[np.isfinite(source).all(axis=1)]
    if active.size == 0:
        raise ValueError("at least one finite source is required")
    scale = np.std(np.vstack([active, target]), axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    rng = np.random.default_rng(seed)
    weights = []
    for _ in range(n_repeats):
        perturbed = target + rng.normal(0.0, noise_scale, size=target.shape) * scale
        weights.append(estimator.estimate(source, perturbed, **kwargs).weights)
    matrix = np.stack(weights)
    std = matrix.std(axis=0)
    return PerturbationReport(matrix.mean(axis=0), std, float(std.max()))
