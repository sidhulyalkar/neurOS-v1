"""Reliability-aware source weighting for transfer and fusion.

The default estimator solves the *actual* simplex-constrained quadratic
program with projected gradient descent instead of solving an unconstrained
least-squares problem and projecting only once.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np


def project_to_simplex(v: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{w >= 0, sum(w) = total}``."""
    x = np.asarray(v, dtype=float)
    if x.ndim != 1:
        raise ValueError("simplex projection expects a 1-D vector")
    if x.size == 0:
        raise ValueError("simplex projection expects at least one value")
    if not np.all(np.isfinite(x)):
        raise ValueError("simplex projection requires finite values")
    if total <= 0:
        raise ValueError("total must be positive")
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u) - total
    ind = np.arange(1, x.size + 1)
    active = u - cssv / ind > 0
    if not np.any(active):
        return np.full(x.size, total / x.size, dtype=float)
    rho = np.nonzero(active)[0][-1]
    theta = cssv[rho] / float(rho + 1)
    w = np.maximum(x - theta, 0.0)
    s = float(w.sum())
    if s <= 0:
        return np.full(x.size, total / x.size, dtype=float)
    return w * (total / s)


def _project_with_floor(v: np.ndarray, min_weight: float) -> np.ndarray:
    n = int(v.size)
    if min_weight < 0:
        raise ValueError("min_weight must be non-negative")
    remaining = 1.0 - n * min_weight
    if remaining <= 0:
        if np.isclose(remaining, 0.0):
            return np.full(n, 1.0 / n)
        raise ValueError("min_weight is too large for the number of active sources")
    return min_weight + project_to_simplex(v - min_weight, total=remaining)


def _normalise_prior(prior: np.ndarray | None, n: int) -> np.ndarray:
    if prior is None:
        return np.full(n, 1.0 / n, dtype=float)
    p = np.asarray(prior, dtype=float)
    if p.shape != (n,):
        raise ValueError(f"prior must have shape ({n},)")
    if not np.all(np.isfinite(p)) or np.any(p < 0):
        raise ValueError("prior must contain finite non-negative values")
    total = float(p.sum())
    if total <= 0:
        raise ValueError("prior must have positive mass")
    return p / total


def _scaled_problem(
    source_moments: np.ndarray,
    target_moments: np.ndarray,
    *,
    standardize: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not standardize:
        scale = np.ones(source_moments.shape[1], dtype=float)
        return source_moments, target_moments, scale
    stacked = np.vstack([source_moments, target_moments[None, :]])
    center = stacked.mean(axis=0)
    scale = stacked.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    return (source_moments - center) / scale, (target_moments - center) / scale, scale


def _entropy(weights: np.ndarray) -> float:
    p = np.asarray(weights, dtype=float)
    active = p[p > 0]
    return float(-np.sum(active * np.log(active)))


@dataclass(frozen=True, slots=True)
class WeightingDiagnostics:
    method: str
    residual: float
    scaled_residual: float
    objective: float
    effective_sample_size: float
    entropy: float
    max_weight: float
    iterations: int
    converged: bool
    condition_number: float
    source_distances: tuple[float, ...]
    excluded_sources: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "residual": self.residual,
            "scaled_residual": self.scaled_residual,
            "objective": self.objective,
            "effective_sample_size": self.effective_sample_size,
            "entropy": self.entropy,
            "max_weight": self.max_weight,
            "iterations": self.iterations,
            "converged": self.converged,
            "condition_number": self.condition_number,
            "source_distances": list(self.source_distances),
            "excluded_sources": list(self.excluded_sources),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class WeightingResult:
    weights: np.ndarray
    diagnostics: WeightingDiagnostics
    source_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "weights", np.asarray(self.weights, dtype=float))

    @property
    def ess(self) -> float:
        return self.diagnostics.effective_sample_size

    @property
    def residual(self) -> float:
        return self.diagnostics.residual

    def by_source(self) -> dict[str, float]:
        if not self.source_ids:
            return {str(i): float(w) for i, w in enumerate(self.weights)}
        return {sid: float(w) for sid, w in zip(self.source_ids, self.weights)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "weights": self.weights.tolist(),
            "source_ids": list(self.source_ids),
            "diagnostics": self.diagnostics.to_dict(),
        }


class SourceWeigher:
    """Simplex-constrained, reliability-aware moment matching.

    Solves

    ``0.5 ||A w - b||² + 0.5 * ridge ||w-prior||² - quality_strength qᵀw``

    subject to ``w_j >= min_weight`` and ``sum_j w_j = 1``.

    Non-finite source rows are treated as unavailable and receive zero mass.
    The target must be finite.
    """

    def __init__(
        self,
        *,
        ridge: float = 1e-3,
        quality_strength: float = 0.0,
        standardize: bool = True,
        min_weight: float = 0.0,
        max_iter: int = 5000,
        tol: float = 1e-10,
    ) -> None:
        if ridge < 0:
            raise ValueError("ridge must be non-negative")
        if quality_strength < 0:
            raise ValueError("quality_strength must be non-negative")
        if min_weight < 0:
            raise ValueError("min_weight must be non-negative")
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if tol <= 0:
            raise ValueError("tol must be positive")
        self.ridge = float(ridge)
        self.quality_strength = float(quality_strength)
        self.standardize = bool(standardize)
        self.min_weight = float(min_weight)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

    def estimate(
        self,
        source_moments: np.ndarray,
        target_moments: np.ndarray,
        *,
        prior: np.ndarray | None = None,
        quality_scores: np.ndarray | None = None,
        source_ids: Sequence[str] | None = None,
    ) -> WeightingResult:
        source = np.asarray(source_moments, dtype=float)
        target = np.asarray(target_moments, dtype=float)
        if source.ndim != 2:
            raise ValueError("source_moments must be a 2-D array")
        if target.ndim != 1:
            raise ValueError("target_moments must be a 1-D array")
        if source.shape[0] == 0 or source.shape[1] == 0:
            raise ValueError("source_moments cannot be empty")
        if source.shape[1] != target.size:
            raise ValueError(
                f"target_moments has length {target.size}; expected {source.shape[1]}"
            )
        if not np.all(np.isfinite(target)):
            raise ValueError("target_moments must contain only finite values")

        n_sources = source.shape[0]
        ids = tuple(source_ids or (str(i) for i in range(n_sources)))
        if len(ids) != n_sources:
            raise ValueError("source_ids must match the number of sources")

        active_mask = np.all(np.isfinite(source), axis=1)
        excluded = tuple(int(i) for i in np.flatnonzero(~active_mask))
        if not np.any(active_mask):
            raise ValueError("no finite source domains are available")

        active_source = source[active_mask]
        n_active = active_source.shape[0]

        full_prior = _normalise_prior(prior, n_sources)
        active_prior = full_prior[active_mask]
        if active_prior.sum() <= 0:
            active_prior = np.full(n_active, 1.0 / n_active)
        else:
            active_prior = active_prior / active_prior.sum()

        active_quality = np.zeros(n_active, dtype=float)
        if quality_scores is not None:
            q = np.asarray(quality_scores, dtype=float)
            if q.shape != (n_sources,):
                raise ValueError(f"quality_scores must have shape ({n_sources},)")
            if not np.all(np.isfinite(q[active_mask])):
                raise ValueError("quality_scores for active sources must be finite")
            active_quality = q[active_mask]
            q_min, q_max = float(active_quality.min()), float(active_quality.max())
            if q_max > q_min:
                active_quality = (active_quality - q_min) / (q_max - q_min)
            else:
                active_quality = np.zeros_like(active_quality)

        scaled_source, scaled_target, _ = _scaled_problem(
            active_source, target, standardize=self.standardize
        )
        A = scaled_source.T
        b = scaled_target
        spectral = float(np.linalg.norm(A, ord=2))
        lipschitz = spectral * spectral + self.ridge
        step = 1.0 / max(lipschitz, 1e-12)

        w = _project_with_floor(active_prior, self.min_weight)
        converged = False
        iterations = 0
        for iterations in range(1, self.max_iter + 1):
            grad = A.T @ (A @ w - b)
            if self.ridge:
                grad = grad + self.ridge * (w - active_prior)
            if self.quality_strength:
                grad = grad - self.quality_strength * active_quality
            new_w = _project_with_floor(w - step * grad, self.min_weight)
            if np.linalg.norm(new_w - w, ord=1) <= self.tol:
                w = new_w
                converged = True
                break
            w = new_w

        full_w = np.zeros(n_sources, dtype=float)
        full_w[active_mask] = w

        raw_residual_vector = source[active_mask].T @ w - target
        scaled_residual_vector = A @ w - b
        raw_residual = float(np.linalg.norm(raw_residual_vector))
        scaled_residual = float(np.linalg.norm(scaled_residual_vector))
        objective = 0.5 * scaled_residual**2
        if self.ridge:
            objective += 0.5 * self.ridge * float(np.sum((w - active_prior) ** 2))
        if self.quality_strength:
            objective -= self.quality_strength * float(active_quality @ w)

        gram = A.T @ A + self.ridge * np.eye(n_active)
        try:
            condition = float(np.linalg.cond(gram))
        except np.linalg.LinAlgError:
            condition = float("inf")
        source_distances = np.full(n_sources, np.inf, dtype=float)
        source_distances[active_mask] = np.linalg.norm(
            scaled_source - scaled_target[None, :], axis=1
        )

        diagnostics = WeightingDiagnostics(
            method="constrained_moment_match",
            residual=raw_residual,
            scaled_residual=scaled_residual,
            objective=float(objective),
            effective_sample_size=float(1.0 / np.sum(full_w**2)),
            entropy=_entropy(full_w),
            max_weight=float(full_w.max()),
            iterations=iterations,
            converged=converged,
            condition_number=condition,
            source_distances=tuple(float(x) for x in source_distances),
            excluded_sources=excluded,
            metadata={
                "ridge": self.ridge,
                "quality_strength": self.quality_strength,
                "standardize": self.standardize,
                "min_weight": self.min_weight,
            },
        )
        return WeightingResult(full_w, diagnostics, ids)

    def estimate_weights(
        self,
        source_moments: np.ndarray,
        target_moments: np.ndarray,
        **kwargs: Any,
    ) -> np.ndarray:
        """Backwards-compatible weights-only API."""
        return self.estimate(source_moments, target_moments, **kwargs).weights
