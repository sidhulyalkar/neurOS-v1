"""Alternative source-weighting strategies.

These algorithms deliberately expose different assumptions rather than hiding
all source selection behind one score.
"""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .weigher import (
    WeightingDiagnostics,
    WeightingResult,
    _entropy,
    _normalise_prior,
    _project_with_floor,
    _scaled_problem,
)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=float)
    x = x - np.max(x)
    exp = np.exp(x)
    total = float(exp.sum())
    if not np.isfinite(total) or total <= 0:
        return np.full(x.size, 1.0 / x.size)
    return exp / total


class DistanceWeigher:
    """Weight sources by their distance from the target summary."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        standardize: bool = True,
        min_weight: float = 0.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.standardize = bool(standardize)
        self.min_weight = float(min_weight)

    def estimate(
        self,
        source_moments: np.ndarray,
        target_moments: np.ndarray,
        *,
        prior: np.ndarray | None = None,
        source_ids: Sequence[str] | None = None,
        **_: Any,
    ) -> WeightingResult:
        source = np.asarray(source_moments, dtype=float)
        target = np.asarray(target_moments, dtype=float)
        if source.ndim != 2 or target.ndim != 1:
            raise ValueError("source_moments must be 2-D and target_moments 1-D")
        if source.shape[1] != target.size:
            raise ValueError("source and target dimensions do not match")
        if not np.all(np.isfinite(target)):
            raise ValueError("target_moments must be finite")

        n = source.shape[0]
        ids = tuple(source_ids or (str(i) for i in range(n)))
        if len(ids) != n:
            raise ValueError("source_ids must match the number of sources")
        active = np.all(np.isfinite(source), axis=1)
        if not np.any(active):
            raise ValueError("no finite source domains are available")
        excluded = tuple(int(i) for i in np.flatnonzero(~active))

        src, tgt, _ = _scaled_problem(source[active], target, standardize=self.standardize)
        distances = np.linalg.norm(src - tgt[None, :], axis=1)
        full_prior = _normalise_prior(prior, n)
        active_prior = full_prior[active]
        active_prior = active_prior / active_prior.sum()

        logits = -distances / self.temperature + np.log(np.maximum(active_prior, 1e-15))
        w = _softmax(logits)
        w = _project_with_floor(w, self.min_weight)

        full_w = np.zeros(n, dtype=float)
        full_w[active] = w
        reconstruction = source[active].T @ w
        residual = float(np.linalg.norm(reconstruction - target))
        source_distances = np.full(n, np.inf)
        source_distances[active] = distances
        diag = WeightingDiagnostics(
            method="distance_softmax",
            residual=residual,
            scaled_residual=float(np.linalg.norm(src.T @ w - tgt)),
            objective=float(distances @ w),
            effective_sample_size=float(1.0 / np.sum(full_w**2)),
            entropy=_entropy(full_w),
            max_weight=float(full_w.max()),
            iterations=1,
            converged=True,
            condition_number=1.0,
            source_distances=tuple(float(x) for x in source_distances),
            excluded_sources=excluded,
            metadata={
                "temperature": self.temperature,
                "standardize": self.standardize,
                "min_weight": self.min_weight,
            },
        )
        return WeightingResult(full_w, diag, ids)

    def estimate_weights(self, source_moments, target_moments, **kwargs):
        return self.estimate(source_moments, target_moments, **kwargs).weights


class GibbsRiskWeigher:
    """Convert per-source risk estimates into an entropy-regularised mixture."""

    def __init__(self, *, temperature: float = 0.1, min_weight: float = 0.0) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = float(temperature)
        self.min_weight = float(min_weight)

    def estimate(
        self,
        risks: np.ndarray,
        *,
        prior: np.ndarray | None = None,
        source_ids: Sequence[str] | None = None,
    ) -> WeightingResult:
        risk = np.asarray(risks, dtype=float)
        if risk.ndim != 1 or risk.size == 0:
            raise ValueError("risks must be a non-empty 1-D vector")
        n = risk.size
        ids = tuple(source_ids or (str(i) for i in range(n)))
        if len(ids) != n:
            raise ValueError("source_ids must match risks")

        active = np.isfinite(risk)
        if not np.any(active):
            raise ValueError("at least one finite risk is required")
        p = _normalise_prior(prior, n)
        p_active = p[active]
        p_active = p_active / p_active.sum()
        logits = -risk[active] / self.temperature + np.log(np.maximum(p_active, 1e-15))
        w = _project_with_floor(_softmax(logits), self.min_weight)
        full_w = np.zeros(n)
        full_w[active] = w

        diag = WeightingDiagnostics(
            method="gibbs_risk",
            residual=float("nan"),
            scaled_residual=float("nan"),
            objective=float(risk[active] @ w),
            effective_sample_size=float(1.0 / np.sum(full_w**2)),
            entropy=_entropy(full_w),
            max_weight=float(full_w.max()),
            iterations=1,
            converged=True,
            condition_number=1.0,
            source_distances=tuple(float(x) for x in risk),
            excluded_sources=tuple(int(i) for i in np.flatnonzero(~active)),
            metadata={"temperature": self.temperature, "min_weight": self.min_weight},
        )
        return WeightingResult(full_w, diag, ids)


class OnlineSourceWeigher:
    """Smooth a source-weight estimator over time for drifting BCI streams."""

    def __init__(
        self,
        estimator: Any,
        *,
        adaptation_rate: float = 0.2,
        max_l1_step: float | None = 0.4,
    ) -> None:
        if not 0 < adaptation_rate <= 1:
            raise ValueError("adaptation_rate must be in (0, 1]")
        if max_l1_step is not None and max_l1_step <= 0:
            raise ValueError("max_l1_step must be positive")
        self.estimator = estimator
        self.adaptation_rate = float(adaptation_rate)
        self.max_l1_step = max_l1_step
        self.weights_: np.ndarray | None = None

    def reset(self) -> None:
        self.weights_ = None

    def update(self, *args: Any, **kwargs: Any) -> WeightingResult:
        instant = self.estimator.estimate(*args, **kwargs)
        if self.weights_ is None:
            smoothed = instant.weights.copy()
            drift = 0.0
        else:
            if self.weights_.shape != instant.weights.shape:
                raise ValueError("source count changed; call reset() before updating")
            candidate = (1.0 - self.adaptation_rate) * self.weights_ + self.adaptation_rate * instant.weights
            delta = candidate - self.weights_
            drift = float(np.linalg.norm(delta, ord=1))
            if self.max_l1_step is not None and drift > self.max_l1_step:
                delta *= self.max_l1_step / drift
                candidate = self.weights_ + delta
                drift = float(np.linalg.norm(delta, ord=1))
            smoothed = candidate / candidate.sum()
        self.weights_ = smoothed

        d = instant.diagnostics
        diag = WeightingDiagnostics(
            method=f"online:{d.method}",
            residual=d.residual,
            scaled_residual=d.scaled_residual,
            objective=d.objective,
            effective_sample_size=float(1.0 / np.sum(smoothed**2)),
            entropy=_entropy(smoothed),
            max_weight=float(smoothed.max()),
            iterations=d.iterations,
            converged=d.converged,
            condition_number=d.condition_number,
            source_distances=d.source_distances,
            excluded_sources=d.excluded_sources,
            metadata={
                **dict(d.metadata),
                "adaptation_rate": self.adaptation_rate,
                "l1_weight_drift": drift,
                "instantaneous_weights": instant.weights.tolist(),
            },
        )
        return WeightingResult(smoothed.copy(), diag, instant.source_ids)
