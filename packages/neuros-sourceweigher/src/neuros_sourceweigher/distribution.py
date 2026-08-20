"""Distribution-level source similarity measures."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .strategies import _softmax
from .weigher import WeightingDiagnostics, WeightingResult, _entropy, _project_with_floor


def _matrix(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = arr.reshape(arr.shape[0], -1)
    if arr.ndim != 2 or arr.shape[0] < 2:
        raise ValueError("each domain must contain at least two samples")
    if not np.all(np.isfinite(arr)):
        raise ValueError("domain samples must be finite")
    return arr


def _subsample(x: np.ndarray, max_samples: int, rng: np.random.Generator) -> np.ndarray:
    if x.shape[0] <= max_samples:
        return x
    idx = rng.choice(x.shape[0], size=max_samples, replace=False)
    return x[idx]


def _median_gamma(
    domains: Sequence[np.ndarray],
    *,
    max_samples: int,
    seed: int,
) -> float:
    """Choose one shared RBF bandwidth so source discrepancies are comparable."""
    rng = np.random.default_rng(seed)
    pooled = np.vstack(
        [_subsample(_matrix(domain), max_samples=max_samples, rng=rng) for domain in domains]
    )
    if pooled.shape[0] > max_samples:
        pooled = _subsample(pooled, max_samples=max_samples, rng=rng)
    sq = np.sum((pooled[:, None, :] - pooled[None, :, :]) ** 2, axis=-1)
    tri = sq[np.triu_indices_from(sq, k=1)]
    positive = tri[tri > 0]
    median = float(np.median(positive)) if positive.size else 1.0
    return 1.0 / max(2.0 * median, 1e-12)


def rbf_mmd2(
    x: np.ndarray,
    y: np.ndarray,
    *,
    gamma: float | None = None,
    max_samples: int = 512,
    seed: int = 0,
) -> float:
    """Biased RBF-kernel maximum mean discrepancy squared."""
    if max_samples < 2:
        raise ValueError("max_samples must be at least 2")
    rng = np.random.default_rng(seed)
    x = _subsample(_matrix(x), max_samples, rng)
    y = _subsample(_matrix(y), max_samples, rng)
    if x.shape[1] != y.shape[1]:
        raise ValueError("source and target features must have equal dimension")
    if gamma is None:
        gamma = _median_gamma([x, y], max_samples=max_samples, seed=seed)
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    kxx = np.exp(-gamma * np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1))
    kyy = np.exp(-gamma * np.sum((y[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    kxy = np.exp(-gamma * np.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1))
    return float(max(kxx.mean() + kyy.mean() - 2.0 * kxy.mean(), 0.0))


def spd_affine_invariant_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Affine-invariant Riemannian distance between SPD covariance matrices."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or b.shape != a.shape:
        raise ValueError("a and b must be square matrices of equal shape")
    vals, vecs = np.linalg.eigh(a)
    if np.any(vals <= 0):
        raise ValueError("a must be positive definite")
    inv_sqrt = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    c = inv_sqrt @ b @ inv_sqrt
    c = 0.5 * (c + c.T)
    eig = np.linalg.eigvalsh(c)
    if np.any(eig <= 0):
        raise ValueError("b must be positive definite")
    return float(np.linalg.norm(np.log(eig)))


class MMDSourceWeigher:
    """Gibbs weighting from a shared-kernel distribution discrepancy.

    A single RBF bandwidth is estimated across the target and all sources. This
    matters because independently tuning a kernel per source makes the resulting
    MMD values unsuitable for direct source ranking.
    """

    def __init__(
        self,
        *,
        temperature: float = 0.1,
        gamma: float | None = None,
        max_samples: int = 512,
        seed: int = 0,
        min_weight: float = 0.0,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if max_samples < 2:
            raise ValueError("max_samples must be at least 2")
        if gamma is not None and gamma <= 0:
            raise ValueError("gamma must be positive")
        self.temperature = float(temperature)
        self.gamma = gamma
        self.max_samples = int(max_samples)
        self.seed = int(seed)
        self.min_weight = float(min_weight)

    def estimate(
        self,
        source_samples: Mapping[str, np.ndarray],
        target_samples: np.ndarray,
    ) -> WeightingResult:
        if not source_samples:
            raise ValueError("source_samples cannot be empty")
        ids = tuple(source_samples)
        target = _matrix(target_samples)
        sources = {sid: _matrix(source_samples[sid]) for sid in ids}
        dims = {target.shape[1], *(value.shape[1] for value in sources.values())}
        if len(dims) != 1:
            raise ValueError("all source and target features must have equal dimension")
        gamma = self.gamma
        if gamma is None:
            gamma = _median_gamma(
                [target, *(sources[sid] for sid in ids)],
                max_samples=self.max_samples,
                seed=self.seed,
            )
        distances = np.array(
            [
                rbf_mmd2(
                    sources[sid],
                    target,
                    gamma=gamma,
                    max_samples=self.max_samples,
                    seed=self.seed + i + 1,
                )
                for i, sid in enumerate(ids)
            ],
            dtype=float,
        )
        w = _project_with_floor(_softmax(-distances / self.temperature), self.min_weight)
        diag = WeightingDiagnostics(
            method="mmd_gibbs",
            residual=float("nan"),
            scaled_residual=float("nan"),
            objective=float(distances @ w),
            effective_sample_size=float(1.0 / np.sum(w**2)),
            entropy=_entropy(w),
            max_weight=float(w.max()),
            iterations=1,
            converged=True,
            condition_number=1.0,
            source_distances=tuple(float(x) for x in distances),
            metadata={
                "temperature": self.temperature,
                "gamma": float(gamma),
                "max_samples": self.max_samples,
                "shared_bandwidth": True,
            },
        )
        return WeightingResult(w, diag, ids)


class RiemannianCovarianceWeigher:
    """Weight domains using affine-invariant covariance geometry."""

    def __init__(
        self,
        *,
        temperature: float = 1.0,
        shrinkage: float = 1e-3,
        min_weight: float = 0.0,
    ) -> None:
        if temperature <= 0 or shrinkage <= 0:
            raise ValueError("temperature and shrinkage must be positive")
        self.temperature = float(temperature)
        self.shrinkage = float(shrinkage)
        self.min_weight = float(min_weight)

    def _cov(self, x: np.ndarray) -> np.ndarray:
        x = _matrix(x)
        cov = np.atleast_2d(np.cov(x, rowvar=False, bias=True))
        trace_scale = float(np.trace(cov) / cov.shape[0]) if cov.size else 1.0
        reg = self.shrinkage * max(trace_scale, 1e-8)
        return cov + reg * np.eye(cov.shape[0])

    def estimate(
        self,
        source_samples: Mapping[str, np.ndarray],
        target_samples: np.ndarray,
    ) -> WeightingResult:
        if not source_samples:
            raise ValueError("source_samples cannot be empty")
        ids = tuple(source_samples)
        target_cov = self._cov(target_samples)
        distances = np.array(
            [
                spd_affine_invariant_distance(self._cov(source_samples[sid]), target_cov)
                for sid in ids
            ],
            dtype=float,
        )
        w = _project_with_floor(_softmax(-distances / self.temperature), self.min_weight)
        diag = WeightingDiagnostics(
            method="riemannian_covariance",
            residual=float("nan"),
            scaled_residual=float("nan"),
            objective=float(distances @ w),
            effective_sample_size=float(1.0 / np.sum(w**2)),
            entropy=_entropy(w),
            max_weight=float(w.max()),
            iterations=1,
            converged=True,
            condition_number=1.0,
            source_distances=tuple(float(x) for x in distances),
            metadata={
                "temperature": self.temperature,
                "shrinkage": self.shrinkage,
            },
        )
        return WeightingResult(w, diag, ids)
