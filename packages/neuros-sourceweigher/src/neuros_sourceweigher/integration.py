"""Integration helpers for foundation-model adaptation and neurOS fusion."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .summaries import summarize_features
from .weigher import SourceWeigher, WeightingResult


class RepresentationSourceWeigher:
    """Estimate source weights from distribution summaries of embeddings."""

    def __init__(
        self,
        estimator: Any | None = None,
        *,
        statistics: tuple[str, ...] = ("mean", "log_std"),
    ) -> None:
        self.estimator = estimator or SourceWeigher(ridge=1e-2)
        self.statistics = statistics

    def estimate(
        self,
        source_embeddings: Mapping[str, np.ndarray],
        target_embeddings: np.ndarray,
        *,
        prior: Mapping[str, float] | None = None,
        quality_scores: Mapping[str, float] | None = None,
    ) -> WeightingResult:
        if not source_embeddings:
            raise ValueError("source_embeddings cannot be empty")
        source_ids = tuple(source_embeddings)
        summaries = np.stack(
            [
                summarize_features(source_embeddings[sid], statistics=self.statistics)
                for sid in source_ids
            ],
            axis=0,
        )
        target_summary = summarize_features(target_embeddings, statistics=self.statistics)
        prior_array = None
        if prior is not None:
            prior_array = np.array([prior.get(sid, 0.0) for sid in source_ids], dtype=float)
        quality_array = None
        if quality_scores is not None:
            quality_array = np.array(
                [quality_scores.get(sid, 0.0) for sid in source_ids], dtype=float
            )
        return self.estimator.estimate(
            summaries,
            target_summary,
            prior=prior_array,
            quality_scores=quality_array,
            source_ids=source_ids,
        )


class ReliabilityWeightedFusion:
    """neurOS ``NodeKind.FUSION`` operator with explicit reliability weights."""

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        *,
        mode: str = "scale_concat",
        normalize: bool = True,
    ) -> None:
        if mode not in {"scale_concat", "weighted_mean"}:
            raise ValueError("mode must be 'scale_concat' or 'weighted_mean'")
        self.mode = mode
        self.normalize = bool(normalize)
        self._weights = dict(weights or {})

    @property
    def weights(self) -> dict[str, float]:
        return dict(self._weights)

    def set_weights(self, weights: Mapping[str, float]) -> None:
        values = np.asarray(list(weights.values()), dtype=float)
        if values.size == 0 or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("weights must be finite non-negative values")
        if float(values.sum()) <= 0:
            raise ValueError("weights must have positive mass")
        self._weights = {str(k): float(v) for k, v in weights.items()}

    @staticmethod
    def _data(item: Any) -> np.ndarray:
        payload = getattr(item, "data", item)
        return np.asarray(payload, dtype=float)

    def fuse(self, latest: Mapping[str, Any]) -> np.ndarray:
        if not latest:
            raise ValueError("fusion requires at least one source")
        keys = tuple(latest)
        raw = np.array([self._weights.get(k, 1.0) for k in keys], dtype=float)
        if np.any(raw < 0) or not np.all(np.isfinite(raw)) or raw.sum() <= 0:
            raise ValueError("configured fusion weights are invalid")
        weights = raw / raw.sum() if self.normalize else raw
        arrays = [self._data(latest[k]) for k in keys]

        if self.mode == "weighted_mean":
            shape = arrays[0].shape
            if any(arr.shape != shape for arr in arrays):
                raise ValueError("weighted_mean requires equal input shapes")
            return np.sum(
                np.stack([w * arr for w, arr in zip(weights, arrays)], axis=0),
                axis=0,
            )

        return np.concatenate([(w * arr).reshape(-1) for w, arr in zip(weights, arrays)])
