"""Optional real-data anchoring for synthetic world populations.

This module does not collapse synthetic-vs-real similarity into a universal
'realism score'. It uses neurOS SourceWeigher to estimate which candidate worlds
are most similar to an observed target domain under an explicitly chosen feature
geometry (sensor covariance or representation embeddings).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .runner import ArenaRun


@dataclass(frozen=True)
class RealityAnchorResult:
    world_ids: tuple[str, ...]
    weights: np.ndarray
    distances: tuple[float, ...]
    method: str
    effective_world_count: float
    max_weight: float

    def by_world(self) -> dict[str, float]:
        return {world_id: float(weight) for world_id, weight in zip(self.world_ids, self.weights, strict=True)}

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.synthetic_bci_arena.reality_anchor.v1",
            "method": self.method,
            "weights": self.by_world(),
            "source_distances": {key: float(value) for key, value in zip(self.world_ids, self.distances, strict=True)},
            "effective_world_count": self.effective_world_count,
            "max_weight": self.max_weight,
            "evidence_boundary": (
                "Similarity weighting to an observed domain; weights are not probabilities that a world is physiologically true."
            ),
        }


def _require_sourceweigher():
    try:
        import neuros_sourceweigher
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Reality anchoring requires neuros-sourceweigher. Install `neuros-arena[reality]`."
        ) from exc
    return neuros_sourceweigher


def _as_samples_by_channels(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=float)
    if array.ndim != 2 or min(array.shape) < 2:
        raise ValueError("EEG domain must be a 2-D channels x samples array")
    if not np.all(np.isfinite(array)):
        raise ValueError("EEG domain must be finite")
    return array.T


def anchor_worlds_by_covariance(
    worlds: Mapping[str, ArenaRun],
    target_data_uv: np.ndarray,
    *,
    temperature: float = 1.0,
    shrinkage: float = 1e-3,
) -> RealityAnchorResult:
    """Weight synthetic worlds against target EEG covariance geometry."""
    if not worlds:
        raise ValueError("worlds cannot be empty")
    sourceweigher = _require_sourceweigher()
    source_samples = {
        world_id: _as_samples_by_channels(run.device_output.data_uv)
        for world_id, run in worlds.items()
    }
    target = _as_samples_by_channels(target_data_uv)
    dims = {target.shape[1], *(samples.shape[1] for samples in source_samples.values())}
    if len(dims) != 1:
        raise ValueError("all worlds and target EEG must have the same channel count")
    result = sourceweigher.RiemannianCovarianceWeigher(
        temperature=temperature,
        shrinkage=shrinkage,
    ).estimate(source_samples, target)
    return RealityAnchorResult(
        world_ids=tuple(result.source_ids),
        weights=np.asarray(result.weights, dtype=float),
        distances=tuple(float(value) for value in result.diagnostics.source_distances),
        method=result.diagnostics.method,
        effective_world_count=float(result.diagnostics.effective_sample_size),
        max_weight=float(result.diagnostics.max_weight),
    )


def anchor_worlds_by_embeddings(
    world_embeddings: Mapping[str, np.ndarray],
    target_embeddings: np.ndarray,
    *,
    statistics: tuple[str, ...] = ("mean", "log_std"),
) -> RealityAnchorResult:
    """Weight worlds in a shared neurOS foundation-model representation space.

    Embedding extraction remains the responsibility of ``neuros-foundation`` or
    another model package so Arena does not acquire model-specific preprocessing.
    """
    if not world_embeddings:
        raise ValueError("world_embeddings cannot be empty")
    sourceweigher = _require_sourceweigher()
    result = sourceweigher.RepresentationSourceWeigher(statistics=statistics).estimate(
        source_embeddings=world_embeddings,
        target_embeddings=np.asarray(target_embeddings, dtype=float),
    )
    distances = tuple(float(value) for value in result.diagnostics.source_distances)
    return RealityAnchorResult(
        world_ids=tuple(result.source_ids),
        weights=np.asarray(result.weights, dtype=float),
        distances=distances,
        method=result.diagnostics.method,
        effective_world_count=float(result.diagnostics.effective_sample_size),
        max_weight=float(result.diagnostics.max_weight),
    )
