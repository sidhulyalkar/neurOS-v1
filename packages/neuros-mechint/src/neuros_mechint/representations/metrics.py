"""Orientation-invariant representation geometry metrics."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr


def _paired_arrays(
    source: np.ndarray,
    embedding: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    embedding = np.asarray(embedding, dtype=np.float64)
    if source.ndim != 2 or embedding.ndim != 2:
        raise ValueError("source and embedding must both be 2-D")
    if source.shape[0] != embedding.shape[0]:
        raise ValueError("source and embedding must have matching timepoints")
    if source.shape[0] < 3:
        raise ValueError("at least three timepoints are required")
    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(embedding)):
        raise ValueError("metric inputs must be finite")
    return source, embedding


def pairwise_distance_rank_preservation(
    source: np.ndarray,
    embedding: np.ndarray,
    *,
    max_points: int = 512,
) -> float | None:
    """Spearman correlation between source and latent pairwise distances."""

    source, embedding = _paired_arrays(source, embedding)
    if isinstance(max_points, bool) or not isinstance(max_points, (int, np.integer)):
        raise TypeError("max_points must be an integer")
    max_points = int(max_points)
    if max_points < 3:
        raise ValueError("max_points must be at least 3")
    if source.shape[0] > max_points:
        indices = np.linspace(0, source.shape[0] - 1, max_points, dtype=int)
        source = source[indices]
        embedding = embedding[indices]
    source_dist = pdist(source)
    embedding_dist = pdist(embedding)
    if np.allclose(source_dist, source_dist[0]) or np.allclose(
        embedding_dist, embedding_dist[0]
    ):
        return None
    statistic = spearmanr(source_dist, embedding_dist).statistic
    if statistic is None or not np.isfinite(statistic):
        return None
    return float(statistic)


def local_neighborhood_preservation(
    source: np.ndarray,
    embedding: np.ndarray,
    *,
    k: int = 5,
) -> float:
    """Mean fraction of source-space kNNs retained in latent space."""

    source, embedding = _paired_arrays(source, embedding)
    if isinstance(k, bool) or not isinstance(k, (int, np.integer)):
        raise TypeError("k must be an integer")
    k = int(k)
    if k <= 0:
        raise ValueError("k must be positive")
    k = min(k, source.shape[0] - 1)
    source_dist = squareform(pdist(source))
    embedding_dist = squareform(pdist(embedding))
    np.fill_diagonal(source_dist, np.inf)
    np.fill_diagonal(embedding_dist, np.inf)
    source_neighbors = np.argpartition(source_dist, kth=k - 1, axis=1)[:, :k]
    embedding_neighbors = np.argpartition(embedding_dist, kth=k - 1, axis=1)[:, :k]
    overlaps = [
        len(set(source_neighbors[row]) & set(embedding_neighbors[row])) / k
        for row in range(source.shape[0])
    ]
    return float(np.mean(overlaps))


def temporal_continuity_ratio(embedding: np.ndarray) -> float | None:
    """Adjacent-step distance divided by nonlocal temporal distance.

    Lower values indicate adjacent latent states are closer than states separated
    by at least two steps. This is descriptive, not a universal optimization
    target: over-smoothing can score well while erasing meaningful transitions.
    """

    embedding = np.asarray(embedding, dtype=np.float64)
    if embedding.ndim != 2 or embedding.shape[0] < 3:
        raise ValueError("embedding must be 2-D with at least three timepoints")
    if not np.all(np.isfinite(embedding)):
        raise ValueError("embedding must contain only finite values")
    adjacent = np.linalg.norm(np.diff(embedding, axis=0), axis=1)
    distances = squareform(pdist(embedding))
    mask = np.triu(np.ones(distances.shape, dtype=bool), k=2)
    nonlocal_distances = distances[mask]
    if nonlocal_distances.size == 0:
        return None
    denominator = float(np.median(nonlocal_distances))
    if denominator <= np.finfo(float).eps:
        return None
    return float(np.median(adjacent) / denominator)


def aggregate_geometry_metrics(
    sources: tuple[np.ndarray, ...],
    embeddings: tuple[np.ndarray, ...],
    *,
    k: int = 5,
) -> dict[str, float | None]:
    """Aggregate trajectory-local metrics without pooling coordinate axes."""

    if len(sources) != len(embeddings) or not sources:
        raise ValueError("sources and embeddings must be aligned and nonempty")
    neighborhood: list[float] = []
    rank: list[float] = []
    continuity: list[float] = []
    for source, embedding in zip(sources, embeddings, strict=True):
        neighborhood.append(local_neighborhood_preservation(source, embedding, k=k))
        rank_value = pairwise_distance_rank_preservation(source, embedding)
        if rank_value is not None:
            rank.append(rank_value)
        continuity_value = temporal_continuity_ratio(embedding)
        if continuity_value is not None:
            continuity.append(continuity_value)
    return {
        "local_knn_preservation": float(np.mean(neighborhood)),
        "pairwise_distance_rank": float(np.mean(rank)) if rank else None,
        "temporal_continuity_ratio": float(np.mean(continuity)) if continuity else None,
    }
