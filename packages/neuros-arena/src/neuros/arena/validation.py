"""Held-out validation protocols for reality-anchored synthetic populations.

A simulator must not be tuned and judged on the same recording. These helpers
freeze synthetic-world weights on calibration EEG, then evaluate those weights
against independent EEG under the same declared feature geometry.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .reality import RealityAnchorResult, anchor_worlds_by_covariance
from .runner import ArenaRun


@dataclass(frozen=True)
class HeldOutRealityValidation:
    calibration: RealityAnchorResult
    validation_distances: dict[str, float]
    weighted_validation_distance: float
    uniform_validation_distance: float
    relative_improvement: float
    calibration_validation_distance_correlation: float
    best_calibration_world: str
    best_calibration_world_validation_distance: float
    validation_best_world: str

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.synthetic_bci_arena.heldout_reality_validation.v1",
            "calibration": self.calibration.to_dict(),
            "validation_distances": self.validation_distances,
            "weighted_validation_distance": self.weighted_validation_distance,
            "uniform_validation_distance": self.uniform_validation_distance,
            "relative_improvement": self.relative_improvement,
            "calibration_validation_distance_correlation": self.calibration_validation_distance_correlation,
            "best_calibration_world": self.best_calibration_world,
            "best_calibration_world_validation_distance": self.best_calibration_world_validation_distance,
            "validation_best_world": self.validation_best_world,
            "evidence_boundary": (
                "Out-of-sample domain-similarity validation only. Improvement does not establish human behavioral or decoder performance."
            ),
        }


def split_contiguous_recording(
    data_uv: np.ndarray,
    *,
    calibration_fraction: float = 0.5,
    guard_samples: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Create a leakage-resistant temporal calibration/validation split.

    A contiguous split is intentionally preferred to random sample shuffling for
    nonstationary EEG. ``guard_samples`` can remove data around the split point
    when preprocessing/window overlap would otherwise leak information.
    """
    data = np.asarray(data_uv, dtype=float)
    if data.ndim != 2 or data.shape[1] < 8:
        raise ValueError("expected channels x samples EEG with at least 8 samples")
    if not 0.1 <= calibration_fraction <= 0.9:
        raise ValueError("calibration_fraction must be in [0.1, 0.9]")
    if guard_samples < 0:
        raise ValueError("guard_samples must be non-negative")
    split = int(round(data.shape[1] * calibration_fraction))
    left_stop = split - guard_samples
    right_start = split + guard_samples
    if left_stop < 2 or data.shape[1] - right_start < 2:
        raise ValueError("guard interval leaves too little data for calibration/validation")
    return data[:, :left_stop].copy(), data[:, right_start:].copy()


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or b.size != a.size:
        return 0.0
    if np.std(a) <= 1e-12 or np.std(b) <= 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def validate_covariance_anchor_held_out(
    worlds: Mapping[str, ArenaRun],
    calibration_data_uv: np.ndarray,
    validation_data_uv: np.ndarray,
    *,
    temperature: float = 1.0,
    shrinkage: float = 1e-3,
) -> HeldOutRealityValidation:
    """Fit world weights on calibration EEG and score them on independent EEG."""
    if len(worlds) < 2:
        raise ValueError("held-out validation requires at least two synthetic worlds")
    calibration = anchor_worlds_by_covariance(
        worlds,
        calibration_data_uv,
        temperature=temperature,
        shrinkage=shrinkage,
    )
    # A second anchor computation is used only to obtain validation distances in
    # the identical covariance geometry. Its weights are not used for scoring.
    validation = anchor_worlds_by_covariance(
        worlds,
        validation_data_uv,
        temperature=temperature,
        shrinkage=shrinkage,
    )
    calibration_weights = calibration.by_world()
    calibration_distances = {
        world_id: float(distance)
        for world_id, distance in zip(calibration.world_ids, calibration.distances, strict=True)
    }
    validation_distances = {
        world_id: float(distance)
        for world_id, distance in zip(validation.world_ids, validation.distances, strict=True)
    }
    if set(calibration_weights) != set(validation_distances):
        raise ValueError("calibration and validation world sets differ")
    ids = tuple(calibration.world_ids)
    weights = np.asarray([calibration_weights[world_id] for world_id in ids], dtype=float)
    distances = np.asarray([validation_distances[world_id] for world_id in ids], dtype=float)
    weighted = float(np.dot(weights, distances))
    uniform = float(np.mean(distances))
    relative = float((uniform - weighted) / max(abs(uniform), 1e-12))
    calibration_vector = np.asarray([calibration_distances[world_id] for world_id in ids], dtype=float)
    corr = _correlation(calibration_vector, distances)
    best_calibration = min(ids, key=lambda world_id: calibration_distances[world_id])
    best_validation = min(ids, key=lambda world_id: validation_distances[world_id])
    return HeldOutRealityValidation(
        calibration=calibration,
        validation_distances=validation_distances,
        weighted_validation_distance=weighted,
        uniform_validation_distance=uniform,
        relative_improvement=relative,
        calibration_validation_distance_correlation=corr,
        best_calibration_world=best_calibration,
        best_calibration_world_validation_distance=validation_distances[best_calibration],
        validation_best_world=best_validation,
    )
