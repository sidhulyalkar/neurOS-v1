"""Controlled temporal-manifold fixtures for representation experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from .contracts import SequenceBatch, _freeze_metadata


def _positive_count(value: int, *, name: str, minimum: int = 3) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return value


def _nonnegative_finite(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a finite nonnegative real")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return value


@dataclass(frozen=True, slots=True)
class ControlledTemporalManifold:
    """Train/evaluation observations plus exact clean latent reference geometry."""

    train: SequenceBatch
    evaluation: SequenceBatch
    reference: SequenceBatch
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if self.train.feature_count != self.evaluation.feature_count:
            raise ValueError("controlled train/evaluation feature dimensions must match")
        if self.reference.sequence_ids != self.evaluation.sequence_ids:
            raise ValueError("reference identity must match evaluation identity")
        for observed, reference in zip(
            self.evaluation.sequences,
            self.reference.sequences,
            strict=True,
        ):
            if observed.shape[0] != reference.shape[0]:
                raise ValueError(
                    "reference timepoints must match evaluation timepoints"
                )
        object.__setattr__(self, "metadata", _freeze_metadata(self.metadata))


def latent_trajectory(n: int, *, phase: float) -> np.ndarray:
    """Deterministic 3-D periodic latent trajectory for the controlled fixture."""

    n = _positive_count(n, name="n")
    if isinstance(phase, bool) or not isinstance(
        phase, (int, float, np.integer, np.floating)
    ):
        raise TypeError("phase must be a finite real")
    phase = float(phase)
    if not np.isfinite(phase):
        raise ValueError("phase must be a finite real")
    t = np.linspace(0.0, 4.0 * np.pi, n, endpoint=False) + phase
    latent = np.column_stack(
        [
            np.cos(t),
            np.sin(t),
            0.35 * np.sin(2.0 * t),
        ]
    )
    latent.setflags(write=False)
    return latent


def _nonlinear_features(latent: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            latent,
            latent[:, 0] * latent[:, 1],
            latent[:, 0] ** 2 - latent[:, 1] ** 2,
            np.sin(1.5 * latent[:, 2]),
        ]
    )


def _observations(
    latent: np.ndarray,
    *,
    mixing: np.ndarray,
    standardized_noise: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    return _nonlinear_features(latent) @ mixing + noise_std * standardized_noise


def make_controlled_temporal_manifold(
    *,
    noise_std: float,
    seed: int,
    train_points: int = 160,
    evaluation_points: int = 140,
    observed_features: int = 24,
    evaluation_phase: float = 0.37,
) -> ControlledTemporalManifold:
    """Create one coupled-noise controlled manifold condition.

    For a fixed ``seed``, the mixing matrix and standardized noise draws are
    identical across ``noise_std`` values. A noise sweep therefore changes only
    noise amplitude rather than silently changing the observation mapping.
    """

    noise_std = _nonnegative_finite(noise_std, name="noise_std")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    train_points = _positive_count(train_points, name="train_points")
    evaluation_points = _positive_count(
        evaluation_points, name="evaluation_points"
    )
    observed_features = _positive_count(
        observed_features, name="observed_features", minimum=1
    )
    if isinstance(evaluation_phase, bool) or not isinstance(
        evaluation_phase, (int, float, np.integer, np.floating)
    ):
        raise TypeError("evaluation_phase must be a finite real")
    evaluation_phase = float(evaluation_phase)
    if not np.isfinite(evaluation_phase):
        raise ValueError("evaluation_phase must be a finite real")

    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(6, observed_features))
    train_latent = latent_trajectory(train_points, phase=0.0)
    evaluation_latent = latent_trajectory(
        evaluation_points, phase=evaluation_phase
    )
    train_noise = rng.normal(size=(train_points, observed_features))
    evaluation_noise = rng.normal(size=(evaluation_points, observed_features))

    train = SequenceBatch(
        sequences=(
            _observations(
                train_latent,
                mixing=mixing,
                standardized_noise=train_noise,
                noise_std=noise_std,
            ),
        ),
        sequence_ids=("train",),
        metadata={
            "generator": "controlled_temporal_manifold.v2",
            "noise_std": noise_std,
            "seed": seed,
        },
    )
    evaluation = SequenceBatch(
        sequences=(
            _observations(
                evaluation_latent,
                mixing=mixing,
                standardized_noise=evaluation_noise,
                noise_std=noise_std,
            ),
        ),
        sequence_ids=("eval",),
        metadata={
            "generator": "controlled_temporal_manifold.v2",
            "noise_std": noise_std,
            "seed": seed,
        },
    )
    reference = SequenceBatch(
        sequences=(evaluation_latent,),
        sequence_ids=("eval",),
        metadata={
            "authority": "known_clean_latent_geometry",
            "generator": "controlled_temporal_manifold.v2",
            "seed": seed,
        },
    )
    return ControlledTemporalManifold(
        train=train,
        evaluation=evaluation,
        reference=reference,
        metadata={
            "schema": "neuros.representation.controlled_temporal_manifold.v2",
            "seed": seed,
            "noise_std": noise_std,
            "train_points": train_points,
            "evaluation_points": evaluation_points,
            "observed_features": observed_features,
            "evaluation_phase": evaluation_phase,
            "coupled_noise_policy": (
                "fixed_seed_reuses_mixing_and_standardized_noise_across_noise_levels"
            ),
        },
    )
