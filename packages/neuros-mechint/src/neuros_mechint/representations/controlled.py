"""Controlled temporal-manifold fixtures for representation qualification."""
from __future__ import annotations

import math

import numpy as np

from .contracts import SequenceBatch


def latent_trajectory(n: int, phase: float) -> np.ndarray:
    """Generate the fixed smooth latent trajectory used by controlled benchmarks."""
    if isinstance(n, bool) or int(n) != n or n < 3:
        raise ValueError("n must be an integer >= 3")
    phase = float(phase)
    if not math.isfinite(phase):
        raise ValueError("phase must be finite")
    t = np.linspace(0.0, 4.0 * np.pi, int(n), endpoint=False) + phase
    return np.column_stack(
        [
            np.cos(t),
            np.sin(t),
            0.35 * np.sin(2.0 * t),
        ]
    )


def observations(
    latent: np.ndarray,
    *,
    rng: np.random.Generator,
    mixing: np.ndarray,
    noise: float,
) -> np.ndarray:
    """Map latent state nonlinearly into observed space and add Gaussian noise."""
    latent = np.asarray(latent)
    mixing = np.asarray(mixing)
    noise = float(noise)
    if latent.ndim != 2 or latent.shape[1] != 3:
        raise ValueError("latent must have shape [time, 3]")
    if mixing.ndim != 2 or mixing.shape[0] != 6:
        raise ValueError("mixing must have shape [6, observed_features]")
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    nonlinear = np.column_stack(
        [
            latent,
            latent[:, 0] * latent[:, 1],
            latent[:, 0] ** 2 - latent[:, 1] ** 2,
            np.sin(1.5 * latent[:, 2]),
        ]
    )
    return nonlinear @ mixing + rng.normal(scale=noise, size=(latent.shape[0], mixing.shape[1]))


def build_controlled_temporal_manifold(
    noise: float,
    seed: int,
    *,
    train_timepoints: int = 160,
    evaluation_timepoints: int = 140,
    observed_features: int = 24,
) -> tuple[SequenceBatch, SequenceBatch, SequenceBatch]:
    """Build deterministic train/evaluation observations plus clean evaluation geometry."""
    noise = float(noise)
    if not math.isfinite(noise) or noise < 0:
        raise ValueError("noise must be finite and nonnegative")
    if isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    for name, value in (
        ("train_timepoints", train_timepoints),
        ("evaluation_timepoints", evaluation_timepoints),
        ("observed_features", observed_features),
    ):
        if isinstance(value, bool) or int(value) != value:
            raise TypeError(f"{name} must be an integer")
        if int(value) < (3 if name != "observed_features" else 1):
            raise ValueError(f"{name} is too small")

    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(6, int(observed_features)))
    train_latent = latent_trajectory(int(train_timepoints), phase=0.0)
    eval_latent = latent_trajectory(int(evaluation_timepoints), phase=0.37)
    common_metadata = {
        "generator": "controlled_temporal_manifold",
        "generator_version": 1,
        "noise_std": noise,
        "seed": seed,
    }
    train = SequenceBatch(
        sequences=(observations(train_latent, rng=rng, mixing=mixing, noise=noise),),
        sequence_ids=("train",),
        metadata={**common_metadata, "role": "train_observed"},
    )
    evaluation = SequenceBatch(
        sequences=(observations(eval_latent, rng=rng, mixing=mixing, noise=noise),),
        sequence_ids=("eval",),
        metadata={**common_metadata, "role": "evaluation_observed"},
    )
    reference = SequenceBatch(
        sequences=(eval_latent,),
        sequence_ids=("eval",),
        metadata={
            "authority": "known_clean_latent_geometry",
            "generator": "controlled_temporal_manifold",
            "generator_version": 1,
            "seed": seed,
        },
    )
    return train, evaluation, reference
