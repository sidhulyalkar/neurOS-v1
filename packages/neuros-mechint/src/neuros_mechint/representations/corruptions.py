"""Controlled temporal corruption fixtures for representation ablations."""
from __future__ import annotations

from enum import Enum

import numpy as np

from .contracts import SequenceBatch
from .synthetic import (
    ControlledTemporalManifold,
    make_controlled_temporal_manifold,
)


class TemporalCorruption(str, Enum):
    IID_GAUSSIAN = "iid_gaussian"
    AR1 = "ar1"
    SPARSE_SPIKES = "sparse_spikes"
    SLOW_DRIFT = "slow_drift"


def _finite_nonnegative(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"{name} must be a finite nonnegative real")
    numeric = float(value)
    if not np.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{name} must be a finite nonnegative real")
    return numeric


def _probability(value: float, *, name: str) -> float:
    numeric = _finite_nonnegative(value, name=name)
    if numeric <= 0 or numeric > 1:
        raise ValueError(f"{name} must be in (0, 1]")
    return numeric


def _unit_scale(array: np.ndarray) -> np.ndarray:
    centered = array - np.mean(array, axis=0, keepdims=True)
    scale = np.std(centered, axis=0, ddof=0, keepdims=True)
    scale = np.where(scale > np.finfo(float).eps, scale, 1.0)
    return centered / scale


def _standardized_corruption(
    kind: TemporalCorruption,
    rng: np.random.Generator,
    shape: tuple[int, int],
    *,
    ar_coefficient: float,
    spike_probability: float,
    drift_cycles: float,
) -> np.ndarray:
    n_rows, n_features = shape
    if kind is TemporalCorruption.IID_GAUSSIAN:
        return rng.normal(size=shape)

    if kind is TemporalCorruption.AR1:
        innovations = rng.normal(size=shape)
        output = np.empty(shape, dtype=np.float64)
        output[0] = innovations[0]
        innovation_scale = float(
            np.sqrt(1.0 - ar_coefficient * ar_coefficient)
        )
        for index in range(1, n_rows):
            output[index] = (
                ar_coefficient * output[index - 1]
                + innovation_scale * innovations[index]
            )
        return output

    if kind is TemporalCorruption.SPARSE_SPIKES:
        mask = rng.random(size=shape) < spike_probability
        amplitudes = rng.normal(size=shape) / np.sqrt(spike_probability)
        return mask * amplitudes

    if kind is TemporalCorruption.SLOW_DRIFT:
        phase = rng.uniform(
            0.0,
            2.0 * np.pi,
            size=(1, n_features),
        )
        amplitude = rng.normal(size=(1, n_features))
        time = np.linspace(
            0.0,
            2.0 * np.pi * drift_cycles,
            n_rows,
            endpoint=False,
        )[:, None]
        return _unit_scale(amplitude * np.sin(time + phase))

    raise ValueError(f"unsupported temporal corruption {kind!r}")


def make_controlled_corruption_manifold(
    *,
    corruption: TemporalCorruption | str,
    corruption_scale: float,
    seed: int,
    train_points: int = 160,
    evaluation_points: int = 140,
    observed_features: int = 24,
    evaluation_phase: float = 0.37,
    ar_coefficient: float = 0.85,
    spike_probability: float = 0.03,
    drift_cycles: float = 1.5,
) -> ControlledTemporalManifold:
    """Create one fixed standardized temporal corruption over clean geometry.

    For a fixed seed and corruption kind, changing ``corruption_scale`` changes
    only corruption amplitude. Clean latent geometry, observation mixing, and
    the standardized corruption draw remain fixed.
    """

    kind = TemporalCorruption(corruption)
    scale = _finite_nonnegative(
        corruption_scale,
        name="corruption_scale",
    )
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be nonnegative")

    rho = _finite_nonnegative(
        ar_coefficient,
        name="ar_coefficient",
    )
    if rho >= 1:
        raise ValueError("ar_coefficient must be in [0, 1)")
    spike_probability = _probability(
        spike_probability,
        name="spike_probability",
    )
    drift_cycles = _finite_nonnegative(
        drift_cycles,
        name="drift_cycles",
    )
    if drift_cycles <= 0:
        raise ValueError("drift_cycles must be positive")

    clean = make_controlled_temporal_manifold(
        noise_std=0.0,
        seed=seed,
        train_points=train_points,
        evaluation_points=evaluation_points,
        observed_features=observed_features,
        evaluation_phase=evaluation_phase,
    )

    kind_code = list(TemporalCorruption).index(kind) + 1
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x4E455552, kind_code])
    )
    train_standardized = _standardized_corruption(
        kind,
        rng,
        clean.train.sequences[0].shape,
        ar_coefficient=rho,
        spike_probability=spike_probability,
        drift_cycles=drift_cycles,
    )
    evaluation_standardized = _standardized_corruption(
        kind,
        rng,
        clean.evaluation.sequences[0].shape,
        ar_coefficient=rho,
        spike_probability=spike_probability,
        drift_cycles=drift_cycles,
    )

    metadata = {
        "generator": "controlled_temporal_corruption.v1",
        "corruption_kind": kind.value,
        "corruption_scale": scale,
        "seed": seed,
        "ar_coefficient": rho,
        "spike_probability": spike_probability,
        "drift_cycles": drift_cycles,
        "coupling_policy": (
            "fixed_seed_and_kind_reuse_clean_mapping_and_standardized_"
            "corruption_across_scales"
        ),
        "base_generator_metadata": dict(clean.metadata),
    }
    train = SequenceBatch(
        sequences=(
            clean.train.sequences[0] + scale * train_standardized,
        ),
        sequence_ids=clean.train.sequence_ids,
        metadata=metadata,
    )
    evaluation = SequenceBatch(
        sequences=(
            clean.evaluation.sequences[0]
            + scale * evaluation_standardized,
        ),
        sequence_ids=clean.evaluation.sequence_ids,
        metadata=metadata,
    )
    return ControlledTemporalManifold(
        train=train,
        evaluation=evaluation,
        reference=clean.reference,
        metadata={
            **metadata,
            "schema": (
                "neuros.representation.controlled_temporal_corruption.v1"
            ),
        },
    )
