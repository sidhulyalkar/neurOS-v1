"""Run a controlled autoencoder optimization-horizon sweep on synthetic temporal geometry."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from neuros_mechint.representations import (
    AutoencoderRepresentation,
    MethodStatus,
    PCARepresentation,
    RepresentationBenchmark,
    SweepCase,
    build_controlled_temporal_manifold,
    build_representation_sweep,
)

_DEFAULT_NOISE_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
_DEFAULT_SEEDS = tuple(range(10))
_DEFAULT_EPOCH_GRID = (4, 16, 64, 256)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _unique(values: Sequence[Any], *, name: str) -> tuple[Any, ...]:
    normalized = tuple(values)
    if not normalized:
        raise ValueError(f"{name} must be non-empty")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must not contain duplicates")
    return normalized


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("optimization summary requires finite non-empty values")
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": std,
        "sem": float(std / math.sqrt(array.size)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _optimization_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    noise_grid: tuple[float, ...],
    epoch_grid: tuple[int, ...],
    declared_seeds: int,
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for epochs in epoch_grid:
        method_id = f"autoencoder_e{epochs}"
        for noise in noise_grid:
            selected = [
                row
                for row in rows
                if row["method_id"] == method_id and row["noise_std"] == noise
            ]
            losses = [float(row["final_training_loss"]) for row in selected]
            summaries.append(
                {
                    "method_id": method_id,
                    "epochs": epochs,
                    "noise_std": noise,
                    "declared_seeds": declared_seeds,
                    "observed_losses": len(losses),
                    "missing_losses": declared_seeds - len(losses),
                    "final_training_loss": _numeric_summary(losses) if losses else None,
                }
            )
    return summaries


def run_study(
    *,
    noise_grid: Sequence[float] = _DEFAULT_NOISE_GRID,
    seeds: Sequence[int] = _DEFAULT_SEEDS,
    epoch_grid: Sequence[int] = _DEFAULT_EPOCH_GRID,
    components: int = 3,
    neighborhood_k: int = 8,
    source_revision: str | None = None,
) -> dict[str, Any]:
    noise_grid = tuple(float(value) for value in _unique(noise_grid, name="noise grid"))
    seeds = tuple(int(value) for value in _unique(seeds, name="seed grid"))
    epoch_grid = tuple(int(value) for value in _unique(epoch_grid, name="epoch grid"))
    if any(value < 0 or not math.isfinite(value) for value in noise_grid):
        raise ValueError("noise values must be finite and nonnegative")
    if any(value <= 0 for value in epoch_grid):
        raise ValueError("epoch values must be positive")
    if tuple(sorted(epoch_grid)) != epoch_grid:
        raise ValueError("epoch grid must be strictly increasing")
    if components <= 0 or neighborhood_k <= 0:
        raise ValueError("components and neighborhood_k must be positive")

    case_results = []
    optimization_evidence: list[dict[str, Any]] = []
    for noise in noise_grid:
        for seed in seeds:
            case = SweepCase(noise, seed)
            train, evaluation, reference = build_controlled_temporal_manifold(
                case.noise_std,
                case.seed,
            )
            autoencoders = tuple(
                AutoencoderRepresentation(
                    components,
                    hidden_dim=32,
                    epochs=epochs,
                    batch_size=64,
                    seed=case.seed,
                    method_id=f"autoencoder_e{epochs}",
                )
                for epochs in epoch_grid
            )
            methods = (PCARepresentation(components), *autoencoders)
            result = RepresentationBenchmark(
                methods,
                neighborhood_k=neighborhood_k,
            ).run(train, evaluation, reference=reference)
            case_results.append((case, result))

            outcomes = result.by_method()
            for epochs, method in zip(epoch_grid, autoencoders, strict=True):
                outcome = outcomes[method.method_id]
                if outcome.status is not MethodStatus.OK:
                    continue
                loss = method.training_loss_
                if loss is None or not math.isfinite(loss) or loss < 0:
                    raise RuntimeError(
                        f"successful {method.method_id} lacks finite final training loss"
                    )
                optimization_evidence.append(
                    {
                        **case.to_dict(),
                        "method_id": method.method_id,
                        "epochs": epochs,
                        "final_training_loss": float(loss),
                        "diagnostic_scope": "train_reconstruction_optimization",
                    }
                )

    sweep = build_representation_sweep(
        case_results,
        metadata={
            "study": "controlled_autoencoder_optimization_horizon_sweep",
            "study_version": 1,
            "source_revision": source_revision
            or os.environ.get("GITHUB_SHA", "local-unspecified"),
            "components": components,
            "hidden_dim": 32,
            "batch_size": 64,
            "neighborhood_k": neighborhood_k,
            "noise_grid": list(noise_grid),
            "seeds": list(seeds),
            "epoch_grid": list(epoch_grid),
            "anchor_method": "pca",
            "independent_variable": "autoencoder_training_epochs",
            "held_fixed": [
                "synthetic manifold generator",
                "latent dimension",
                "autoencoder hidden width",
                "batch size",
                "learning rate",
                "case seed",
                "neighborhood metric",
            ],
            "ranking_policy": "none",
            "claim_scope": "controlled_autoencoder_optimization_sensitivity",
            "excluded_methods": ["tphate", "temporal_ssl"],
            "information_boundary": (
                "Known clean latent geometry exists only because this is a synthetic optimization "
                "control. The study tests whether short-horizon autoencoder behavior persists as "
                "optimization continues; it does not establish a generally superior representation."
            ),
        },
    )
    optimization_summaries = _optimization_summaries(
        optimization_evidence,
        noise_grid=noise_grid,
        epoch_grid=epoch_grid,
        declared_seeds=len(seeds),
    )

    scientific_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "neuros_autoencoder_optimization_horizon_scientific",
        **sweep.to_scientific_dict(),
        "optimization_evidence": optimization_evidence,
        "optimization_summaries": optimization_summaries,
    }
    scientific_fingerprint = _fingerprint(scientific_payload)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "neuros_autoencoder_optimization_horizon",
        **sweep.to_dict(),
        "optimization_evidence": optimization_evidence,
        "optimization_summaries": optimization_summaries,
        "scientific_fingerprint": scientific_fingerprint,
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=float, action="append", dest="noise_grid")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--epochs", type=int, action="append", dest="epoch_grid")
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--neighborhood-k", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_study(
        noise_grid=args.noise_grid or _DEFAULT_NOISE_GRID,
        seeds=args.seeds or _DEFAULT_SEEDS,
        epoch_grid=args.epoch_grid or _DEFAULT_EPOCH_GRID,
        components=args.components,
        neighborhood_k=args.neighborhood_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
