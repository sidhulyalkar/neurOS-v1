"""Run a matched temporal-objective control on controlled trajectory geometry."""
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
    LagPredictiveAutoencoderRepresentation,
    MethodStatus,
    PCARepresentation,
    RepresentationBenchmark,
    SweepCase,
    build_controlled_temporal_manifold,
    build_representation_sweep,
)

_NOISE_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
_SEEDS = tuple(range(10))
_EFFECT_METRICS = {
    "local_knn_preservation": "higher_is_better",
    "pairwise_distance_rank": "higher_is_better",
    "reference_local_knn_preservation": "higher_is_better",
    "reference_pairwise_distance_rank": "higher_is_better",
    "temporal_continuity_ratio": "lower_is_better",
}
_COMPARISONS = (
    ("predictive_vs_shuffled", "predictive_autoencoder", "predictive_shuffled"),
    ("predictive_vs_reconstruction", "predictive_autoencoder", "reconstruction_autoencoder"),
)


def _fingerprint(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _numeric_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(tuple(values), dtype=float)
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary requires finite non-empty values")
    std = float(np.std(array, ddof=1)) if array.size > 1 else 0.0
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "std": std,
        "sem": float(std / math.sqrt(array.size)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def _oriented_effect(
    candidate: float,
    control: float,
    *,
    direction: str,
) -> float:
    if direction == "higher_is_better":
        return candidate - control
    if direction == "lower_is_better":
        return control - candidate
    raise ValueError(f"unsupported direction {direction!r}")


def _matched_effect_rows(
    case: SweepCase,
    result: Any,
) -> list[dict[str, Any]]:
    by_method = result.by_method()
    rows: list[dict[str, Any]] = []
    for comparison_id, candidate_id, control_id in _COMPARISONS:
        candidate = by_method[candidate_id]
        control = by_method[control_id]
        for metric, direction in _EFFECT_METRICS.items():
            row: dict[str, Any] = {
                **case.to_dict(),
                "comparison_id": comparison_id,
                "candidate_method_id": candidate_id,
                "control_method_id": control_id,
                "metric": metric,
                "direction": direction,
                "effect_definition": (
                    "candidate_minus_control"
                    if direction == "higher_is_better"
                    else "control_minus_candidate"
                ),
            }
            candidate_value = candidate.metrics.get(metric)
            control_value = control.metrics.get(metric)
            if (
                candidate.status is MethodStatus.OK
                and control.status is MethodStatus.OK
                and candidate_value is not None
                and control_value is not None
            ):
                row["status"] = "ok"
                row["oriented_effect"] = _oriented_effect(
                    float(candidate_value),
                    float(control_value),
                    direction=direction,
                )
            else:
                row["status"] = "not_evaluable"
                row["oriented_effect"] = None
            rows.append(row)
    return rows


def _matched_effect_summaries(
    rows: Sequence[dict[str, Any]],
    *,
    noise_grid: Sequence[float],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for comparison_id, _, _ in _COMPARISONS:
        for noise in noise_grid:
            for metric, direction in _EFFECT_METRICS.items():
                selected = [
                    row
                    for row in rows
                    if row["comparison_id"] == comparison_id
                    and row["noise_std"] == noise
                    and row["metric"] == metric
                ]
                values = [
                    float(row["oriented_effect"])
                    for row in selected
                    if row["oriented_effect"] is not None
                ]
                summaries.append(
                    {
                        "comparison_id": comparison_id,
                        "noise_std": noise,
                        "metric": metric,
                        "direction": direction,
                        "positive_effect_means": "candidate_preserves_more_of_the_declared_geometry",
                        "declared_seeds": len(selected),
                        "evaluable_seeds": len(values),
                        "missing_seeds": len(selected) - len(values),
                        "oriented_effect": _numeric_summary(values) if values else None,
                    }
                )
    return summaries


def run_study(
    *,
    noise_grid: Sequence[float] = _NOISE_GRID,
    seeds: Sequence[int] = _SEEDS,
    epochs: int = 256,
    components: int = 3,
    hidden_dim: int = 32,
    batch_size: int = 64,
    lag: int = 1,
    neighborhood_k: int = 8,
    source_revision: str | None = None,
) -> dict[str, Any]:
    noise_grid = tuple(float(value) for value in noise_grid)
    seeds = tuple(int(value) for value in seeds)
    if not noise_grid or not seeds:
        raise ValueError("noise and seed grids must be non-empty")
    if len(set(noise_grid)) != len(noise_grid) or len(set(seeds)) != len(seeds):
        raise ValueError("noise and seed grids must not contain duplicates")
    if any(not math.isfinite(value) or value < 0 for value in noise_grid):
        raise ValueError("noise values must be finite and nonnegative")
    if epochs <= 0 or components <= 0 or hidden_dim <= 0 or batch_size <= 0 or lag <= 0:
        raise ValueError("all architecture and training integers must be positive")

    case_results = []
    objective_evidence: list[dict[str, Any]] = []
    matched_effects: list[dict[str, Any]] = []

    for noise in noise_grid:
        for seed in seeds:
            case = SweepCase(noise, seed)
            train, evaluation, reference = build_controlled_temporal_manifold(
                case.noise_std,
                case.seed,
            )
            reconstruction = AutoencoderRepresentation(
                components,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                seed=case.seed,
                method_id="reconstruction_autoencoder",
            )
            predictive = LagPredictiveAutoencoderRepresentation(
                components,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                lag=lag,
                seed=case.seed,
                shuffle_targets=False,
                method_id="predictive_autoencoder",
            )
            shuffled = LagPredictiveAutoencoderRepresentation(
                components,
                hidden_dim=hidden_dim,
                epochs=epochs,
                batch_size=batch_size,
                lag=lag,
                seed=case.seed,
                shuffle_targets=True,
                method_id="predictive_shuffled",
            )
            methods = (
                PCARepresentation(components),
                reconstruction,
                predictive,
                shuffled,
            )
            result = RepresentationBenchmark(
                methods,
                neighborhood_k=neighborhood_k,
            ).run(train, evaluation, reference=reference)
            case_results.append((case, result))
            matched_effects.extend(_matched_effect_rows(case, result))

            outcomes = result.by_method()
            objective_specs = (
                (
                    "reconstruction_autoencoder",
                    reconstruction.training_loss_,
                    "same_timepoint_reconstruction_mse",
                    train.sample_count,
                ),
                (
                    "predictive_autoencoder",
                    predictive.training_loss_,
                    "within_sequence_successor_mse",
                    predictive.training_pair_count_,
                ),
                (
                    "predictive_shuffled",
                    shuffled.training_loss_,
                    "within_sequence_shuffled_successor_mse",
                    shuffled.training_pair_count_,
                ),
            )
            for method_id, loss, objective, pair_count in objective_specs:
                if outcomes[method_id].status is not MethodStatus.OK:
                    continue
                if loss is None or not math.isfinite(loss) or loss < 0:
                    raise RuntimeError(f"successful {method_id} lacks a finite objective loss")
                if pair_count is None or int(pair_count) <= 0:
                    raise RuntimeError(f"successful {method_id} lacks a positive fit count")
                objective_evidence.append(
                    {
                        **case.to_dict(),
                        "method_id": method_id,
                        "objective": objective,
                        "final_training_loss": float(loss),
                        "fit_pair_or_sample_count": int(pair_count),
                        "diagnostic_scope": "objective_specific_train_loss_not_cross_objective_score",
                    }
                )

    sweep = build_representation_sweep(
        case_results,
        metadata={
            "study": "controlled_temporal_objective_matched_capacity",
            "study_version": 1,
            "source_revision": source_revision
            or os.environ.get("GITHUB_SHA", "local-unspecified"),
            "noise_grid": list(noise_grid),
            "seeds": list(seeds),
            "components": components,
            "hidden_dim": hidden_dim,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": 1e-3,
            "lag": lag,
            "neighborhood_k": neighborhood_k,
            "ranking_policy": "none",
            "claim_scope": "controlled_temporal_objective_geometry",
            "independent_variable": "training_objective_and_temporal_correspondence",
            "matched_capacity_contract": (
                "reconstruction, predictive, and shuffled-predictive methods share the exact "
                "MLP encoder/decoder shape, latent size, hidden width, Adam learning rate, epoch "
                "budget, batch size, and case seed"
            ),
            "sequence_boundary_policy": "never_cross",
            "primary_temporal_control": "predictive_autoencoder_vs_predictive_shuffled",
            "secondary_objective_control": "predictive_autoencoder_vs_reconstruction_autoencoder",
            "excluded_methods": ["tphate", "external_temporal_ssl"],
            "information_boundary": (
                "Known clean latent geometry exists only because this is a controlled synthetic "
                "study. The shuffled-successor null preserves within-sequence target marginals "
                "while destroying temporal correspondence. This study does not establish "
                "biological, decoding, BCI, or clinical superiority."
            ),
        },
    )
    effect_summaries = _matched_effect_summaries(
        matched_effects,
        noise_grid=noise_grid,
    )

    scientific_payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "neuros_temporal_objective_control_scientific",
        **sweep.to_scientific_dict(),
        "objective_evidence": objective_evidence,
        "matched_effects": matched_effects,
        "matched_effect_summaries": effect_summaries,
    }
    scientific_fingerprint = _fingerprint(scientific_payload)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "neuros_temporal_objective_control",
        **sweep.to_dict(),
        "objective_evidence": objective_evidence,
        "matched_effects": matched_effects,
        "matched_effect_summaries": effect_summaries,
        "scientific_fingerprint": scientific_fingerprint,
    }
    payload["fingerprint"] = _fingerprint(payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=float, action="append", dest="noise_grid")
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--epochs", type=int, default=256)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lag", type=int, default=1)
    parser.add_argument("--neighborhood-k", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = run_study(
        noise_grid=args.noise_grid or _NOISE_GRID,
        seeds=args.seeds or _SEEDS,
        epochs=args.epochs,
        components=args.components,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        lag=args.lag,
        neighborhood_k=args.neighborhood_k,
    )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
