"""Compare neural representations on a controlled temporal manifold.

The example runs without T-PHATE installed. In that case its outcome is recorded
as unavailable rather than dropped. T-PHATE is an optional external dependency
with upstream license terms that must be reviewed separately.

An optional fixed temporal-SSL embedding can be supplied as an ``.npz`` file
containing an ``eval`` array with shape [time, latent]. This example never trains
or downloads a foundation model, so model/pretraining provenance remains
explicit and user-supplied.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from neuros_mechint.representations import (
    AutoencoderRepresentation,
    PCARepresentation,
    PrecomputedTemporalSSLRepresentation,
    RepresentationBenchmark,
    SequenceBatch,
    TPHATERepresentation,
)


def _latent_trajectory(n: int, phase: float) -> np.ndarray:
    t = np.linspace(0.0, 4.0 * np.pi, n, endpoint=False) + phase
    return np.column_stack(
        [
            np.cos(t),
            np.sin(t),
            0.35 * np.sin(2.0 * t),
        ]
    )


def _observations(
    latent: np.ndarray,
    *,
    rng: np.random.Generator,
    mixing: np.ndarray,
    noise: float,
) -> np.ndarray:
    nonlinear = np.column_stack(
        [
            latent,
            latent[:, 0] * latent[:, 1],
            latent[:, 0] ** 2 - latent[:, 1] ** 2,
            np.sin(1.5 * latent[:, 2]),
        ]
    )
    return nonlinear @ mixing + rng.normal(scale=noise, size=(latent.shape[0], mixing.shape[1]))


def _build_data(noise: float, seed: int) -> tuple[SequenceBatch, SequenceBatch, SequenceBatch]:
    rng = np.random.default_rng(seed)
    mixing = rng.normal(size=(6, 24))
    train_latent = _latent_trajectory(160, phase=0.0)
    eval_latent = _latent_trajectory(140, phase=0.37)
    train = SequenceBatch(
        sequences=(
            _observations(train_latent, rng=rng, mixing=mixing, noise=noise),
        ),
        sequence_ids=("train",),
        metadata={"generator": "controlled_temporal_manifold", "noise_std": noise},
    )
    evaluation = SequenceBatch(
        sequences=(
            _observations(eval_latent, rng=rng, mixing=mixing, noise=noise),
        ),
        sequence_ids=("eval",),
        metadata={"generator": "controlled_temporal_manifold", "noise_std": noise},
    )
    reference = SequenceBatch(
        sequences=(eval_latent,),
        sequence_ids=("eval",),
        metadata={"authority": "known_clean_latent_geometry"},
    )
    return train, evaluation, reference


def _ssl_method(args: argparse.Namespace, evaluation: SequenceBatch):
    if args.ssl_npz is None:
        return None
    with np.load(args.ssl_npz, allow_pickle=False) as payload:
        if "eval" not in payload:
            raise ValueError("--ssl-npz must contain an 'eval' array")
        embedding = np.array(payload["eval"], copy=True)
    if embedding.shape[0] != evaluation.sequences[0].shape[0]:
        raise ValueError(
            "--ssl-npz 'eval' timepoint count must match the generated evaluation sequence"
        )
    return PrecomputedTemporalSSLRepresentation(
        {"eval": embedding},
        model_id=args.ssl_model_id,
        model_version=args.ssl_model_version,
        pretraining_datasets=tuple(args.ssl_pretraining_dataset),
        pretraining_lineage_status=args.ssl_lineage,
        method_id="temporal_ssl",
    )


def _serializable_result(result) -> dict[str, object]:
    methods: dict[str, object] = {}
    for outcome in result.outcomes:
        methods[outcome.method_id] = {
            "fit_regime": outcome.fit_regime.value,
            "status": outcome.status.value,
            "metrics": dict(outcome.metrics),
            "error_type": outcome.error_type,
            "error_message": outcome.error_message,
            "metadata": dict(outcome.metadata),
            "embedding_metadata": (
                dict(outcome.embedding.metadata) if outcome.embedding is not None else None
            ),
        }
    return {
        "train_sequence_ids": list(result.train_sequence_ids),
        "evaluation_sequence_ids": list(result.evaluation_sequence_ids),
        "benchmark_metadata": dict(result.metadata),
        "methods": methods,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--ae-epochs", type=int, default=20)
    parser.add_argument("--ssl-npz", type=Path)
    parser.add_argument("--ssl-model-id", default="external-temporal-ssl")
    parser.add_argument("--ssl-model-version", default="unspecified")
    parser.add_argument(
        "--ssl-lineage",
        choices=(
            "disjoint_verified",
            "overlap_detected",
            "possible_overlap",
            "unknown_lineage",
            "not_audited",
        ),
        default="not_audited",
    )
    parser.add_argument(
        "--ssl-pretraining-dataset",
        action="append",
        default=[],
        help="Repeat for every known pretraining dataset ID.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not np.isfinite(args.noise) or args.noise < 0:
        raise ValueError("--noise must be finite and nonnegative")

    train, evaluation, reference = _build_data(args.noise, args.seed)
    methods = [
        PCARepresentation(args.components),
        AutoencoderRepresentation(
            args.components,
            hidden_dim=32,
            epochs=args.ae_epochs,
            batch_size=64,
            seed=args.seed,
        ),
        TPHATERepresentation(args.components, random_state=args.seed),
    ]
    ssl = _ssl_method(args, evaluation)
    if ssl is not None:
        methods.append(ssl)

    result = RepresentationBenchmark(methods, neighborhood_k=8).run(
        train,
        evaluation,
        reference=reference,
    )
    print(json.dumps(_serializable_result(result), indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
