"""Run the controlled multi-seed/noise representation sweep from issue #143."""
from __future__ import annotations

import argparse
import hashlib
import json
import os

from neuros_mechint.representations import (
    AutoencoderRepresentation,
    PCARepresentation,
    RepresentationBenchmark,
    SweepCase,
    TPHATERepresentation,
    build_controlled_temporal_manifold,
    build_representation_sweep,
)

_DEFAULT_NOISE_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0)
_DEFAULT_SEEDS = tuple(range(10))


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise",
        type=float,
        action="append",
        dest="noise_grid",
        help="Repeat to override the default seven-point noise grid.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help="Repeat to override the default deterministic seeds 0..9.",
    )
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--ae-epochs", type=int, default=4)
    parser.add_argument("--neighborhood-k", type=int, default=8)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    noise_grid = tuple(args.noise_grid) if args.noise_grid is not None else _DEFAULT_NOISE_GRID
    seeds = tuple(args.seeds) if args.seeds is not None else _DEFAULT_SEEDS
    if not noise_grid or not seeds:
        raise ValueError("noise and seed grids must be non-empty")
    if len(set(noise_grid)) != len(noise_grid):
        raise ValueError("noise grid must not contain duplicates")
    if len(set(seeds)) != len(seeds):
        raise ValueError("seed grid must not contain duplicates")

    case_results = []
    for noise in noise_grid:
        for seed in seeds:
            case = SweepCase(noise, seed)
            train, evaluation, reference = build_controlled_temporal_manifold(
                case.noise_std,
                case.seed,
            )
            methods = (
                PCARepresentation(args.components),
                AutoencoderRepresentation(
                    args.components,
                    hidden_dim=32,
                    epochs=args.ae_epochs,
                    batch_size=64,
                    seed=case.seed,
                ),
                TPHATERepresentation(args.components, random_state=case.seed),
            )
            result = RepresentationBenchmark(
                methods,
                neighborhood_k=args.neighborhood_k,
            ).run(train, evaluation, reference=reference)
            case_results.append((case, result))

    sweep = build_representation_sweep(
        case_results,
        metadata={
            "study": "controlled_temporal_manifold_noise_sweep",
            "study_version": 2,
            "source_revision": os.environ.get("GITHUB_SHA", "local-unspecified"),
            "components": args.components,
            "autoencoder_epochs": args.ae_epochs,
            "neighborhood_k": args.neighborhood_k,
            "noise_grid": list(noise_grid),
            "seeds": list(seeds),
            "information_boundary": (
                "Known clean latent geometry is available only because this is a controlled "
                "synthetic qualification study; it is not biological ground truth."
            ),
        },
    )

    scientific_payload: dict[str, object] = {
        "schema_version": 2,
        "kind": "neuros_controlled_representation_noise_sweep_scientific",
        **sweep.to_scientific_dict(),
    }
    scientific_fingerprint = _fingerprint(scientific_payload)

    payload: dict[str, object] = {
        "schema_version": 2,
        "kind": "neuros_controlled_representation_noise_sweep",
        **sweep.to_dict(),
        "scientific_fingerprint": scientific_fingerprint,
    }
    payload["fingerprint"] = _fingerprint(payload)
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
