"""Run a failure-preserving multi-seed neural representation noise sweep.

T-PHATE remains an optional separately installed dependency. When unavailable,
its method × seed × noise records remain in the exported evidence with explicit
``unavailable`` status rather than being dropped.
"""
from __future__ import annotations

import argparse
import json

from neuros_mechint.representations import (
    AutoencoderRepresentation,
    PCARepresentation,
    TPHATERepresentation,
    run_controlled_noise_sweep,
)


def _floats(value: str) -> tuple[float, ...]:
    try:
        parsed = tuple(
            float(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated real values"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one value is required")
    return parsed


def _ints(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(
            int(item.strip()) for item in value.split(",") if item.strip()
        )
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected comma-separated integer values"
        ) from exc
    if not parsed:
        raise argparse.ArgumentTypeError("at least one seed is required")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--noise-levels",
        type=_floats,
        default=(0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
        help="Comma-separated observation-noise standard deviations.",
    )
    parser.add_argument(
        "--seeds",
        type=_ints,
        default=tuple(range(10)),
        help="Comma-separated independent controlled-data seeds.",
    )
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--ae-hidden-dim", type=int, default=32)
    parser.add_argument("--ae-epochs", type=int, default=10)
    parser.add_argument("--ae-batch-size", type=int, default=64)
    parser.add_argument("--ae-learning-rate", type=float, default=1e-3)
    parser.add_argument("--model-seed", type=int, default=17)
    parser.add_argument("--neighborhood-k", type=int, default=7)
    parser.add_argument(
        "--exclude-tphate",
        action="store_true",
        help="Omit T-PHATE entirely instead of preserving unavailable rows.",
    )
    return parser


def _serializable(result) -> dict[str, object]:
    summaries: list[dict[str, object]] = []
    for summary in result.summaries():
        summaries.append(
            {
                "method_id": summary.method_id,
                "fit_regime": summary.fit_regime.value,
                "noise_std": summary.noise_std,
                "total_cases": summary.total_cases,
                "ok_cases": summary.ok_cases,
                "failed_cases": summary.failed_cases,
                "unavailable_cases": summary.unavailable_cases,
                "nonconverged_cases": summary.nonconverged_cases,
                "failure_rate": summary.failure_rate,
                "metric_mean": dict(summary.metric_mean),
                "metric_std": dict(summary.metric_std),
                "metric_sem": dict(summary.metric_sem),
                "metadata": dict(summary.metadata),
            }
        )
    records: list[dict[str, object]] = []
    for record in result.records:
        records.append(
            {
                "noise_std": record.noise_std,
                "seed": record.seed,
                "method_id": record.method_id,
                "sequence_id": record.sequence_id,
                "fit_regime": record.fit_regime.value,
                "status": record.status.value,
                "metrics": dict(record.metrics),
                "error_type": record.error_type,
                "error_message": record.error_message,
            }
        )
    return {
        "schema": result.metadata["schema"],
        "metadata": dict(result.metadata),
        "noise_levels": list(result.noise_levels),
        "seeds": list(result.seeds),
        "method_ids": list(result.method_ids),
        "evaluation_sequence_ids": list(result.evaluation_sequence_ids),
        "summaries": summaries,
        "records": records,
    }


def main() -> None:
    args = build_parser().parse_args()

    def method_factory():
        methods = [
            PCARepresentation(args.components),
            AutoencoderRepresentation(
                n_components=args.components,
                hidden_dim=args.ae_hidden_dim,
                epochs=args.ae_epochs,
                batch_size=args.ae_batch_size,
                learning_rate=args.ae_learning_rate,
                seed=args.model_seed,
            ),
        ]
        if not args.exclude_tphate:
            methods.append(
                TPHATERepresentation(
                    n_components=args.components,
                    random_state=args.model_seed,
                )
            )
        return tuple(methods)

    result = run_controlled_noise_sweep(
        method_factory,
        noise_levels=args.noise_levels,
        seeds=args.seeds,
        neighborhood_k=args.neighborhood_k,
    )
    print(json.dumps(_serializable(result), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
