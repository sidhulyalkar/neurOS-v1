#!/usr/bin/env python3
"""Run native neurOS EEGNet vs upstream Braindecode EEGNet under one authority.

This is an isolated paired evidence study. It intentionally does not modify the
canonical seven-lane longitudinal model ladder. Every native/external pair
restores the same serialized subject/session/calibration authority before
fitting and is evaluated on the exact same final examples.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
    ordered_group_values,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.longitudinal_external import (
    ExternalTaskDecoderMethodSpec,
    pair_task_performance,
    run_external_task_decoder_case,
)
from neuros.foundation_models.longitudinal_methods import (
    TaskDecoderMethodSpec,
    run_task_decoder_case,
)
from neuros.foundation_models.moabb_longitudinal import (
    MOABB_LONGITUDINAL_DATASETS,
    build_moabb_longitudinal_dataset,
    validate_observed_sessions,
)
from neuros.foundation_models.real_world import collect_moabb, hold_out_groups


def _parse_int_list(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be integers") from exc


def _parse_text_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _stable_seed(base: int, *parts: Any) -> int:
    raw = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(raw.encode("utf-8")).digest()[:4], "big")


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _versions() -> dict[str, str | None]:
    names = (
        "neuros-foundation",
        "neuros-core",
        "neuros-models",
        "braindecode",
        "moabb",
        "mne",
        "skorch",
        "scikit-learn",
        "torch",
        "numpy",
    )
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise ValueError("cannot write an empty paired result table")
    fields = sorted({key for row in materialized for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in materialized:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _mean(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if len(array) == 0 else float(array.mean())


def _std(values: Sequence[float]) -> float | None:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return 0.0 if len(array) == 1 else float(array.std(ddof=1))


def _case_set_fingerprint(case_ids: Sequence[str]) -> str:
    raw = json.dumps(sorted(set(case_ids)), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _summary(rows: Sequence[Mapping[str, Any]], budgets: Sequence[int]) -> dict[str, Any]:
    """Collapse optimization seeds inside case before summarizing cases.

    This avoids treating repeated training seeds from one subject/session as
    independent deployment units. The summary remains descriptive; promoted
    inferential analysis should use participant-aware repeated-measures models.
    """

    by_budget_case: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_budget_case[(int(row["calibration_per_class"]), str(row["case_id"]))].append(row)

    metric_names = (
        "delta_external_minus_native_accuracy",
        "delta_external_minus_native_balanced_accuracy",
        "delta_external_minus_native_roc_auc",
        "delta_external_minus_native_brier_score",
        "delta_external_minus_native_expected_calibration_error",
        "fit_ratio_external_over_native",
        "inference_ratio_external_over_native",
    )
    budget_rows: list[dict[str, Any]] = []
    case_sets: list[str] = []
    for budget in budgets:
        case_values: dict[str, dict[str, float | None]] = {}
        seed_pairs = 0
        for (row_budget, case_id), values in by_budget_case.items():
            if row_budget != int(budget):
                continue
            seed_pairs += len(values)
            collapsed: dict[str, float | None] = {}
            for metric in metric_names:
                numeric = [
                    float(row[metric])
                    for row in values
                    if row.get(metric) is not None and np.isfinite(float(row[metric]))
                ]
                collapsed[metric] = _mean(numeric)
            case_values[case_id] = collapsed

        case_ids = sorted(case_values)
        fingerprint = _case_set_fingerprint(case_ids)
        case_sets.append(fingerprint)
        record: dict[str, Any] = {
            "calibration_per_class": int(budget),
            "n_cases": len(case_ids),
            "n_seed_pairs": seed_pairs,
            "case_set_fingerprint": fingerprint,
        }
        for metric in metric_names:
            values = [
                float(case_values[case_id][metric])
                for case_id in case_ids
                if case_values[case_id][metric] is not None
            ]
            record[f"mean_case_{metric}"] = _mean(values)
            record[f"std_case_{metric}"] = _std(values)
        bal_deltas = [
            float(case_values[case_id]["delta_external_minus_native_balanced_accuracy"])
            for case_id in case_ids
            if case_values[case_id]["delta_external_minus_native_balanced_accuracy"] is not None
        ]
        record["external_balanced_accuracy_case_win_fraction"] = (
            None if not bal_deltas else float(np.mean(np.asarray(bal_deltas) > 0.0))
        )
        budget_rows.append(record)

    unique_case_sets = sorted(set(case_sets))
    return {
        "schema_version": 1,
        "descriptive_only": True,
        "paired_case_set_constant_across_budgets": len(unique_case_sets) == 1,
        "case_set_fingerprints": unique_case_sets,
        "budget_summary": budget_rows,
        "interpretation": [
            "optimization seeds are averaged within subject/session case before case-level summaries",
            "positive accuracy deltas favor upstream Braindecode; negative Brier/ECE deltas favor upstream calibration",
            "runtime ratios above 1 mean Braindecode is slower than native neurOS in this environment",
            "descriptive case means are not participant-independent inferential statistics",
        ],
    }


def _render_report(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    lines = [
        "# Native neurOS EEGNet vs Braindecode EEGNet",
        "",
        "This report is generated from one frozen longitudinal evidence bundle.",
        "It is a paired implementation/ecosystem comparison, not a claim that the two",
        "EEGNet implementations have identical architecture or parameterization.",
        "",
        "## Study identity",
        "",
        f"- Dataset: `{manifest['dataset_id']}`",
        f"- History policy: `{manifest['history_policy']}`",
        f"- Subjects: `{manifest['subjects']}`",
        f"- Calibration budgets / class: `{manifest['budgets_per_class']}`",
        f"- Model seeds: `{manifest['model_seeds']}`",
        f"- Explicit resample rate: `{manifest['resample_hz']} Hz`",
        f"- Git revision: `{manifest['git_revision']}`",
        "",
        "## Paired results",
        "",
        "| budget/class | cases | seeds | Δ balanced accuracy | Δ Brier | Δ ECE | fit ratio | inference ratio |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["budget_summary"]:
        def fmt(value: Any) -> str:
            return "n/a" if value is None else f"{float(value):.4f}"

        lines.append(
            "| {budget} | {cases} | {seeds} | {bal} | {brier} | {ece} | {fit} | {infer} |".format(
                budget=row["calibration_per_class"],
                cases=row["n_cases"],
                seeds=row["n_seed_pairs"],
                bal=fmt(row["mean_case_delta_external_minus_native_balanced_accuracy"]),
                brier=fmt(row["mean_case_delta_external_minus_native_brier_score"]),
                ece=fmt(row["mean_case_delta_external_minus_native_expected_calibration_error"]),
                fit=fmt(row["mean_case_fit_ratio_external_over_native"]),
                infer=fmt(row["mean_case_inference_ratio_external_over_native"]),
            )
        )
    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "- Every pair restores the same serialized source/calibration/evaluation authority.",
            "- Final evaluation examples do not enter fitting or model selection.",
            "- Sampling frequency is explicit and fingerprinted rather than inferred from window length.",
            "- Both methods use the same training epochs, batch size, learning rate, weight decay, device class, and model seed.",
            "- Architecture defaults and parameter counts remain visible and are not represented as matched architecture.",
            "- Braindecode representation and mechanistic evidence remain unavailable in this study.",
            "- This is offline real-dataset evidence, not hardware, closed-loop, clinical, or biological-mechanism evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run native EEGNet and Braindecode EEGNet under identical longitudinal authority."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(MOABB_LONGITUDINAL_DATASETS),
        default="kumar2024",
    )
    parser.add_argument("--subjects", type=_parse_int_list, default=[1])
    parser.add_argument("--held-out-sessions", type=_parse_text_list, default=None)
    parser.add_argument("--model-seeds", type=_parse_int_list, default=[101, 503, 1601])
    parser.add_argument("--budgets", type=_parse_int_list, default=[0, 1, 2, 5, 10])
    parser.add_argument("--history-policy", choices=("prior", "all-other"), default="prior")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--evaluation-fraction", type=float, default=0.5)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument(
        "--resample",
        type=float,
        default=128.0,
        help="Explicit common sample rate; required for physical window-duration identity.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.evaluation_fraction < 1.0:
        raise SystemExit("--evaluation-fraction must lie strictly between 0 and 1")
    if args.fmin <= 0 or args.fmax <= args.fmin:
        raise SystemExit("require 0 < --fmin < --fmax")
    if not np.isfinite(args.resample) or args.resample <= 0:
        raise SystemExit("--resample must be finite and positive")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise SystemExit("require positive --learning-rate and non-negative --weight-decay")

    budgets = tuple(sorted(set(int(value) for value in args.budgets)))
    if not budgets or budgets[0] < 0:
        raise SystemExit("--budgets must contain non-negative values")
    if 0 not in budgets:
        budgets = (0, *budgets)
    model_seeds = tuple(dict.fromkeys(int(value) for value in args.model_seeds))
    if not model_seeds:
        raise SystemExit("at least one --model-seeds value is required")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output / "study_manifest.json",
        "authority": output / "split_authority.json",
        "native": output / "native_runs.json",
        "external": output / "external_runs.json",
        "pairs": output / "paired_runs.json",
        "results": output / "paired_results.csv",
        "summary": output / "summary.json",
        "report": output / "report.md",
        "hashes": output / "artifact_hashes.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "refusing to overwrite existing artifacts: "
            + ", ".join(path.name for path in existing)
        )

    dataset_spec, dataset, paradigm = build_moabb_longitudinal_dataset(
        args.dataset,
        fmin=float(args.fmin),
        fmax=float(args.fmax),
        resample=float(args.resample),
    )
    authorities: list[LongitudinalCaseAuthority] = []
    native_runs: list[dict[str, Any]] = []
    external_runs: list[dict[str, Any]] = []
    paired_runs: list[dict[str, Any]] = []
    paired_rows: list[dict[str, Any]] = []
    started_study = time.perf_counter()

    common_training = {
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "n_epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": str(args.device),
    }

    for subject in args.subjects:
        data = collect_moabb(
            dataset,
            paradigm,
            subjects=[int(subject)],
            dataset_id=dataset_spec.source_id,
        )
        observed = validate_observed_sessions(
            dataset_spec,
            ordered_group_values(data, split_unit="session"),
        )
        if len(observed) < 2:
            raise RuntimeError(f"subject {subject} has fewer than two usable sessions")

        if args.held_out_sessions is not None:
            missing = [value for value in args.held_out_sessions if value not in observed]
            if missing:
                raise RuntimeError(
                    f"subject {subject} missing requested session(s) {missing}; observed={list(observed)}"
                )
            targets = tuple(args.held_out_sessions)
        elif args.history_policy == "prior":
            targets = observed[1:]
        else:
            targets = observed

        for target in targets:
            if args.history_policy == "prior":
                partition = chronological_partition(
                    data,
                    split_unit="session",
                    held_out_value=target,
                    order=observed,
                )
            else:
                partition = hold_out_groups(
                    data,
                    split_unit="session",
                    held_out_values=[target],
                )
            case_split_seed = _stable_seed(
                args.split_seed,
                dataset_spec.source_id,
                subject,
                target,
            )
            split = make_nested_calibration_split(
                partition,
                evaluation_fraction=float(args.evaluation_fraction),
                seed=case_split_seed,
            )
            if budgets[-1] > split.max_budget_per_class:
                raise RuntimeError(
                    f"strict paired frontier requires {budgets[-1]}/class, but "
                    f"subject={subject}, session={target} supports only "
                    f"{split.max_budget_per_class}/class"
                )

            metadata = dataset_spec.case_metadata(int(subject))
            metadata.update(
                {
                    "held_out_session": str(target),
                    "split_seed": int(case_split_seed),
                }
            )
            authority = LongitudinalCaseAuthority.from_split(
                split,
                case_id=(
                    f"{dataset_spec.source_id}/subject-{subject}/session-{target}/"
                    f"split-{case_split_seed}"
                ),
                history_policy=args.history_policy,
                observed_group_order=observed,
                case_metadata=metadata,
            )
            authority.restore(data)
            authorities.append(authority)

            for model_seed in model_seeds:
                native = run_task_decoder_case(
                    data,
                    authority,
                    spec=TaskDecoderMethodSpec(
                        "eegnet",
                        model_seed=int(model_seed),
                        model_kwargs=common_training,
                    ),
                    budgets_per_class=budgets,
                )
                external = run_external_task_decoder_case(
                    data,
                    authority,
                    spec=ExternalTaskDecoderMethodSpec(
                        "braindecode-eegnet",
                        model_seed=int(model_seed),
                        sample_rate_hz=float(args.resample),
                        model_kwargs=common_training,
                    ),
                    budgets_per_class=budgets,
                )
                paired = pair_task_performance(native, external)

                native_payload = native.to_dict()
                external_payload = external.to_dict()
                paired_payload = paired.to_dict()
                native_runs.append(native_payload)
                external_runs.append(external_payload)
                paired_runs.append(paired_payload)

                for raw in paired.rows:
                    row = dict(raw)
                    row.update(dict(authority.case_metadata))
                    row["held_out_session"] = authority.held_out_values[0]
                    row["history_policy"] = authority.history_policy
                    row["model_seed"] = int(model_seed)
                    row["split_seed"] = int(case_split_seed)
                    row["pair_fingerprint"] = paired.pair_fingerprint
                    paired_rows.append(row)

    summary = _summary(paired_rows, budgets)
    if not summary["paired_case_set_constant_across_budgets"]:
        raise RuntimeError("paired case membership changed across calibration budgets")

    manifest = {
        "schema_version": 1,
        "evidence_tier": "real_dataset",
        "study": "native_vs_braindecode_eegnet_longitudinal_pair",
        "dataset_key": dataset_spec.key,
        "dataset_class": dataset_spec.class_name,
        "dataset_id": dataset_spec.source_id,
        "subjects": [int(value) for value in args.subjects],
        "history_policy": args.history_policy,
        "held_out_sessions_requested": args.held_out_sessions,
        "split_seed_base": int(args.split_seed),
        "evaluation_fraction": float(args.evaluation_fraction),
        "budgets_per_class": list(budgets),
        "model_seeds": list(model_seeds),
        "band_hz": [float(args.fmin), float(args.fmax)],
        "resample_hz": float(args.resample),
        "shared_training_config": common_training,
        "native_method_id": "eegnet",
        "external_method_id": "braindecode-eegnet",
        "authority_fingerprints": [item.authority_fingerprint for item in authorities],
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _versions(),
        "wall_time_s": float(time.perf_counter() - started_study),
        "claim_boundary": [
            "native and external methods restore byte-identical processed-data and sample authority",
            "prior history policy excludes future sessions when selected",
            "final evaluation examples never enter fitting or model selection",
            "sampling frequency is explicit and fingerprinted",
            "training epochs, batch size, optimizer hyperparameters, device and seeds are paired",
            "EEGNet architecture defaults are not claimed to be parameter-matched across implementations",
            "Braindecode representation and mechanistic surfaces are not claimed by this study",
            "offline real-dataset evidence is not hardware, closed-loop, clinical or biological evidence",
        ],
    }

    _json_dump(paths["manifest"], manifest)
    _json_dump(
        paths["authority"],
        {"schema_version": 1, "cases": [item.to_dict() for item in authorities]},
    )
    _json_dump(paths["native"], {"schema_version": 1, "runs": native_runs})
    _json_dump(paths["external"], {"schema_version": 1, "runs": external_runs})
    _json_dump(paths["pairs"], {"schema_version": 1, "runs": paired_runs})
    _write_csv(paths["results"], paired_rows)
    _json_dump(paths["summary"], summary)
    paths["report"].write_text(_render_report(manifest, summary), encoding="utf-8")
    _json_dump(
        paths["hashes"],
        {
            "sha256": {
                path.name: _sha256(path)
                for key, path in paths.items()
                if key != "hashes"
            }
        },
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "authority_cases": len(authorities),
                "paired_rows": len(paired_rows),
                "case_sets_paired": summary["paired_case_set_constant_across_budgets"],
                "artifacts": [path.name for path in paths.values()],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
