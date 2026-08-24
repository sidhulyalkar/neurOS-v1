#!/usr/bin/env python3
"""Run multiple longitudinal EEG methods under one frozen evidence authority.

For every subject/target-session case the runner freezes source, calibration,
and final-evaluation identities once, serializes the actual indices plus a
processed-data SHA-256, and gives every method exactly that authority.

The script deliberately preserves failures and method-unavailable operating
points. A missing or failed case is evidence, not a row to silently drop.
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
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
    ordered_group_values,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.longitudinal_baseline import run_csp_case
from neuros.foundation_models.longitudinal_methods import (
    TaskDecoderMethodSpec,
    run_task_decoder_case,
)
from neuros.foundation_models.longitudinal_transfer import (
    FrozenTransferMethodSpec,
    run_frozen_transfer_case,
)
from neuros.foundation_models.moabb_longitudinal import (
    MOABB_LONGITUDINAL_DATASETS,
    build_moabb_longitudinal_dataset,
    validate_observed_sessions,
)
from neuros.foundation_models.real_world import collect_moabb, hold_out_groups

_METHODS = (
    "csp-lda",
    "eegnet",
    "eeg-conformer",
    "frozen-eegnet",
    "frozen-eeg-conformer",
    "sourceweigher-eegnet",
    "sourceweigher-eeg-conformer",
)
_SOURCEWEIGHER_METHODS = {
    "sourceweigher-eegnet",
    "sourceweigher-eeg-conformer",
}


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


def _parse_methods(value: str) -> list[str]:
    values = _parse_text_list(value)
    unknown = sorted(set(values) - set(_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown methods {unknown}; available={list(_METHODS)}"
        )
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
        "neuros-sourceweigher",
        "moabb",
        "mne",
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


def _method_spec_payload(
    method: str,
    seed: int | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "method_id": method,
        "model_seed": seed,
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": args.device,
        "csp_components": int(args.csp_components),
        "readout_c": float(args.readout_c),
    }


def _failure_row(
    *,
    authority: LongitudinalCaseAuthority,
    method: str,
    seed: int | None,
    budget: int,
    exc: Exception,
) -> dict[str, Any]:
    return {
        "case_id": authority.case_id,
        "authority_fingerprint": authority.authority_fingerprint,
        "processed_data_sha256": authority.processed_data_sha256,
        "partition_fingerprint": authority.partition_fingerprint,
        "calibration_split_fingerprint": authority.calibration_split_fingerprint,
        "method_id": method,
        "model_seed": seed,
        "calibration_per_class": int(budget),
        "status": "failed",
        "failure_reason": f"{type(exc).__name__}: {exc}",
    }


def _enrich_row(
    row: Mapping[str, Any],
    authority: LongitudinalCaseAuthority,
    *,
    split_seed: int,
) -> dict[str, Any]:
    value = dict(row)
    value.setdefault("status", "ok")
    value.setdefault("failure_reason", None)
    value.setdefault("authority_fingerprint", authority.authority_fingerprint)
    value.setdefault("processed_data_sha256", authority.processed_data_sha256)
    value.setdefault("partition_fingerprint", authority.partition_fingerprint)
    value.setdefault(
        "calibration_split_fingerprint", authority.calibration_split_fingerprint
    )
    value["dataset_id"] = authority.dataset_id
    value["history_policy"] = authority.history_policy
    value["split_seed"] = int(split_seed)
    value.update(dict(authority.case_metadata))
    value["held_out_session"] = authority.held_out_values[0]
    return value


def _run_one_method(
    *,
    method: str,
    data: Any,
    authority: LongitudinalCaseAuthority,
    budgets: tuple[int, ...],
    model_seed: int | None,
    args: argparse.Namespace,
):
    if method == "csp-lda":
        return run_csp_case(
            data,
            authority,
            budgets_per_class=budgets,
            csp_components=args.csp_components,
        )

    if model_seed is None:
        raise ValueError(f"method {method} requires a model seed")

    common_kwargs = {
        "n_epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": args.device,
    }
    if method in {"eegnet", "eeg-conformer"}:
        return run_task_decoder_case(
            data,
            authority,
            spec=TaskDecoderMethodSpec(
                method_id=method,
                model_seed=model_seed,
                model_kwargs=common_kwargs,
            ),
            budgets_per_class=budgets,
        )

    if method.startswith("frozen-"):
        encoder = "eegnet" if method == "frozen-eegnet" else "eeg-conformer"
        return run_frozen_transfer_case(
            data,
            authority,
            spec=FrozenTransferMethodSpec(
                method_id=method,
                strategy="frozen-logistic",
                encoder_id=encoder,
                encoder_seed=model_seed,
                encoder_kwargs=common_kwargs,
                readout_c=args.readout_c,
            ),
            budgets_per_class=budgets,
        )

    if method in _SOURCEWEIGHER_METHODS:
        encoder = "eegnet" if method.endswith("eegnet") else "eeg-conformer"
        return run_frozen_transfer_case(
            data,
            authority,
            spec=FrozenTransferMethodSpec(
                method_id=method,
                strategy="sourceweigher-mean",
                encoder_id=encoder,
                encoder_seed=model_seed,
                encoder_kwargs=common_kwargs,
                readout_c=args.readout_c,
            ),
            budgets_per_class=budgets,
        )

    raise ValueError(f"unsupported method {method!r}")


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return value


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty model-ladder result table")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if len(array) == 0 else float(array.mean())


def _std(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return None
    return float(array.std(ddof=1)) if len(array) > 1 else 0.0


def _seed_averaged_case_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Average optimization seeds inside one frozen subject/session case."""
    ok = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("balanced_accuracy") is not None
    ]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        key = (
            str(row["method_id"]),
            int(row["calibration_per_class"]),
            str(row["case_id"]),
        )
        groups[key].append(row)

    collapsed: list[dict[str, Any]] = []
    for (method, budget, case_id), values in groups.items():
        first = values[0]
        collapsed.append(
            {
                "method_id": method,
                "calibration_per_class": budget,
                "case_id": case_id,
                "subject": first.get("subject"),
                "original_protocol": first.get("original_protocol"),
                "held_out_session": first.get("held_out_session"),
                "balanced_accuracy": _mean(
                    float(value["balanced_accuracy"]) for value in values
                ),
                "roc_auc": _mean(
                    float(value["roc_auc"])
                    for value in values
                    if value.get("roc_auc") is not None
                ),
                "n_model_seeds": len(values),
            }
        )
    return collapsed


def _case_set_fingerprint(case_ids: Sequence[str]) -> str:
    raw = json.dumps(sorted(case_ids), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _group_budget_summary(
    collapsed: list[dict[str, Any]],
    *,
    group_field: str | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in collapsed:
        key: tuple[Any, ...] = (
            row["method_id"],
            row["calibration_per_class"],
        )
        if group_field is not None:
            key = (*key, row.get(group_field))
        groups[key].append(row)

    result: list[dict[str, Any]] = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        method, budget, *group_value = key
        case_ids = [str(row["case_id"]) for row in values]
        record: dict[str, Any] = {
            "method_id": method,
            "calibration_per_class": int(budget),
            "n_cases": len(values),
            "case_set_fingerprint": _case_set_fingerprint(case_ids),
            "mean_balanced_accuracy": _mean(
                float(row["balanced_accuracy"]) for row in values
            ),
            "std_balanced_accuracy": _std(
                float(row["balanced_accuracy"]) for row in values
            ),
            "mean_roc_auc": _mean(
                float(row["roc_auc"])
                for row in values
                if row["roc_auc"] is not None
            ),
        }
        if group_field is not None:
            record[group_field] = group_value[0]
        result.append(record)
    return result


def _manual_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy >=1.24 compatible trapezoidal integration without deprecated aliases."""
    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
        raise ValueError("trapezoid inputs must be aligned 1-D vectors")
    if len(x) < 2:
        return 0.0
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))


def _frontier_auc(
    rows: list[dict[str, Any]],
    requested_budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    """Compute seed-averaged per-case normalized AUC for one declared budget set."""
    if len(requested_budgets) < 2:
        return []
    ok = [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("balanced_accuracy") is not None
    ]
    by_method_case_seed: dict[tuple[str, str, str], dict[int, float]] = defaultdict(dict)
    for row in ok:
        seed = "none" if row.get("model_seed") is None else str(row["model_seed"])
        key = (str(row["method_id"]), str(row["case_id"]), seed)
        by_method_case_seed[key][int(row["calibration_per_class"])] = float(
            row["balanced_accuracy"]
        )

    budget_x = np.asarray(requested_budgets, dtype=np.float64)
    span = float(budget_x[-1] - budget_x[0])
    if span <= 0:
        return []

    per_method_case: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (method, case_id, _seed), curve in by_method_case_seed.items():
        if any(budget not in curve for budget in requested_budgets):
            continue
        y = np.asarray([curve[budget] for budget in requested_budgets], dtype=np.float64)
        auc = _manual_trapezoid(y, budget_x) / span
        per_method_case[(method, case_id)].append(float(auc))

    by_method: dict[str, list[float]] = defaultdict(list)
    for (method, _case), seed_values in per_method_case.items():
        by_method[method].append(float(np.mean(seed_values)))

    return [
        {
            "method_id": method,
            "complete_frontier_cases": len(values),
            "mean_seed_averaged_frontier_auc": _mean(values),
            "std_seed_averaged_frontier_auc": _std(values),
            "requested_budgets": list(requested_budgets),
        }
        for method, values in sorted(by_method.items())
    ]


def _paired_case_set_audit(
    collapsed: list[dict[str, Any]],
    methods: Sequence[str],
    budgets: tuple[int, ...],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method in methods:
        supported = tuple(
            budget
            for budget in budgets
            if not (method in _SOURCEWEIGHER_METHODS and budget == 0)
        )
        fingerprints: list[str] = []
        counts: list[int] = []
        for budget in supported:
            case_ids = [
                str(row["case_id"])
                for row in collapsed
                if row["method_id"] == method
                and row["calibration_per_class"] == budget
            ]
            fingerprints.append(_case_set_fingerprint(case_ids))
            counts.append(len(case_ids))
        identical = bool(fingerprints) and len(set(fingerprints)) == 1
        result.append(
            {
                "method_id": method,
                "supported_budgets": list(supported),
                "case_counts": counts,
                "case_set_fingerprints": fingerprints,
                "paired_across_supported_budgets": identical,
            }
        )
    return result


def _summarize(
    rows: list[dict[str, Any]],
    budgets: tuple[int, ...],
    methods: Sequence[str],
) -> dict[str, Any]:
    failures = [row for row in rows if row.get("status") == "failed"]
    unavailable = [
        row for row in rows if str(row.get("status", "")).startswith("unavailable")
    ]
    unexpected_unavailable = [
        row
        for row in unavailable
        if not (
            row.get("method_id") in _SOURCEWEIGHER_METHODS
            and int(row.get("calibration_per_class", -1)) == 0
        )
    ]
    collapsed = _seed_averaged_case_rows(rows)
    paired_audit = _paired_case_set_audit(collapsed, methods, budgets)
    positive_budgets = tuple(budget for budget in budgets if budget > 0)
    promotion_ready = (
        len(failures) == 0
        and len(unexpected_unavailable) == 0
        and all(item["paired_across_supported_budgets"] for item in paired_audit)
    )
    return {
        "schema_version": 2,
        "primary_metric": "balanced_accuracy",
        "budget_summary_seed_averaged_within_case": _group_budget_summary(collapsed),
        "cohort_budget_summary": _group_budget_summary(
            collapsed, group_field="original_protocol"
        ),
        "target_session_budget_summary": _group_budget_summary(
            collapsed, group_field="held_out_session"
        ),
        "complete_frontier_auc": _frontier_auc(rows, budgets),
        "positive_budget_adaptation_auc": _frontier_auc(rows, positive_budgets),
        "paired_case_set_audit": paired_audit,
        "failed_rows": len(failures),
        "unavailable_rows": len(unavailable),
        "unexpected_unavailable_rows": len(unexpected_unavailable),
        "failure_reasons": sorted(
            {str(row.get("failure_reason")) for row in failures if row.get("failure_reason")}
        ),
        "promotion_ready_descriptive": promotion_ready,
        "statistical_boundary": (
            "Descriptive only. Promoted Kumar2024 inference follows preregistration issue #27: "
            "subject-clustered/hierarchical inference with GR/PAR cohort reporting."
        ),
    }


def _render_report(manifest: dict[str, Any], summary: dict[str, Any]) -> str:
    lines = [
        "# neurOS Longitudinal Model Ladder",
        "",
        f"- Dataset: **{manifest['dataset_key']}**",
        f"- History: **{manifest['history_policy']}**",
        f"- Methods: **{', '.join(manifest['methods'])}**",
        f"- Budgets / class: **{', '.join(str(v) for v in manifest['budgets_per_class'])}**",
        f"- Model seeds: **{', '.join(str(v) for v in manifest['model_seeds'])}**",
        f"- Frozen authority cases: **{len(manifest['authority_fingerprints'])}**",
        f"- Descriptive promotion gate: **{'PASS' if summary['promotion_ready_descriptive'] else 'FAIL'}**",
        "",
        "## Seed-averaged calibration frontier",
        "",
        "| method | calibration/class | cases | case-set fp | balanced accuracy | std | ROC-AUC |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]

    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

    for row in summary["budget_summary_seed_averaged_within_case"]:
        lines.append(
            "| {method_id} | {calibration_per_class} | {n_cases} | {case_set_fingerprint} | "
            "{ba} | {std} | {auc} |".format(
                **row,
                ba=fmt(row["mean_balanced_accuracy"]),
                std=fmt(row["std_balanced_accuracy"]),
                auc=fmt(row["mean_roc_auc"]),
            )
        )

    lines.extend(
        [
            "",
            "## Full calibration-frontier AUC",
            "",
            "This is the preregistered 0-to-max calibration frontier. Methods that are not "
            "defined at zero target calibration, including target-dependent SourceWeigher, "
            "do not receive a fabricated full-frontier AUC.",
            "",
            "| method | complete cases | mean AUC | std |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["complete_frontier_auc"]:
        lines.append(
            "| {method_id} | {complete_frontier_cases} | {mean} | {std} |".format(
                **row,
                mean=fmt(row["mean_seed_averaged_frontier_auc"]),
                std=fmt(row["std_seed_averaged_frontier_auc"]),
            )
        )

    lines.extend(
        [
            "",
            "## Positive-budget adaptation AUC",
            "",
            "Secondary comparison across strictly positive labeled-calibration budgets. This "
            "allows target-dependent adaptation methods to be compared without pretending they "
            "are zero-shot methods.",
            "",
            "| method | complete cases | mean AUC | std |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in summary["positive_budget_adaptation_auc"]:
        lines.append(
            "| {method_id} | {complete_frontier_cases} | {mean} | {std} |".format(
                **row,
                mean=fmt(row["mean_seed_averaged_frontier_auc"]),
                std=fmt(row["std_seed_averaged_frontier_auc"]),
            )
        )

    lines.extend(
        [
            "",
            "## Evidence boundary",
            "",
            "Every method row references a serialized `LongitudinalCaseAuthority` containing "
            "the actual source/calibration/evaluation indices and processed-data SHA-256.",
            "",
            "SourceWeigher rows at zero target calibration are explicitly unavailable. Final "
            "evaluation examples are never repurposed as unlabeled target observations.",
            "",
            f"Failed rows: **{summary['failed_rows']}**. Unavailable rows: **{summary['unavailable_rows']}**. "
            f"Unexpected unavailable rows: **{summary['unexpected_unavailable_rows']}**.",
            "",
            "Kumar2024 cohort and target-session summaries are emitted from the same result "
            "bundle so GR/PAR and session-index effects remain visible.",
            "",
            "This report is descriptive offline real-dataset evidence, not hardware, closed-loop, "
            "clinical, or ORION superiority evidence.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a multi-method longitudinal EEG ladder under frozen sample authority."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(MOABB_LONGITUDINAL_DATASETS),
        default="kumar2024",
    )
    parser.add_argument("--subjects", type=_parse_int_list, default=[1, 10])
    parser.add_argument("--held-out-sessions", type=_parse_text_list, default=None)
    parser.add_argument(
        "--methods",
        type=_parse_methods,
        default=["csp-lda"],
        help=f"Comma-separated subset of: {','.join(_METHODS)}",
    )
    parser.add_argument("--model-seeds", type=_parse_int_list, default=[101, 503, 1601])
    parser.add_argument("--budgets", type=_parse_int_list, default=[0, 1, 2, 5, 10])
    parser.add_argument("--history-policy", choices=("prior", "all-other"), default="prior")
    parser.add_argument("--split-seed", type=int, default=2026)
    parser.add_argument("--evaluation-fraction", type=float, default=0.5)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument("--resample", type=float, default=None)
    parser.add_argument("--csp-components", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--readout-c", type=float, default=1.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0.0 < args.evaluation_fraction < 1.0:
        raise SystemExit("--evaluation-fraction must lie strictly between 0 and 1")
    if args.fmin <= 0 or args.fmax <= args.fmin:
        raise SystemExit("require 0 < --fmin < --fmax")
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    if args.csp_components <= 0 or args.readout_c <= 0:
        raise SystemExit("--csp-components and --readout-c must be positive")

    budgets = tuple(sorted(set(int(value) for value in args.budgets)))
    if not budgets or budgets[0] < 0:
        raise SystemExit("--budgets must contain non-negative values")
    if 0 not in budgets:
        budgets = (0, *budgets)
    methods = tuple(dict.fromkeys(args.methods))
    model_seeds = tuple(dict.fromkeys(int(value) for value in args.model_seeds))
    if any(method != "csp-lda" for method in methods) and not model_seeds:
        raise SystemExit("neural/transfer methods require at least one --model-seeds value")

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "manifest": output / "study_manifest.json",
        "authority": output / "split_authority.json",
        "methods": output / "method_runs.json",
        "results": output / "results.csv",
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
        fmin=args.fmin,
        fmax=args.fmax,
        resample=args.resample,
    )

    authorities: list[LongitudinalCaseAuthority] = []
    result_rows: list[dict[str, Any]] = []
    method_runs: list[dict[str, Any]] = []
    started_study = time.perf_counter()

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
                    f"subject {subject} missing requested session(s) {missing}; "
                    f"observed={list(observed)}"
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
                evaluation_fraction=args.evaluation_fraction,
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

            for method in methods:
                seeds: tuple[int | None, ...] = (
                    (None,) if method == "csp-lda" else tuple(model_seeds)
                )
                for model_seed in seeds:
                    try:
                        run = _run_one_method(
                            method=method,
                            data=data,
                            authority=authority,
                            budgets=budgets,
                            model_seed=model_seed,
                            args=args,
                        )
                        payload = run.to_dict()
                        method_runs.append(
                            {
                                "case_id": authority.case_id,
                                "requested_method_id": method,
                                "requested_model_seed": model_seed,
                                "status": "ok",
                                "result": payload,
                            }
                        )
                        for row in payload["rows"]:
                            enriched = _enrich_row(
                                row,
                                authority,
                                split_seed=case_split_seed,
                            )
                            if enriched.get("method_id") != method:
                                raise RuntimeError(
                                    "method result identity mismatch: "
                                    f"requested={method}, returned={enriched.get('method_id')}"
                                )
                            if model_seed is not None:
                                enriched["model_seed"] = int(model_seed)
                            result_rows.append(enriched)
                    except Exception as exc:
                        method_runs.append(
                            {
                                "case_id": authority.case_id,
                                "requested_method_id": method,
                                "requested_model_seed": model_seed,
                                "status": "failed",
                                "failure_reason": f"{type(exc).__name__}: {exc}",
                                "method_request": _method_spec_payload(
                                    method, model_seed, args
                                ),
                            }
                        )
                        for budget in budgets:
                            result_rows.append(
                                _enrich_row(
                                    _failure_row(
                                        authority=authority,
                                        method=method,
                                        seed=model_seed,
                                        budget=budget,
                                        exc=exc,
                                    ),
                                    authority,
                                    split_seed=case_split_seed,
                                )
                            )

    summary = _summarize(result_rows, budgets, methods)
    manifest = {
        "schema_version": 2,
        "evidence_tier": "real_dataset",
        "study": "longitudinal_eeg_model_ladder",
        "dataset_key": dataset_spec.key,
        "dataset_class": dataset_spec.class_name,
        "dataset_id": dataset_spec.source_id,
        "subjects": [int(value) for value in args.subjects],
        "history_policy": args.history_policy,
        "held_out_sessions_requested": args.held_out_sessions,
        "split_seed_base": int(args.split_seed),
        "evaluation_fraction": float(args.evaluation_fraction),
        "budgets_per_class": list(budgets),
        "methods": list(methods),
        "model_seeds": list(model_seeds),
        "band_hz": [float(args.fmin), float(args.fmax)],
        "resample_hz": args.resample,
        "csp_components": int(args.csp_components),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "device": args.device,
        "readout_c": float(args.readout_c),
        "authority_fingerprints": [item.authority_fingerprint for item in authorities],
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _versions(),
        "wall_time_s": float(time.perf_counter() - started_study),
        "promotion_ready_descriptive": summary["promotion_ready_descriptive"],
        "claim_boundary": [
            "all methods consume serialized frozen sample authority",
            "prior policy excludes future sessions",
            "final evaluation examples never enter fitting or SourceWeigher target estimation",
            "SourceWeigher zero-target-calibration is explicitly unavailable",
            "full frontier AUC is not fabricated for methods undefined at zero calibration",
            "GR/PAR cohort and target-session effects remain visible for Kumar2024",
            "descriptive offline real-dataset evidence is not hardware/closed-loop/clinical evidence",
        ],
    }

    _json_dump(paths["manifest"], manifest)
    _json_dump(
        paths["authority"],
        {
            "schema_version": 2,
            "cases": [authority.to_dict() for authority in authorities],
        },
    )
    _json_dump(paths["methods"], {"schema_version": 2, "runs": method_runs})
    _write_csv(paths["results"], result_rows)
    _json_dump(paths["summary"], summary)
    paths["report"].write_text(_render_report(manifest, summary), encoding="utf-8")

    hashed = {
        path.name: _sha256(path)
        for key, path in paths.items()
        if key != "hashes"
    }
    _json_dump(paths["hashes"], {"sha256": hashed})

    print(
        json.dumps(
            {
                "output": str(output),
                "authority_cases": len(authorities),
                "result_rows": len(result_rows),
                "failed_rows": summary["failed_rows"],
                "unavailable_rows": summary["unavailable_rows"],
                "promotion_ready_descriptive": summary["promotion_ready_descriptive"],
                "artifacts": [path.name for path in paths.values()],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())