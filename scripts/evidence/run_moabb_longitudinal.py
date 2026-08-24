#!/usr/bin/env python3
"""Run a leakage-resistant longitudinal EEG calibration benchmark on MOABB.

The first goal is a transparent evidence floor, not a state-of-the-art claim.
For each subject and held-out session this runner:

1. creates a neurOS deployment-unit-disjoint pre-model partition;
2. by default trains only on sessions observed *before* the held-out session;
3. freezes one class-stratified evaluation set inside the held-out session;
4. freezes the remaining held-out trials as an ordered calibration pool;
5. refits the same CSP+LDA baseline at nested per-class calibration budgets;
6. evaluates every budget on the exact same held-out examples;
7. writes machine-readable manifests, results, summary, hashes, and a report.

Large public datasets are downloaded by MOABB when this script is run. CI does
not download them; contract behavior is tested with synthetic MOABB-shaped data.
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
from typing import Any, Iterable

import numpy as np

from neuros.foundation_models import (
    chronological_partition,
    collect_moabb,
    get_evidence_source,
    hold_out_groups,
    make_nested_calibration_split,
    ordered_group_values,
)


_DATASETS = {
    "kumar2024": {
        "class": "Kumar2024",
        "source_id": "moabb-kumar2024",
        "description": "18 participants x 6 separate-day MI sessions",
        "paradigm": "left_right",
        "events": None,
    },
    "ma2020": {
        "class": "Ma2020",
        "source_id": "moabb-ma2020",
        "description": "25 participants x 15 right-hand/right-elbow MI sessions",
        "paradigm": "motor_imagery",
        "events": ("right_hand", "right_elbow"),
    },
    "lee2019-mi": {
        "class": "Lee2019_MI",
        "source_id": "moabb-lee2019-family",
        "description": "OpenBMI motor-imagery member of the 54-person shared cohort",
        "paradigm": "left_right",
        "events": None,
    },
    "wang2026": {
        "class": "Wang2026",
        "source_id": "moabb-wang2026",
        "description": "39 participants x 5 sessions with online cursor-control study",
        "paradigm": "left_right",
        "events": None,
    },
}


def _json_dump(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stable_int_seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(base), *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:4], "big")


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
    result: dict[str, str | None] = {}
    for distribution in (
        "neuros-foundation",
        "neuros-core",
        "neuros-models",
        "moabb",
        "mne",
        "scikit-learn",
        "numpy",
    ):
        try:
            result[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[distribution] = None
    return result


def _parse_int_list(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    try:
        parsed = [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("values must be integers") from exc
    return parsed


def _parse_text_list(value: str) -> list[str]:
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated value")
    return values


def _dataset_and_paradigm(dataset_key: str, *, fmin: float, fmax: float):
    try:
        import moabb.datasets as datasets
        from moabb.paradigms import LeftRightImagery, MotorImagery
    except ImportError as exc:  # pragma: no cover - exercised by real environment
        raise RuntimeError(
            "This benchmark requires the optional evidence stack. Install "
            "`neuros-foundation[evidence]`."
        ) from exc

    spec = _DATASETS[dataset_key]
    dataset_cls = getattr(datasets, spec["class"], None)
    if dataset_cls is None:
        raise RuntimeError(
            f"Installed MOABB does not expose {spec['class']}. Pin a MOABB release "
            "that contains this dataset or choose another --dataset."
        )
    dataset = dataset_cls()
    if spec["paradigm"] == "left_right":
        paradigm = LeftRightImagery(fmin=float(fmin), fmax=float(fmax))
    elif spec["paradigm"] == "motor_imagery":
        events = list(spec["events"] or ())
        paradigm = MotorImagery(
            n_classes=len(events),
            events=events,
            fmin=float(fmin),
            fmax=float(fmax),
        )
    else:  # pragma: no cover - internal registry invariant
        raise RuntimeError(f"unsupported paradigm kind {spec['paradigm']!r}")

    try:
        valid = paradigm.is_valid(dataset)
    except Exception as exc:
        raise RuntimeError(
            f"MOABB rejected {spec['class']} for declared paradigm {spec['paradigm']}"
        ) from exc
    if valid is False:
        raise RuntimeError(
            f"MOABB rejected {spec['class']} for declared paradigm {spec['paradigm']}"
        )
    return dataset, paradigm


def _build_csp_lda(*, components: int):
    try:
        from mne.decoding import CSP
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.pipeline import make_pipeline
    except ImportError as exc:  # pragma: no cover - real environment boundary
        raise RuntimeError("CSP+LDA baseline requires MNE and scikit-learn") from exc
    return make_pipeline(CSP(n_components=int(components), reg=None), LDA())


def _score_binary(model: Any, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score

    started = time.perf_counter()
    prediction = model.predict(X)
    inference_s = time.perf_counter() - started
    metrics = {
        "accuracy": float(accuracy_score(y, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "inference_s": float(inference_s),
        "inference_ms_per_trial": float(1000.0 * inference_s / max(len(y), 1)),
    }
    classes = np.asarray(getattr(model, "classes_", []))
    if len(classes) == 2 and hasattr(model, "predict_proba"):
        probability = np.asarray(model.predict_proba(X), dtype=np.float64)
        positive = classes[1]
        y_binary = (np.asarray(y) == positive).astype(np.int64)
        metrics["roc_auc"] = float(roc_auc_score(y_binary, probability[:, 1]))
    else:
        metrics["roc_auc"] = float("nan")
    return metrics


def _mean(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if len(array) == 0 else float(np.mean(array))


def _std(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return None if len(array) == 0 else float(np.std(array, ddof=1)) if len(array) > 1 else 0.0


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_budget: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_budget[int(row["calibration_per_class"])].append(row)

    curve = []
    for budget in sorted(by_budget):
        group = by_budget[budget]
        curve.append(
            {
                "calibration_per_class": budget,
                "n_subject_session_cases": len(group),
                "mean_balanced_accuracy": _mean(row["balanced_accuracy"] for row in group),
                "std_balanced_accuracy": _std(row["balanced_accuracy"] for row in group),
                "mean_roc_auc": _mean(row["roc_auc"] for row in group),
                "mean_fit_s": _mean(row["fit_s"] for row in group),
                "mean_inference_ms_per_trial": _mean(
                    row["inference_ms_per_trial"] for row in group
                ),
            }
        )

    improvement = None
    if len(curve) >= 2 and curve[0]["calibration_per_class"] == 0:
        start = curve[0]["mean_balanced_accuracy"]
        end = curve[-1]["mean_balanced_accuracy"]
        if start is not None and end is not None:
            improvement = float(end - start)

    return {
        "schema_version": 1,
        "primary_metric": "balanced_accuracy",
        "curve": curve,
        "largest_budget_minus_zero_balanced_accuracy": improvement,
        "interpretation": (
            "Descriptive baseline summary only. Statistical comparison and model "
            "promotion require repeated subjects/sessions and a predeclared analysis plan."
        ),
    }


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write empty benchmark results")
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _render_report(
    *,
    source: Any,
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    split_records: list[dict[str, Any]],
) -> str:
    history_text = (
        "prior sessions only; future sessions excluded"
        if args.history_policy == "prior"
        else "all other sessions; symmetric cross-session evaluation"
    )
    lines = [
        "# neurOS Longitudinal EEG Evidence Report",
        "",
        f"- Dataset: **{source.title}** (`{source.id}`)",
        f"- Baseline: **CSP({args.csp_components}) + LDA**",
        f"- Band: **{args.fmin:g}-{args.fmax:g} Hz**",
        f"- History policy: **{history_text}**",
        f"- Fixed held-out evaluation fraction: **{args.evaluation_fraction:.2f}**",
        f"- Subjects requested: **{', '.join(str(v) for v in args.subjects)}**",
        f"- Subject-session cases: **{len(split_records)}**",
        "",
        "## Calibration frontier",
        "",
        "| calibration / class | cases | balanced accuracy mean | std | ROC-AUC mean | fit s | inference ms/trial |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for point in summary["curve"]:
        def fmt(value: Any, digits: int = 4) -> str:
            return "n/a" if value is None else f"{float(value):.{digits}f}"

        lines.append(
            "| {calibration_per_class} | {n_subject_session_cases} | {ba} | {std} | "
            "{auc} | {fit} | {infer} |".format(
                **point,
                ba=fmt(point["mean_balanced_accuracy"]),
                std=fmt(point["std_balanced_accuracy"]),
                auc=fmt(point["mean_roc_auc"]),
                fit=fmt(point["mean_fit_s"], 3),
                infer=fmt(point["mean_inference_ms_per_trial"], 3),
            )
        )
    lines.extend(
        [
            "",
            "## Evidence contract",
            "",
            "Every subject-session case freezes the held-out session before model fitting. "
            "Within that session, the evaluation examples are frozen once and never change "
            "as calibration budget grows. Calibration subsets are nested per class.",
            "",
        ]
    )
    if args.history_policy == "prior":
        lines.extend(
            [
                "The default chronological policy trains only on sessions preceding the "
                "held-out session in upstream metadata order. Later sessions are excluded "
                "from both fitting and evaluation, preventing future-data leakage.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "The `all-other` policy is symmetric leave-one-session-out evaluation and "
                "may include sessions recorded after the held-out session. It must not be "
                "described as next-session or prospective deployment evidence.",
                "",
            ]
        )
    lines.extend(
        [
            "The zero-calibration row uses source-history data only; larger budgets add "
            "held-out-session calibration examples but are scored on the same evaluation trials.",
            "",
            "## Limits",
            "",
            "- This report is a transparent baseline, not an ORION or neurOS superiority claim.",
            "- Dataset files/checksums remain governed by MOABB/upstream repositories; pin those identities for promoted evidence.",
            "- CSP+LDA is refit when calibration examples are added; it is not an online-update algorithm.",
            "- Upstream metadata order is treated as chronology under `prior`; promoted studies must verify that assumption against dataset documentation.",
            "- Hardware/closed-loop/clinical evidence is outside this offline real-dataset tier.",
            "",
            f"Raw result rows: {len(rows)}. See `results.csv`, `study_manifest.json`, and `artifact_hashes.json`.",
            "",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a fixed-evaluation longitudinal MOABB calibration benchmark."
    )
    parser.add_argument("--dataset", choices=sorted(_DATASETS), default="kumar2024")
    parser.add_argument(
        "--subjects",
        type=_parse_int_list,
        default=[1],
        help="Comma-separated MOABB subject IDs (default: 1).",
    )
    parser.add_argument(
        "--held-out-sessions",
        type=_parse_text_list,
        default=None,
        help="Optional comma-separated session IDs. Default: evaluate every eligible session.",
    )
    parser.add_argument(
        "--history-policy",
        choices=("prior", "all-other"),
        default="prior",
        help=(
            "Source-session policy. 'prior' (default) trains only on earlier sessions; "
            "'all-other' is symmetric leave-one-session-out and may use future sessions."
        ),
    )
    parser.add_argument(
        "--budgets",
        type=_parse_int_list,
        default=[0, 1, 2, 5, 10],
        help="Nested labeled calibration trials per class.",
    )
    parser.add_argument("--evaluation-fraction", type=float, default=0.5)
    parser.add_argument("--fmin", type=float, default=8.0)
    parser.add_argument("--fmax", type=float, default=30.0)
    parser.add_argument("--csp-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fmin <= 0 or args.fmax <= args.fmin:
        raise SystemExit("require 0 < --fmin < --fmax")
    if args.csp_components <= 0:
        raise SystemExit("--csp-components must be positive")
    if not 0.0 < args.evaluation_fraction < 1.0:
        raise SystemExit("--evaluation-fraction must lie strictly between 0 and 1")
    if any(budget < 0 for budget in args.budgets):
        raise SystemExit("--budgets must be non-negative")
    args.budgets = sorted(set(args.budgets))
    if 0 not in args.budgets:
        args.budgets.insert(0, 0)

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    controlled = [
        output / "study_manifest.json",
        output / "results.csv",
        output / "summary.json",
        output / "report.md",
        output / "artifact_hashes.json",
    ]
    existing = [path for path in controlled if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(path.name for path in existing)
        raise SystemExit(f"refusing to overwrite existing artifacts: {names}")

    dataset, paradigm = _dataset_and_paradigm(
        args.dataset,
        fmin=args.fmin,
        fmax=args.fmax,
    )
    dataset_spec = _DATASETS[args.dataset]
    source_id = dataset_spec["source_id"]
    source = get_evidence_source(source_id)

    rows: list[dict[str, Any]] = []
    split_records: list[dict[str, Any]] = []
    skipped_budgets: list[dict[str, Any]] = []

    for subject in args.subjects:
        bundle = collect_moabb(
            dataset,
            paradigm,
            subjects=[int(subject)],
            dataset_id=source_id,
        )
        sessions = ordered_group_values(bundle, split_unit="session")
        if len(sessions) < 2:
            raise RuntimeError(
                f"subject {subject} has fewer than two sessions after MOABB preprocessing"
            )

        if args.held_out_sessions is not None:
            missing = [value for value in args.held_out_sessions if value not in sessions]
            if missing:
                raise RuntimeError(
                    f"subject {subject} does not expose requested session(s) {missing}; "
                    f"available={list(sessions)}"
                )
            selected_sessions = list(args.held_out_sessions)
        elif args.history_policy == "prior":
            selected_sessions = list(sessions[1:])
        else:
            selected_sessions = list(sessions)

        for session in selected_sessions:
            if args.history_policy == "prior":
                try:
                    partition = chronological_partition(
                        bundle,
                        split_unit="session",
                        held_out_value=session,
                        order=sessions,
                    )
                except ValueError as exc:
                    raise RuntimeError(
                        f"cannot construct prior-only evidence for subject {subject}, "
                        f"session {session}: {exc}"
                    ) from exc
            else:
                partition = hold_out_groups(
                    bundle,
                    split_unit="session",
                    held_out_values=[session],
                )

            split_seed = _stable_int_seed(args.seed, source_id, subject, session)
            calibration = make_nested_calibration_split(
                partition,
                evaluation_fraction=args.evaluation_fraction,
                seed=split_seed,
            )
            events_text = (
                "left_hand/right_hand"
                if dataset_spec["events"] is None
                else "/".join(dataset_spec["events"])
            )
            protocol = partition.protocol(
                name=f"{source_id}-subject-{subject}-session-{session}",
                transfer_regime="few_shot",
                preprocessing=(
                    f"MOABB {dataset_spec['paradigm']} events={events_text}; "
                    f"bandpass {args.fmin:g}-{args.fmax:g} Hz; CSP+LDA fit only on "
                    "declared source sessions plus held-out calibration examples"
                ),
                notes=(
                    f"history_policy={args.history_policy}",
                    "fixed evaluation subset inside held-out session",
                    "nested balanced calibration budgets per class",
                ),
                seed=split_seed,
            )
            record = partition.manifest(protocol=protocol)
            source_sessions = tuple(
                dict.fromkeys(
                    np.asarray(bundle.groups["session"])[partition.train_indices]
                    .astype(str)
                    .tolist()
                )
            )
            record["subject"] = int(subject)
            record["held_out_session"] = str(session)
            record["history_policy"] = args.history_policy
            record["observed_session_order"] = list(sessions)
            record["source_sessions"] = list(source_sessions)
            record["calibration"] = calibration.manifest()
            split_records.append(record)

            X = np.asarray(bundle.X)
            y = np.asarray(bundle.y)
            eval_idx = calibration.evaluation_indices

            for budget in args.budgets:
                if budget > calibration.max_budget_per_class:
                    skipped_budgets.append(
                        {
                            "subject": int(subject),
                            "held_out_session": str(session),
                            "requested_per_class": int(budget),
                            "max_balanced_per_class": int(calibration.max_budget_per_class),
                        }
                    )
                    continue

                train_idx = calibration.train_indices_for_budget(budget)
                cal_idx = calibration.calibration_indices(budget)
                model = _build_csp_lda(components=args.csp_components)
                started = time.perf_counter()
                model.fit(X[train_idx], y[train_idx])
                fit_s = time.perf_counter() - started
                metrics = _score_binary(model, X[eval_idx], y[eval_idx])

                rows.append(
                    {
                        "dataset_id": source_id,
                        "subject": int(subject),
                        "held_out_session": str(session),
                        "history_policy": args.history_policy,
                        "partition_fingerprint": partition.fingerprint,
                        "calibration_split_fingerprint": calibration.fingerprint,
                        "calibration_per_class": int(budget),
                        "source_train_samples": int(len(calibration.source_train_indices)),
                        "calibration_samples": int(len(cal_idx)),
                        "evaluation_samples": int(len(eval_idx)),
                        "total_fit_samples": int(len(train_idx)),
                        "accuracy": metrics["accuracy"],
                        "balanced_accuracy": metrics["balanced_accuracy"],
                        "roc_auc": metrics["roc_auc"],
                        "fit_s": float(fit_s),
                        "inference_s": metrics["inference_s"],
                        "inference_ms_per_trial": metrics["inference_ms_per_trial"],
                    }
                )

    if not rows:
        raise RuntimeError("benchmark produced no evaluable rows")

    summary = _summarize(rows)
    manifest = {
        "schema_version": 1,
        "evidence_tier": "real_dataset",
        "study": "longitudinal_eeg_calibration_baseline",
        "method": "CSP+LDA refit with nested held-out-session calibration",
        "source": source.to_dict(),
        "dataset_key": args.dataset,
        "dataset_class": dataset_spec["class"],
        "paradigm": {
            "kind": dataset_spec["paradigm"],
            "events": list(dataset_spec["events"] or ("left_hand", "right_hand")),
        },
        "history_policy": args.history_policy,
        "subjects": [int(value) for value in args.subjects],
        "held_out_sessions_requested": args.held_out_sessions,
        "requested_calibration_per_class": [int(value) for value in args.budgets],
        "evaluation_fraction": float(args.evaluation_fraction),
        "band_hz": [float(args.fmin), float(args.fmax)],
        "csp_components": int(args.csp_components),
        "seed": int(args.seed),
        "git_revision": _git_revision(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": _versions(),
        "splits": split_records,
        "skipped_budgets": skipped_budgets,
        "limitations": [
            "upstream MOABB/data-file checksums are not yet captured by this runner",
            "CSP+LDA is a transparent baseline, not an ORION superiority claim",
            "prior policy assumes first-observed upstream session metadata order is chronological and must be verified before promotion",
            "all-other policy is symmetric cross-session evaluation and is not prospective next-session evidence",
            "offline real-dataset evidence is not hardware or closed-loop qualification",
        ],
    }

    results_path = output / "results.csv"
    summary_path = output / "summary.json"
    manifest_path = output / "study_manifest.json"
    report_path = output / "report.md"
    hashes_path = output / "artifact_hashes.json"

    _write_results_csv(results_path, rows)
    _json_dump(summary_path, summary)
    _json_dump(manifest_path, manifest)
    report_path.write_text(
        _render_report(
            source=source,
            args=args,
            rows=rows,
            summary=summary,
            split_records=split_records,
        ),
        encoding="utf-8",
    )

    hashes = {
        path.name: _sha256(path)
        for path in (results_path, summary_path, manifest_path, report_path)
    }
    _json_dump(hashes_path, {"sha256": hashes})

    print(
        json.dumps(
            {
                "output": str(output),
                "dataset": source_id,
                "history_policy": args.history_policy,
                "rows": len(rows),
                "subject_session_cases": len(split_records),
                "artifacts": [path.name for path in controlled],
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
