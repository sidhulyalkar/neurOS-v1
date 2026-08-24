#!/usr/bin/env python3
"""Run a longitudinal EEG model ladder under serialized sample authority."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
    ordered_group_values,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.longitudinal_ladder import (
    LADDER_METHODS,
    LadderRuntimeConfig,
    render_ladder_report,
    run_ladder_method,
    summarize_ladder_rows,
)
from neuros.foundation_models.longitudinal_transfer import PreparedFrozenEncoderCase
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


def _parse_methods(value: str) -> list[str]:
    values = _parse_text_list(value)
    unknown = sorted(set(values) - set(LADDER_METHODS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown methods {unknown}; available={list(LADDER_METHODS)}"
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
    result: dict[str, str | None] = {}
    for name in (
        "neuros-foundation",
        "neuros-core",
        "neuros-models",
        "neuros-sourceweigher",
        "moabb",
        "mne",
        "scikit-learn",
        "torch",
        "numpy",
    ):
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


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty model-ladder result table")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


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


def _failure_row(
    authority: LongitudinalCaseAuthority,
    *,
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
        help=f"Comma-separated subset of: {','.join(LADDER_METHODS)}",
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

    runtime = LadderRuntimeConfig(
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        device=str(args.device),
        csp_components=int(args.csp_components),
        readout_c=float(args.readout_c),
    )
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

            prepared_cache: dict[tuple[str, int], PreparedFrozenEncoderCase] = {}
            for method in methods:
                seeds: tuple[int | None, ...] = (
                    (None,) if method == "csp-lda" else tuple(model_seeds)
                )
                for model_seed in seeds:
                    try:
                        run = run_ladder_method(
                            data,
                            authority,
                            method=method,
                            budgets_per_class=budgets,
                            model_seed=model_seed,
                            config=runtime,
                            prepared_cache=prepared_cache,
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
                                "runtime": runtime.to_dict(),
                            }
                        )
                        for budget in budgets:
                            result_rows.append(
                                _enrich_row(
                                    _failure_row(
                                        authority,
                                        method=method,
                                        seed=model_seed,
                                        budget=budget,
                                        exc=exc,
                                    ),
                                    authority,
                                    split_seed=case_split_seed,
                                )
                            )

    summary = summarize_ladder_rows(
        result_rows,
        budgets=budgets,
        methods=methods,
    )
    manifest = {
        "schema_version": 3,
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
        "runtime_config": runtime.to_dict(),
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
            "frozen-logistic and SourceWeigher sharing encoder/seed reuse exact representation tensors",
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
    _json_dump(paths["methods"], {"schema_version": 3, "runs": method_runs})
    _write_csv(paths["results"], result_rows)
    _json_dump(paths["summary"], summary)
    paths["report"].write_text(
        render_ladder_report(manifest, summary),
        encoding="utf-8",
    )
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