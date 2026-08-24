"""Reusable orchestration and aggregation for longitudinal EEG model ladders."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

from .longitudinal_authority import LongitudinalCaseAuthority
from .longitudinal_baseline import run_csp_case
from .longitudinal_methods import TaskDecoderMethodSpec, run_task_decoder_case
from .longitudinal_transfer import (
    FrozenTransferMethodSpec,
    PreparedFrozenEncoderCase,
    prepare_frozen_encoder_case,
    run_frozen_transfer_case,
)

LADDER_METHODS = (
    "csp-lda",
    "eegnet",
    "eeg-conformer",
    "frozen-eegnet",
    "frozen-eeg-conformer",
    "sourceweigher-eegnet",
    "sourceweigher-eeg-conformer",
)
SOURCEWEIGHER_METHODS = frozenset(
    {"sourceweigher-eegnet", "sourceweigher-eeg-conformer"}
)


@dataclass(frozen=True, slots=True)
class LadderRuntimeConfig:
    epochs: int = 20
    batch_size: int = 32
    device: str = "auto"
    csp_components: int = 8
    readout_c: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if self.csp_components <= 0 or self.readout_c <= 0:
            raise ValueError("csp_components and readout_c must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "device": self.device,
            "csp_components": int(self.csp_components),
            "readout_c": float(self.readout_c),
        }


def _encoder_for_method(method: str) -> str:
    if method.endswith("eegnet"):
        return "eegnet"
    if method.endswith("eeg-conformer"):
        return "eeg-conformer"
    raise ValueError(f"method {method!r} is not a frozen encoder lane")


def _prepared_state(
    data: Any,
    authority: LongitudinalCaseAuthority,
    *,
    encoder_id: str,
    model_seed: int,
    config: LadderRuntimeConfig,
    cache: MutableMapping[tuple[str, int], PreparedFrozenEncoderCase],
) -> PreparedFrozenEncoderCase:
    key = (encoder_id, int(model_seed))
    state = cache.get(key)
    if state is None:
        state = prepare_frozen_encoder_case(
            data,
            authority,
            encoder_id=encoder_id,  # type: ignore[arg-type]
            encoder_seed=int(model_seed),
            encoder_kwargs={
                "n_epochs": int(config.epochs),
                "batch_size": int(config.batch_size),
                "device": config.device,
            },
        )
        cache[key] = state
    return state


def run_ladder_method(
    data: Any,
    authority: LongitudinalCaseAuthority,
    *,
    method: str,
    budgets_per_class: Sequence[int],
    model_seed: int | None,
    config: LadderRuntimeConfig,
    prepared_cache: MutableMapping[
        tuple[str, int], PreparedFrozenEncoderCase
    ] | None = None,
):
    """Run one ladder lane under one frozen case authority.

    Frozen and SourceWeigher lanes sharing an encoder ID/seed reuse the exact
    same ``PreparedFrozenEncoderCase`` when the caller supplies one cache.
    """
    if method not in LADDER_METHODS:
        raise ValueError(f"unsupported ladder method {method!r}")
    budgets = tuple(sorted(set(int(value) for value in budgets_per_class)))
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets_per_class must contain non-negative values")

    if method == "csp-lda":
        if model_seed is not None:
            raise ValueError("csp-lda does not consume a model seed")
        return run_csp_case(
            data,
            authority,
            budgets_per_class=budgets,
            csp_components=config.csp_components,
        )

    if model_seed is None:
        raise ValueError(f"method {method} requires model_seed")
    common_kwargs = {
        "n_epochs": int(config.epochs),
        "batch_size": int(config.batch_size),
        "device": config.device,
    }

    if method in {"eegnet", "eeg-conformer"}:
        return run_task_decoder_case(
            data,
            authority,
            spec=TaskDecoderMethodSpec(
                method_id=method,
                model_seed=int(model_seed),
                model_kwargs=common_kwargs,
            ),
            budgets_per_class=budgets,
        )

    encoder = _encoder_for_method(method)
    cache = prepared_cache if prepared_cache is not None else {}
    prepared = _prepared_state(
        data,
        authority,
        encoder_id=encoder,
        model_seed=int(model_seed),
        config=config,
        cache=cache,
    )
    strategy = (
        "sourceweigher-mean" if method in SOURCEWEIGHER_METHODS else "frozen-logistic"
    )
    return run_frozen_transfer_case(
        data,
        authority,
        spec=FrozenTransferMethodSpec(
            method_id=method,
            strategy=strategy,
            encoder_id=encoder,  # type: ignore[arg-type]
            encoder_seed=int(model_seed),
            encoder_kwargs=common_kwargs,
            readout_c=config.readout_c,
        ),
        budgets_per_class=budgets,
        prepared=prepared,
    )


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


def seed_averaged_case_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Average optimization seeds inside one frozen subject/session case."""
    ok = [
        dict(row)
        for row in rows
        if row.get("status") == "ok" and row.get("balanced_accuracy") is not None
    ]
    groups: dict[tuple[str, int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in ok:
        groups[
            (
                str(row["method_id"]),
                int(row["calibration_per_class"]),
                str(row["case_id"]),
            )
        ].append(row)

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


def group_budget_summary(
    collapsed: Sequence[Mapping[str, Any]],
    *,
    group_field: str | None = None,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
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
                if row.get("roc_auc") is not None
            ),
        }
        if group_field is not None:
            record[group_field] = group_value[0]
        result.append(record)
    return result


def manual_trapezoid(y: np.ndarray, x: np.ndarray) -> float:
    """NumPy >=1.24-compatible trapezoidal integration."""
    if y.ndim != 1 or x.ndim != 1 or len(y) != len(x):
        raise ValueError("trapezoid inputs must be aligned 1-D vectors")
    if len(x) < 2:
        return 0.0
    return float(np.sum((y[:-1] + y[1:]) * 0.5 * np.diff(x)))


def frontier_auc(
    rows: Sequence[Mapping[str, Any]],
    requested_budgets: Sequence[int],
) -> list[dict[str, Any]]:
    """Compute seed-averaged per-case normalized AUC for one budget set."""
    budgets = tuple(int(value) for value in requested_budgets)
    if len(budgets) < 2:
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

    budget_x = np.asarray(budgets, dtype=np.float64)
    span = float(budget_x[-1] - budget_x[0])
    if span <= 0:
        return []

    per_method_case: dict[tuple[str, str], list[float]] = defaultdict(list)
    for (method, case_id, _seed), curve in by_method_case_seed.items():
        if any(budget not in curve for budget in budgets):
            continue
        y = np.asarray([curve[budget] for budget in budgets], dtype=np.float64)
        per_method_case[(method, case_id)].append(manual_trapezoid(y, budget_x) / span)

    by_method: dict[str, list[float]] = defaultdict(list)
    for (method, _case), seed_values in per_method_case.items():
        by_method[method].append(float(np.mean(seed_values)))

    return [
        {
            "method_id": method,
            "complete_frontier_cases": len(values),
            "mean_seed_averaged_frontier_auc": _mean(values),
            "std_seed_averaged_frontier_auc": _std(values),
            "requested_budgets": list(budgets),
        }
        for method, values in sorted(by_method.items())
    ]


def paired_case_set_audit(
    collapsed: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    budgets: Sequence[int],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for method in methods:
        supported = tuple(
            int(budget)
            for budget in budgets
            if not (method in SOURCEWEIGHER_METHODS and int(budget) == 0)
        )
        fingerprints: list[str] = []
        counts: list[int] = []
        for budget in supported:
            case_ids = [
                str(row["case_id"])
                for row in collapsed
                if row["method_id"] == method
                and int(row["calibration_per_class"]) == budget
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


def summarize_ladder_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    budgets: Sequence[int],
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
            row.get("method_id") in SOURCEWEIGHER_METHODS
            and int(row.get("calibration_per_class", -1)) == 0
        )
    ]
    collapsed = seed_averaged_case_rows(rows)
    paired_audit = paired_case_set_audit(collapsed, methods, budgets)
    positive_budgets = tuple(int(budget) for budget in budgets if int(budget) > 0)
    promotion_ready = (
        len(failures) == 0
        and len(unexpected_unavailable) == 0
        and all(item["paired_across_supported_budgets"] for item in paired_audit)
    )
    return {
        "schema_version": 2,
        "primary_metric": "balanced_accuracy",
        "budget_summary_seed_averaged_within_case": group_budget_summary(collapsed),
        "cohort_budget_summary": group_budget_summary(
            collapsed, group_field="original_protocol"
        ),
        "target_session_budget_summary": group_budget_summary(
            collapsed, group_field="held_out_session"
        ),
        "complete_frontier_auc": frontier_auc(rows, budgets),
        "positive_budget_adaptation_auc": frontier_auc(rows, positive_budgets),
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


def render_ladder_report(manifest: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    def fmt(value: Any) -> str:
        return "n/a" if value is None else f"{float(value):.4f}"

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
            "Methods not defined at zero target calibration do not receive a fabricated full-frontier AUC.",
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
            "Secondary comparison over strictly positive labeled-calibration budgets.",
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
            "Every method row references serialized sample authority and processed-data identity.",
            "SourceWeigher at zero target calibration is explicitly unavailable rather than transductive.",
            "",
            f"Failed rows: **{summary['failed_rows']}**. Unavailable rows: **{summary['unavailable_rows']}**. "
            f"Unexpected unavailable rows: **{summary['unexpected_unavailable_rows']}**.",
            "",
            "Kumar GR/PAR cohort and target-session summaries are emitted from this same bundle.",
            "",
            "This is descriptive offline real-dataset evidence, not hardware, closed-loop, clinical, or ORION superiority evidence.",
            "",
        ]
    )
    return "\n".join(lines)
