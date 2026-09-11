#!/usr/bin/env python3
"""Score a completed raw NIM swarm smoke in a separate non-network process."""
from __future__ import annotations

import argparse
import json
import math
from hashlib import sha256
from pathlib import Path
from typing import Any

from neuros.research._canonical import canonical_sha256, require_sha256
from neuros.research.nim_provider import DOCUMENTED_NVIDIA_CHAT_MODELS
from neuros.research.nim_swarm import _user_prompt
from neuros.research.swarm import CouncilMember
from neuros.research.swarm_benchmark_cases import (
    BENCHMARK_CORPUS_SHA256,
    build_benchmark_task,
)
from neuros.research.swarm_live_eval import (
    SCHEDULE_POLICY,
    SMOKE_CASE_IDS,
    EvaluationConfiguration,
    build_counterbalanced_schedule,
    build_evaluation_configurations,
    council_run_from_dict,
)

_LIVE_MAX_TOKENS = 1200
_LIVE_TEMPERATURE = 0.1
_PROVIDER_FIELDS = {
    "schema_version",
    "endpoint",
    "discovery_mode",
    "documented_candidates",
    "catalog_models_sha256",
    "catalog_error",
    "probes",
    "qualified_models",
    "discovery_budget",
    "authority_boundary",
    "fingerprint",
}
_PROBE_FIELDS = {"model", "status", "status_code", "response_sha256", "error_excerpt"}
_RECEIPT_FIELDS = {
    "schema",
    "case_id",
    "task_sha256",
    "member_id",
    "role",
    "model",
    "member_prompt_sha256",
    "provider_qualification_fingerprint",
    "endpoint",
    "outcome",
    "latency_ms",
    "call_prompt_sha256",
    "request_sha256",
    "response_sha256",
    "parsed_response_sha256",
    "token_usage",
    "error_class",
    "token_counts_are_provider_reported_not_estimated",
    "receipt_is_scientific_authority",
    "receipt_sha256",
}
_MANIFEST_FIELDS = {
    "schema",
    "repository",
    "source_revision",
    "corpus_sha256",
    "configuration",
    "configuration_sha256",
    "provider_qualification_fingerprint",
    "case_ids",
    "runs",
    "receipts",
    "case_wall_latency_ms",
    "token_counts_are_provider_reported_not_estimated",
    "dollar_cost_estimated",
    "live_review_output_is_scientific_authority",
    "merge_authority",
    "provider_execution_authority",
    "scientific_promotion_authority",
    "manifest_sha256",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _require_false(payload: dict[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise ValueError(f"{key} must remain false")


def _nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_nonnegative_int(value: Any, *, name: str) -> int | None:
    if value is None:
        return None
    return _nonnegative_int(value, name=name)


def _validated_provider_qualification(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict) or set(payload) != _PROVIDER_FIELDS:
        raise ValueError("provider qualification fields do not match the frozen schema")
    fingerprint = require_sha256(payload.get("fingerprint", ""), name="provider_fingerprint")
    unhashed = dict(payload)
    del unhashed["fingerprint"]
    if canonical_sha256(unhashed) != fingerprint:
        raise ValueError("provider qualification fingerprint mismatch")
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported provider qualification schema")
    if payload.get("endpoint") != "https://integrate.api.nvidia.com/v1":
        raise ValueError("provider qualification endpoint is not the pinned NVIDIA endpoint")
    if payload.get("documented_candidates") != list(DOCUMENTED_NVIDIA_CHAT_MODELS):
        raise ValueError("provider qualification candidate roster differs from frozen roster")

    probes = payload.get("probes")
    if not isinstance(probes, list) or len(probes) != len(DOCUMENTED_NVIDIA_CHAT_MODELS):
        raise ValueError("provider qualification must contain one probe per documented candidate")
    derived_qualified: list[str] = []
    for expected_model, probe in zip(DOCUMENTED_NVIDIA_CHAT_MODELS, probes, strict=True):
        if not isinstance(probe, dict) or set(probe) != _PROBE_FIELDS:
            raise ValueError("provider probe fields do not match the frozen schema")
        if probe.get("model") != expected_model:
            raise ValueError("provider probe order/model identity differs from frozen roster")
        status = probe.get("status")
        if status == "qualified":
            require_sha256(probe.get("response_sha256", ""), name="probe_response_sha256")
            if probe.get("status_code") is not None or probe.get("error_excerpt") is not None:
                raise ValueError("qualified provider probe cannot carry failure evidence")
            derived_qualified.append(expected_model)
        elif status in {"http_error", "transport_error", "invalid_response"}:
            error = probe.get("error_excerpt")
            if not isinstance(error, str) or not error.strip():
                raise ValueError("failed provider probe requires bounded failure evidence")
            if probe.get("response_sha256") is not None:
                raise ValueError("failed provider probe cannot claim successful response identity")
        else:
            raise ValueError("unsupported provider probe status")
    if payload.get("qualified_models") != derived_qualified or not derived_qualified:
        raise ValueError("qualified_models must be derived exactly from successful probes")

    budget = payload.get("discovery_budget")
    expected_budget_fields = {"timeout_seconds_per_attempt", "max_attempts_per_route"}
    if not isinstance(budget, dict) or set(budget) != expected_budget_fields:
        raise ValueError("provider discovery budget fields do not match the frozen schema")
    timeout = budget["timeout_seconds_per_attempt"]
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("provider discovery timeout must be positive")
    attempts = budget["max_attempts_per_route"]
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts < 1:
        raise ValueError("provider discovery attempt budget must be positive")
    return payload


def _validated_raw(path: Path) -> tuple[dict[str, Any], tuple[EvaluationConfiguration, ...]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("kind") != "neuros_nim_swarm_live_eval_raw_v1":
        raise ValueError("unexpected raw live-evaluation artifact")
    claimed = require_sha256(payload.get("artifact_sha256", ""), name="artifact_sha256")
    unhashed = dict(payload)
    del unhashed["artifact_sha256"]
    if canonical_sha256(unhashed) != claimed:
        raise ValueError("raw live-evaluation artifact fingerprint mismatch")
    if payload.get("corpus_sha256") != BENCHMARK_CORPUS_SHA256:
        raise ValueError("raw artifact benchmark corpus mismatch")
    if tuple(payload.get("case_ids", ())) != SMOKE_CASE_IDS:
        raise ValueError("raw artifact does not contain the frozen smoke case slice")
    require_sha256(payload.get("source_revision", ""), name="source_revision")
    for key in (
        "smoke_is_full_benchmark",
        "smoke_is_model_performance_claim",
        "live_review_output_is_scientific_authority",
        "merge_authority",
        "provider_execution_authority",
        "scientific_promotion_authority",
    ):
        _require_false(payload, key)

    qualification = _validated_provider_qualification(payload.get("provider_qualification"))
    if payload.get("discovery_mode") != qualification.get("discovery_mode"):
        raise ValueError("raw discovery mode differs from provider qualification")
    selected = payload.get("selected_models")
    if not isinstance(selected, list) or not selected:
        raise ValueError("raw artifact requires selected_models")
    selected_models = tuple(str(model).strip() for model in selected)
    if any(not model for model in selected_models) or len(set(selected_models)) != len(selected_models):
        raise ValueError("selected model identities must be non-empty and unique")
    qualified_models = tuple(str(model) for model in qualification["qualified_models"])
    if selected_models != qualified_models[:3]:
        raise ValueError("selected models differ from frozen qualified-model preference order")

    expected_configurations = build_evaluation_configurations(selected_models)
    if payload.get("schedule_policy") != SCHEDULE_POLICY:
        raise ValueError("raw artifact schedule policy mismatch")
    expected_schedule = [
        {"case_id": case_id, "configuration_id": configuration_id}
        for case_id, configuration_id in build_counterbalanced_schedule(
            expected_configurations,
            SMOKE_CASE_IDS,
        )
    ]
    if payload.get("execution_schedule") != expected_schedule:
        raise ValueError("raw artifact execution schedule mismatch")
    configurations = payload.get("configurations")
    if not isinstance(configurations, list) or len(configurations) != len(expected_configurations):
        raise ValueError("raw artifact configuration count differs from frozen plan")
    expected_calls = sum(
        len(configuration.members) * len(SMOKE_CASE_IDS) for configuration in expected_configurations
    )
    if _nonnegative_int(payload.get("review_call_count"), name="review_call_count") != expected_calls:
        raise ValueError("raw artifact reviewer-call count differs from frozen plan")
    if expected_calls > 33:
        raise ValueError("raw artifact exceeds bounded v1 reviewer-call budget")
    return payload, expected_configurations


def _validated_token_usage(payload: Any) -> dict[str, Any]:
    expected = {"source", "prompt_tokens", "completion_tokens", "total_tokens", "usage_sha256"}
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("token usage fields do not match the v1 schema")
    prompt = _optional_nonnegative_int(payload.get("prompt_tokens"), name="prompt_tokens")
    completion = _optional_nonnegative_int(payload.get("completion_tokens"), name="completion_tokens")
    total = _optional_nonnegative_int(payload.get("total_tokens"), name="total_tokens")
    usage_sha = payload.get("usage_sha256")
    if payload.get("source") == "unavailable":
        if any(value is not None for value in (prompt, completion, total, usage_sha)):
            raise ValueError("unavailable token usage cannot contain provider-reported values")
    elif payload.get("source") == "provider_response":
        require_sha256(usage_sha or "", name="usage_sha256")
    else:
        raise ValueError("unsupported token usage source")
    return payload


def _expected_call_identity(*, task: Any, member: CouncilMember) -> tuple[str, str]:
    user_prompt = _user_prompt(task)
    combined_prompt = f"{member.system_prompt}\n\n{user_prompt}"
    prompt_sha256 = sha256(combined_prompt.encode("utf-8")).hexdigest()
    request_payload = {
        "model": member.model,
        "messages": [
            {"role": "system", "content": member.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": float(_LIVE_TEMPERATURE),
        "max_tokens": int(_LIVE_MAX_TOKENS),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    return prompt_sha256, canonical_sha256(request_payload)


def _validated_receipt(
    receipt: Any,
    *,
    raw: dict[str, Any],
    expected_member: CouncilMember,
    case_id: str,
) -> dict[str, Any]:
    if not isinstance(receipt, dict) or set(receipt) != _RECEIPT_FIELDS:
        raise ValueError("live-evaluation receipt fields do not match the v1 schema")
    receipt_claim = require_sha256(receipt.get("receipt_sha256", ""), name="receipt_sha256")
    receipt_unhashed = dict(receipt)
    del receipt_unhashed["receipt_sha256"]
    if canonical_sha256(receipt_unhashed) != receipt_claim:
        raise ValueError("live-evaluation receipt fingerprint mismatch")
    if receipt.get("schema") != "neuros.nim_swarm_evaluation_call_receipt.v1":
        raise ValueError("unexpected live-evaluation receipt schema")
    if receipt.get("case_id") != case_id:
        raise ValueError("receipt case identity mismatch")
    task = build_benchmark_task(
        case_id,
        repository=raw["repository"],
        source_revision=raw["source_revision"],
    )
    if receipt.get("task_sha256") != task.sha256:
        raise ValueError("receipt task identity mismatch")
    qualification = raw["provider_qualification"]
    if receipt.get("provider_qualification_fingerprint") != qualification["fingerprint"]:
        raise ValueError("receipt provider qualification mismatch")
    if receipt.get("endpoint") != qualification["endpoint"]:
        raise ValueError("receipt endpoint differs from provider qualification")
    if (
        receipt.get("member_id") != expected_member.member_id
        or receipt.get("role") != expected_member.role
        or receipt.get("model") != expected_member.model
        or receipt.get("member_prompt_sha256") != expected_member.prompt_sha256
    ):
        raise ValueError("receipt member identity differs from frozen configuration")
    if receipt.get("token_counts_are_provider_reported_not_estimated") is not True:
        raise ValueError("receipt token-count provenance flag must remain true")
    _require_false(receipt, "receipt_is_scientific_authority")
    _nonnegative_int(receipt.get("latency_ms"), name="latency_ms")
    usage = _validated_token_usage(receipt.get("token_usage"))

    outcome = receipt.get("outcome")
    names = ("call_prompt_sha256", "request_sha256", "response_sha256")
    values = tuple(receipt.get(name) for name in names)
    parsed_sha = receipt.get("parsed_response_sha256")
    error_class = receipt.get("error_class")
    has_transport = outcome in {"success", "review_validation_failure"}
    if has_transport:
        for name, value in zip(names, values, strict=True):
            require_sha256(value or "", name=name)
        expected_prompt, expected_request = _expected_call_identity(task=task, member=expected_member)
        if receipt.get("call_prompt_sha256") != expected_prompt:
            raise ValueError("receipt call-prompt identity differs from frozen request contract")
        if receipt.get("request_sha256") != expected_request:
            raise ValueError("receipt request identity differs from frozen request contract")

    if outcome == "success":
        require_sha256(parsed_sha or "", name="parsed_response_sha256")
        if error_class is not None:
            raise ValueError("successful receipt cannot carry an error class")
    elif outcome == "provider_failure":
        if any(value is not None for value in (*values, parsed_sha)):
            raise ValueError("provider failure cannot claim unavailable transport identities")
        if usage.get("source") != "unavailable":
            raise ValueError("provider failure cannot claim provider token usage")
        if not isinstance(error_class, str) or not error_class.strip():
            raise ValueError("provider failure requires a non-empty error class")
    elif outcome == "review_validation_failure":
        if parsed_sha is not None:
            require_sha256(parsed_sha, name="parsed_response_sha256")
        if not isinstance(error_class, str) or not error_class.strip():
            raise ValueError("review-validation failure requires a non-empty error class")
    else:
        raise ValueError("unsupported receipt outcome")
    return receipt


def _validated_manifest(
    manifest: dict[str, Any],
    *,
    raw: dict[str, Any],
    expected_configuration: EvaluationConfiguration,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, int]]:
    if set(manifest) != _MANIFEST_FIELDS:
        raise ValueError("live-evaluation manifest fields do not match the v1 schema")
    claimed = require_sha256(manifest.get("manifest_sha256", ""), name="manifest_sha256")
    unhashed = dict(manifest)
    del unhashed["manifest_sha256"]
    if canonical_sha256(unhashed) != claimed:
        raise ValueError("live-evaluation manifest fingerprint mismatch")
    if manifest.get("schema") != "neuros.nim_swarm_live_evaluation_manifest.v1":
        raise ValueError("unexpected live-evaluation manifest schema")
    if manifest.get("repository") != raw["repository"]:
        raise ValueError("live-evaluation manifest repository mismatch")
    if manifest.get("source_revision") != raw["source_revision"]:
        raise ValueError("live-evaluation manifest source revision mismatch")
    if manifest.get("corpus_sha256") != BENCHMARK_CORPUS_SHA256:
        raise ValueError("live-evaluation manifest corpus mismatch")
    if tuple(manifest.get("case_ids", ())) != SMOKE_CASE_IDS:
        raise ValueError("live-evaluation manifest case slice mismatch")
    if manifest.get("token_counts_are_provider_reported_not_estimated") is not True:
        raise ValueError("manifest token-count provenance flag must remain true")
    _require_false(manifest, "dollar_cost_estimated")
    for key in (
        "live_review_output_is_scientific_authority",
        "merge_authority",
        "provider_execution_authority",
        "scientific_promotion_authority",
    ):
        _require_false(manifest, key)

    expected_configuration_dict = expected_configuration.to_dict()
    if manifest.get("configuration") != expected_configuration_dict:
        raise ValueError("serialized configuration differs from the frozen reconstructed plan")
    if manifest.get("configuration_sha256") != expected_configuration.sha256:
        raise ValueError("configuration fingerprint mismatch")
    if manifest.get("provider_qualification_fingerprint") != raw["provider_qualification"]["fingerprint"]:
        raise ValueError("manifest provider qualification mismatch")

    timings = manifest.get("case_wall_latency_ms")
    if not isinstance(timings, dict) or set(timings) != set(SMOKE_CASE_IDS):
        raise ValueError("case wall timings do not cover the exact smoke case slice")
    validated_timings = {
        case_id: _nonnegative_int(timings[case_id], name=f"case_wall_latency_ms[{case_id}]")
        for case_id in SMOKE_CASE_IDS
    }

    members = {member.member_id: member for member in expected_configuration.members}
    receipts = manifest.get("receipts")
    if not isinstance(receipts, list):
        raise ValueError("live-evaluation manifest receipts must be a list")
    receipt_map: dict[tuple[str, str], dict[str, Any]] = {}
    for receipt in receipts:
        if not isinstance(receipt, dict):
            raise ValueError("live-evaluation receipt must be a JSON object")
        pair = (receipt.get("case_id"), receipt.get("member_id"))
        if pair in receipt_map:
            raise ValueError("live-evaluation manifest repeats a case/member receipt")
        case_id, member_id = pair
        if case_id not in SMOKE_CASE_IDS or member_id not in members:
            raise ValueError("receipt references case/member outside frozen configuration")
        receipt_map[pair] = _validated_receipt(
            receipt,
            raw=raw,
            expected_member=members[member_id],
            case_id=case_id,
        )
    expected_pairs = {
        (case_id, member_id) for case_id in SMOKE_CASE_IDS for member_id in members
    }
    if set(receipt_map) != expected_pairs:
        raise ValueError("receipts do not cover the exact frozen case/member plan")

    runs_payload = manifest.get("runs")
    if not isinstance(runs_payload, dict) or set(runs_payload) != set(SMOKE_CASE_IDS):
        raise ValueError("configuration does not contain the exact smoke case slice")
    runs = {case_id: council_run_from_dict(runs_payload[case_id]) for case_id in SMOKE_CASE_IDS}
    serialized_members = {
        row["member_id"]: row for row in expected_configuration_dict["members"]
    }
    for case_id, run in runs.items():
        task = build_benchmark_task(
            case_id,
            repository=raw["repository"],
            source_revision=raw["source_revision"],
        )
        if run.task_sha256 != task.sha256:
            raise ValueError("serialized council run task identity mismatch")
        successful = {review.member_id: review for review in run.reviews}
        failed: dict[str, str] = {}
        for entry in run.failed_members:
            member_id, separator, error_class = entry.rpartition(":")
            if not separator or not member_id or not error_class or member_id in failed:
                raise ValueError("serialized failed reviewer identity is malformed or repeated")
            failed[member_id] = error_class
        if set(successful) | set(failed) != set(members) or set(successful) & set(failed):
            raise ValueError("serialized council run does not account for the frozen member plan")
        for member_id, expected_member in serialized_members.items():
            receipt = receipt_map[(case_id, member_id)]
            if member_id in successful:
                review = successful[member_id]
                if (
                    review.role != expected_member["role"]
                    or review.model != expected_member["model"]
                    or review.prompt_sha256 != expected_member["prompt_sha256"]
                ):
                    raise ValueError("serialized review member identity differs from frozen plan")
                if receipt["outcome"] != "success":
                    raise ValueError("successful serialized review lacks successful receipt")
                if receipt["parsed_response_sha256"] != review.response_sha256:
                    raise ValueError("receipt parsed-response identity differs from serialized review")
            else:
                if receipt["outcome"] not in {"provider_failure", "review_validation_failure"}:
                    raise ValueError("failed serialized review lacks failure receipt")
                if receipt["error_class"] != failed[member_id]:
                    raise ValueError("failed serialized review error class differs from receipt")
    return expected_configuration_dict, runs, list(receipt_map.values()), validated_timings


def _p95(values: list[int]) -> int:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _telemetry_summary(receipts: list[dict[str, Any]], timings: dict[str, int]) -> dict[str, Any]:
    latencies = [_nonnegative_int(row["latency_ms"], name="latency_ms") for row in receipts]
    outcome_counts = {
        outcome: sum(row["outcome"] == outcome for row in receipts)
        for outcome in ("success", "provider_failure", "review_validation_failure")
    }
    token_rows = [row["token_usage"] for row in receipts]

    def token_metric(name: str) -> dict[str, Any]:
        known = [row[name] for row in token_rows if row[name] is not None]
        return {
            "reported_receipts": len(known),
            "total_reported": sum(known) if known else None,
            "all_receipts_reported": len(known) == len(token_rows),
        }

    wall = [timings[case_id] for case_id in SMOKE_CASE_IDS]
    return {
        "reviewer_attempts": len(receipts),
        "outcomes": outcome_counts,
        "provider_usage_receipts": sum(
            row["source"] == "provider_response" for row in token_rows
        ),
        "per_call_latency_ms": {
            "total": sum(latencies),
            "mean": (sum(latencies) / len(latencies)) if latencies else None,
            "p95_nearest_rank": _p95(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "case_wall_latency_ms": {
            "total": sum(wall),
            "mean": sum(wall) / len(wall),
            "p95_nearest_rank": _p95(wall),
            "max": max(wall),
            "by_case": {case_id: timings[case_id] for case_id in SMOKE_CASE_IDS},
        },
        "prompt_tokens": token_metric("prompt_tokens"),
        "completion_tokens": token_metric("completion_tokens"),
        "total_tokens": token_metric("total_tokens"),
        "token_counts_are_provider_reported_not_estimated": True,
        "dollar_cost_estimated": False,
    }


def main() -> None:
    args = _args()
    raw, expected_configurations = _validated_raw(args.input)

    validated = []
    for manifest, expected_configuration in zip(
        raw["configurations"],
        expected_configurations,
        strict=True,
    ):
        if not isinstance(manifest, dict):
            raise ValueError("configuration manifest must be a JSON object")
        configuration, runs, receipts, timings = _validated_manifest(
            manifest,
            raw=raw,
            expected_configuration=expected_configuration,
        )
        validated.append(
            (
                manifest,
                expected_configuration,
                configuration,
                runs,
                receipts,
                timings,
            )
        )

    # Ground truth is deliberately imported only after the complete raw artifact and every
    # configuration/receipt/run has passed credential-free structural verification.
    from neuros.research.swarm_benchmark import (  # noqa: PLC0415
        compare_benchmark_reports,
        score_benchmark,
    )

    reports: dict[str, dict[str, Any]] = {}
    report_objects = {}
    for manifest, expected_configuration, configuration, runs, receipts, timings in validated:
        configuration_id = expected_configuration.configuration_id
        report = score_benchmark(
            runs,
            repository=raw["repository"],
            source_revision=raw["source_revision"],
        )
        report_objects[configuration_id] = report
        reports[configuration_id] = {
            "kind": expected_configuration.kind,
            "configuration": configuration,
            "live_manifest_sha256": require_sha256(
                manifest.get("manifest_sha256", ""),
                name="manifest_sha256",
            ),
            "benchmark_report": report.to_dict(),
            "telemetry": _telemetry_summary(receipts, timings),
        }

    baseline_id = "single-reference-v1"
    if baseline_id not in reports:
        raise ValueError("single-reference configuration is required for comparison")
    comparisons = []
    for configuration_id, payload in reports.items():
        if configuration_id == baseline_id:
            continue
        comparisons.append(
            {
                "baseline_configuration_id": baseline_id,
                "candidate_configuration_id": configuration_id,
                "candidate_kind": payload["kind"],
                "comparison": compare_benchmark_reports(
                    report_objects[baseline_id],
                    report_objects[configuration_id],
                ),
            }
        )

    scored = {
        "kind": "neuros_nim_swarm_live_eval_scored_v1",
        "repository": raw["repository"],
        "source_revision": raw["source_revision"],
        "corpus_sha256": BENCHMARK_CORPUS_SHA256,
        "case_ids": list(SMOKE_CASE_IDS),
        "selected_models": raw["selected_models"],
        "reference_model": raw["selected_models"][0],
        "schedule_policy": SCHEDULE_POLICY,
        "raw_artifact_sha256": raw["artifact_sha256"],
        "reports": reports,
        "comparisons": comparisons,
        "public_smoke_is_full_benchmark": False,
        "benchmark_scores_are_scientific_results": False,
        "benchmark_output_is_scientific_authority": False,
        "merge_authority": False,
        "provider_execution_authority": False,
        "scientific_promotion_authority": False,
    }
    scored["artifact_sha256"] = canonical_sha256(scored)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(scored, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
