#!/usr/bin/env python3
"""Run the bounded model-facing NVIDIA swarm telemetry smoke.

This process intentionally never imports scorer-side benchmark ground truth.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from neuros.research._canonical import canonical_sha256, require_sha256
from neuros.research.nim import DEFAULT_NVIDIA_ENDPOINT
from neuros.research.nim_observed import ObservedQualifiedNvidiaNimClient
from neuros.research.swarm import run_council
from neuros.research.swarm_benchmark_cases import (
    BENCHMARK_CORPUS_SHA256,
    build_benchmark_task,
)
from neuros.research.swarm_live_eval import (
    SCHEDULE_POLICY,
    SMOKE_CASE_IDS,
    EvaluationRunManifest,
    ObservedNvidiaCouncilTransport,
    build_counterbalanced_schedule,
    build_evaluation_configurations,
)

REPOSITORY = "sidhulyalkar/neurOS-v1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("NVIDIA_NIM_ENDPOINT", DEFAULT_NVIDIA_ENDPOINT),
    )
    return parser.parse_args()


def _elapsed_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns + 999_999) // 1_000_000)


async def _run_all_configurations(
    *,
    client: ObservedQualifiedNvidiaNimClient,
    configurations,  # type: ignore[no-untyped-def]
    source_revision: str,
    provider_fingerprint: str,
):  # type: ignore[no-untyped-def]
    transports = {
        configuration.configuration_id: ObservedNvidiaCouncilTransport(
            client,
            provider_qualification_fingerprint=provider_fingerprint,
            max_tokens=1200,
        )
        for configuration in configurations
    }
    runs = {configuration.configuration_id: [] for configuration in configurations}
    timings = {configuration.configuration_id: [] for configuration in configurations}
    by_id = {configuration.configuration_id: configuration for configuration in configurations}
    schedule = build_counterbalanced_schedule(configurations, SMOKE_CASE_IDS)

    for case_id, configuration_id in schedule:
        configuration = by_id[configuration_id]
        task = build_benchmark_task(
            case_id,
            repository=REPOSITORY,
            source_revision=source_revision,
        )
        started_ns = time.perf_counter_ns()
        run = await run_council(
            task,
            configuration.members,
            transports[configuration_id],
            require_all=False,
        )
        runs[configuration_id].append((case_id, run))
        timings[configuration_id].append((case_id, _elapsed_ms(started_ns)))

    manifests = []
    for configuration in configurations:
        configuration_id = configuration.configuration_id
        manifests.append(
            EvaluationRunManifest(
                repository=REPOSITORY,
                source_revision=source_revision,
                configuration=configuration,
                provider_qualification_fingerprint=provider_fingerprint,
                case_ids=SMOKE_CASE_IDS,
                runs=tuple(runs[configuration_id]),
                receipts=transports[configuration_id].receipts,
                case_wall_latency_ms=tuple(timings[configuration_id]),
            )
        )
    return tuple(manifests), schedule


async def _main() -> None:
    args = _args()
    revision = require_sha256(args.source_revision, name="source_revision")
    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is required")

    client = ObservedQualifiedNvidiaNimClient(api_key, endpoint=args.endpoint)
    available, discovery_mode = client.discover_models()
    selected_models = client.select_models(available, count=3)
    qualification = client.provider_qualification()
    provider_fingerprint = require_sha256(
        qualification["fingerprint"],
        name="provider_qualification_fingerprint",
    )
    configurations = build_evaluation_configurations(selected_models)
    manifests, schedule = await _run_all_configurations(
        client=client,
        configurations=configurations,
        source_revision=revision,
        provider_fingerprint=provider_fingerprint,
    )

    payload = {
        "kind": "neuros_nim_swarm_live_eval_raw_v1",
        "repository": REPOSITORY,
        "source_revision": revision,
        "corpus_sha256": BENCHMARK_CORPUS_SHA256,
        "case_ids": list(SMOKE_CASE_IDS),
        "discovery_mode": discovery_mode,
        "selected_models": list(selected_models),
        "provider_qualification": qualification,
        "schedule_policy": SCHEDULE_POLICY,
        "execution_schedule": [
            {"case_id": case_id, "configuration_id": configuration_id}
            for case_id, configuration_id in schedule
        ],
        "configurations": [manifest.to_dict() for manifest in manifests],
        "review_call_count": sum(len(manifest.receipts) for manifest in manifests),
        "smoke_is_full_benchmark": False,
        "smoke_is_model_performance_claim": False,
        "live_review_output_is_scientific_authority": False,
        "merge_authority": False,
        "provider_execution_authority": False,
        "scientific_promotion_authority": False,
    }
    payload["artifact_sha256"] = canonical_sha256(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    asyncio.run(_main())
