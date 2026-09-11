"""Bounded live-evaluation telemetry for the neurOS scientific-engineering swarm.

The model-facing path imports only the public benchmark surface. Scorer-side ground
truth remains outside this module and can be applied later in a separate process.
"""
from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from ._canonical import canonical_sha256, require_nonempty, require_sha256
from .nim_observed import (
    NimTokenUsage,
    ObservedNimResponseError,
    ObservedQualifiedNvidiaNimClient,
)
from .nim_swarm import ROLE_PROMPTS, _user_prompt, build_nvidia_council
from .swarm import (
    AgentReview,
    CouncilMember,
    CouncilRun,
    Finding,
    SealedSwarmTask,
    parse_review_payload,
)
from .swarm_benchmark_cases import (
    BENCHMARK_CORPUS_SHA256,
    DEFECT_IDS,
    benchmark_case,
    build_benchmark_task,
)

EvaluationKind = Literal[
    "single_reference",
    "homogeneous_five_role",
    "heterogeneous_five_role",
]
ReceiptOutcome = Literal[
    "success",
    "provider_failure",
    "review_validation_failure",
]

# Intentionally fixed before any hosted-model result is observed. This is a transport/schema
# smoke slice, not a statistically meaningful model benchmark.
SMOKE_CASE_IDS = ("case-001", "case-015", "case-029")
SCHEDULE_POLICY = "deterministic_rotating_configuration_order_v1"

_GENERALIST_PROMPT = (
    "You are one independent neurOS scientific-engineering reviewer. You are advisory only: "
    "you cannot merge code, authorize provider execution, alter frozen scientific authority, "
    "or promote a scientific claim. Return exactly one JSON object matching the requested "
    "schema. Audit across architecture, scientific validity, reproducibility, implementation, "
    "and experimental design. Preserve concrete minority concerns rather than forcing consensus. "
    + " ".join(ROLE_PROMPTS.values())
)


def _require_nonnegative_int(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class EvaluationConfiguration:
    configuration_id: str
    kind: EvaluationKind
    members: tuple[CouncilMember, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "configuration_id",
            require_nonempty(self.configuration_id, name="configuration_id"),
        )
        if self.kind not in {
            "single_reference",
            "homogeneous_five_role",
            "heterogeneous_five_role",
        }:
            raise ValueError(f"unsupported evaluation kind {self.kind!r}")
        if not self.members:
            raise ValueError("evaluation configuration requires at least one member")
        member_ids = [member.member_id for member in self.members]
        if len(member_ids) != len(set(member_ids)):
            raise ValueError("evaluation member IDs must be unique")
        if self.kind == "single_reference" and len(self.members) != 1:
            raise ValueError("single_reference requires exactly one reviewer")
        if self.kind != "single_reference" and len(self.members) != 5:
            raise ValueError("council configurations require exactly five reviewers")
        models = {member.model for member in self.members}
        if self.kind == "homogeneous_five_role" and len(models) != 1:
            raise ValueError("homogeneous council must use one model route")
        if self.kind == "heterogeneous_five_role" and len(models) < 2:
            raise ValueError("heterogeneous council requires at least two model routes")

    def to_dict(self) -> dict[str, Any]:
        return {
            "configuration_id": self.configuration_id,
            "kind": self.kind,
            "members": [
                {
                    "member_id": member.member_id,
                    "role": member.role,
                    "model": member.model,
                    "prompt_sha256": member.prompt_sha256,
                }
                for member in sorted(self.members, key=lambda item: item.member_id)
            ],
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def build_evaluation_configurations(
    qualified_models: tuple[str, ...],
) -> tuple[EvaluationConfiguration, ...]:
    """Freeze reference, homogeneous, and available heterogeneous configurations.

    The first qualified route is the preregistered reference. It is not declared to be the
    strongest route based on benchmark outcomes, preventing outcome-adaptive model selection.
    """
    models = tuple(
        dict.fromkeys(str(model).strip() for model in qualified_models if str(model).strip())
    )
    if not models:
        raise ValueError("at least one qualified model route is required")
    reference = models[0]
    single = EvaluationConfiguration(
        configuration_id="single-reference-v1",
        kind="single_reference",
        members=(
            CouncilMember(
                member_id="generalist:1",
                role="generalist",
                model=reference,
                system_prompt=_GENERALIST_PROMPT,
            ),
        ),
    )
    homogeneous = EvaluationConfiguration(
        configuration_id="homogeneous-five-role-v1",
        kind="homogeneous_five_role",
        members=build_nvidia_council((reference,)),
    )
    configurations = [single, homogeneous]
    if len(models) >= 2:
        configurations.append(
            EvaluationConfiguration(
                configuration_id="heterogeneous-five-role-v1",
                kind="heterogeneous_five_role",
                members=build_nvidia_council(models),
            )
        )
    return tuple(configurations)


def build_counterbalanced_schedule(
    configurations: tuple[EvaluationConfiguration, ...],
    case_ids: tuple[str, ...] = SMOKE_CASE_IDS,
) -> tuple[tuple[str, str], ...]:
    """Rotate frozen configuration order by case to reduce simple temporal confounding."""
    if not configurations:
        raise ValueError("counterbalanced schedule requires at least one configuration")
    if not case_ids or len(case_ids) != len(set(case_ids)):
        raise ValueError("counterbalanced schedule case IDs must be non-empty and unique")
    for case_id in case_ids:
        benchmark_case(case_id)
    rows: list[tuple[str, str]] = []
    size = len(configurations)
    for case_index, case_id in enumerate(case_ids):
        offset = case_index % size
        ordered = configurations[offset:] + configurations[:offset]
        rows.extend((case_id, configuration.configuration_id) for configuration in ordered)
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class EvaluationCallReceipt:
    case_id: str
    task_sha256: str
    member_id: str
    role: str
    model: str
    member_prompt_sha256: str
    provider_qualification_fingerprint: str
    endpoint: str
    outcome: ReceiptOutcome
    latency_ms: int
    call_prompt_sha256: str | None = None
    request_sha256: str | None = None
    response_sha256: str | None = None
    parsed_response_sha256: str | None = None
    token_usage: NimTokenUsage = field(default_factory=NimTokenUsage)
    error_class: str | None = None

    def __post_init__(self) -> None:
        for name in ("case_id", "member_id", "role", "model", "endpoint"):
            object.__setattr__(
                self,
                name,
                require_nonempty(getattr(self, name), name=name),
            )
        for name in (
            "task_sha256",
            "member_prompt_sha256",
            "provider_qualification_fingerprint",
        ):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        if self.outcome not in {
            "success",
            "provider_failure",
            "review_validation_failure",
        }:
            raise ValueError(f"unsupported receipt outcome {self.outcome!r}")
        object.__setattr__(
            self,
            "latency_ms",
            _require_nonnegative_int(self.latency_ms, name="latency_ms"),
        )

        transport_hashes = (
            self.call_prompt_sha256,
            self.request_sha256,
            self.response_sha256,
        )
        if self.outcome == "success":
            if any(value is None for value in (*transport_hashes, self.parsed_response_sha256)):
                raise ValueError("successful receipt requires complete call identities")
            if self.error_class is not None:
                raise ValueError("successful receipt cannot carry error_class")
        elif self.outcome == "provider_failure":
            if any(value is not None for value in (*transport_hashes, self.parsed_response_sha256)):
                raise ValueError("provider failure cannot claim unavailable call identities")
            if self.token_usage.provider_reported:
                raise ValueError("provider failure cannot claim provider token usage")
            object.__setattr__(
                self,
                "error_class",
                require_nonempty(self.error_class or "", name="error_class"),
            )
        else:
            if any(value is None for value in transport_hashes):
                raise ValueError("review-validation failure requires transport call identities")
            object.__setattr__(
                self,
                "error_class",
                require_nonempty(self.error_class or "", name="error_class"),
            )

        for index, value in enumerate(transport_hashes):
            if value is not None:
                require_sha256(value, name=f"transport_hash_{index}")
        if self.parsed_response_sha256 is not None:
            require_sha256(self.parsed_response_sha256, name="parsed_response_sha256")

    def _payload(self) -> dict[str, Any]:
        return {
            "schema": "neuros.nim_swarm_evaluation_call_receipt.v1",
            "case_id": self.case_id,
            "task_sha256": self.task_sha256,
            "member_id": self.member_id,
            "role": self.role,
            "model": self.model,
            "member_prompt_sha256": self.member_prompt_sha256,
            "provider_qualification_fingerprint": self.provider_qualification_fingerprint,
            "endpoint": self.endpoint,
            "outcome": self.outcome,
            "latency_ms": self.latency_ms,
            "call_prompt_sha256": self.call_prompt_sha256,
            "request_sha256": self.request_sha256,
            "response_sha256": self.response_sha256,
            "parsed_response_sha256": self.parsed_response_sha256,
            "token_usage": self.token_usage.to_dict(),
            "error_class": self.error_class,
            "token_counts_are_provider_reported_not_estimated": True,
            "receipt_is_scientific_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["receipt_sha256"] = canonical_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload())


def _benchmark_case_id(task: SealedSwarmTask) -> str:
    context = task.public_context
    if context.get("corpus_sha256") != BENCHMARK_CORPUS_SHA256:
        raise ValueError("task is not bound to the promoted benchmark corpus")
    case = context.get("case")
    if not isinstance(case, Mapping):
        raise ValueError("benchmark task is missing public case payload")
    case_id = require_nonempty(str(case.get("case_id", "")), name="case_id")
    if benchmark_case(case_id).to_public_dict() != dict(case):
        raise ValueError("benchmark task case payload differs from frozen public corpus")
    return case_id


def _latency_ms(started_ns: int) -> int:
    return max(0, (time.perf_counter_ns() - started_ns + 999_999) // 1_000_000)


class ObservedNvidiaCouncilTransport:
    """Concurrent NIM transport that binds each reviewer result to exact telemetry."""

    def __init__(
        self,
        client: ObservedQualifiedNvidiaNimClient,
        *,
        provider_qualification_fingerprint: str,
        max_tokens: int = 1200,
    ) -> None:
        self.client = client
        self.provider_qualification_fingerprint = require_sha256(
            provider_qualification_fingerprint,
            name="provider_qualification_fingerprint",
        )
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 512 or self.max_tokens > 8192:
            raise ValueError("max_tokens must be in [512, 8192]")
        self._receipts: list[EvaluationCallReceipt] = []

    @property
    def receipts(self) -> tuple[EvaluationCallReceipt, ...]:
        return tuple(
            sorted(
                self._receipts,
                key=lambda receipt: (receipt.case_id, receipt.member_id),
            )
        )

    def _receipt_from_record(
        self,
        *,
        task: SealedSwarmTask,
        case_id: str,
        member: CouncilMember,
        outcome: ReceiptOutcome,
        latency_ms: int,
        record,  # type: ignore[no-untyped-def]
        usage: NimTokenUsage,
        parsed: dict[str, Any] | None = None,
        error_class: str | None = None,
    ) -> EvaluationCallReceipt:
        return EvaluationCallReceipt(
            case_id=case_id,
            task_sha256=task.sha256,
            member_id=member.member_id,
            role=member.role,
            model=member.model,
            member_prompt_sha256=member.prompt_sha256,
            provider_qualification_fingerprint=self.provider_qualification_fingerprint,
            endpoint=record.endpoint,
            outcome=outcome,
            latency_ms=latency_ms,
            call_prompt_sha256=record.prompt_sha256,
            request_sha256=record.request_sha256,
            response_sha256=record.response_sha256,
            parsed_response_sha256=(canonical_sha256(parsed) if parsed is not None else None),
            token_usage=usage,
            error_class=error_class,
        )

    async def review(
        self,
        task: SealedSwarmTask,
        member: CouncilMember,
    ) -> dict[str, Any]:
        case_id = _benchmark_case_id(task)
        user_prompt = _user_prompt(task)
        started_ns = time.perf_counter_ns()
        try:
            parsed, record, usage = await asyncio.to_thread(
                self.client.chat_json_observed,
                role=f"swarm:{member.role}",
                model=member.model,
                system_prompt=member.system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
                temperature=0.1,
            )
        except ObservedNimResponseError as exc:
            self._receipts.append(
                self._receipt_from_record(
                    task=task,
                    case_id=case_id,
                    member=member,
                    outcome="review_validation_failure",
                    latency_ms=_latency_ms(started_ns),
                    record=exc.record,
                    usage=exc.token_usage,
                    error_class=type(exc).__name__,
                )
            )
            raise
        except Exception as exc:
            self._receipts.append(
                EvaluationCallReceipt(
                    case_id=case_id,
                    task_sha256=task.sha256,
                    member_id=member.member_id,
                    role=member.role,
                    model=member.model,
                    member_prompt_sha256=member.prompt_sha256,
                    provider_qualification_fingerprint=self.provider_qualification_fingerprint,
                    endpoint=self.client.endpoint,
                    outcome="provider_failure",
                    latency_ms=_latency_ms(started_ns),
                    error_class=type(exc).__name__,
                )
            )
            raise

        try:
            if record.role != f"swarm:{member.role}":
                raise ValueError("provider call record role differs from council member")
            if record.model != member.model:
                raise ValueError("provider call record model differs from council member")
            if record.endpoint != self.client.endpoint:
                raise ValueError("provider call record endpoint differs from qualified client")
            validated = parse_review_payload(parsed, task=task, member=member)
            unknown = sorted({finding.finding_id for finding in validated.findings} - DEFECT_IDS)
            if unknown:
                raise ValueError("review emitted unknown benchmark defect IDs: " + ",".join(unknown))
        except Exception as exc:
            self._receipts.append(
                self._receipt_from_record(
                    task=task,
                    case_id=case_id,
                    member=member,
                    outcome="review_validation_failure",
                    latency_ms=_latency_ms(started_ns),
                    record=record,
                    usage=usage,
                    parsed=parsed,
                    error_class=type(exc).__name__,
                )
            )
            raise

        self._receipts.append(
            self._receipt_from_record(
                task=task,
                case_id=case_id,
                member=member,
                outcome="success",
                latency_ms=_latency_ms(started_ns),
                record=record,
                usage=usage,
                parsed=parsed,
            )
        )
        return parsed


@dataclass(frozen=True, slots=True)
class EvaluationRunManifest:
    repository: str
    source_revision: str
    configuration: EvaluationConfiguration
    provider_qualification_fingerprint: str
    case_ids: tuple[str, ...]
    runs: tuple[tuple[str, CouncilRun], ...]
    receipts: tuple[EvaluationCallReceipt, ...]
    case_wall_latency_ms: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        repo = require_nonempty(self.repository, name="repository")
        revision = require_sha256(self.source_revision, name="source_revision")
        provider = require_sha256(
            self.provider_qualification_fingerprint,
            name="provider_qualification_fingerprint",
        )
        object.__setattr__(self, "repository", repo)
        object.__setattr__(self, "source_revision", revision)
        object.__setattr__(self, "provider_qualification_fingerprint", provider)

        case_ids = tuple(require_nonempty(case_id, name="case_id") for case_id in self.case_ids)
        if not case_ids or len(case_ids) != len(set(case_ids)):
            raise ValueError("case_ids must be non-empty and unique")
        for case_id in case_ids:
            benchmark_case(case_id)
        object.__setattr__(self, "case_ids", case_ids)

        run_map = dict(self.runs)
        if len(run_map) != len(self.runs) or set(run_map) != set(case_ids):
            raise ValueError("runs must contain exactly one entry for every case_id")
        timing_map = dict(self.case_wall_latency_ms)
        if len(timing_map) != len(self.case_wall_latency_ms) or set(timing_map) != set(case_ids):
            raise ValueError("case wall timings must contain exactly one entry for every case_id")
        normalized_timings = tuple(
            (
                case_id,
                _require_nonnegative_int(
                    timing_map[case_id],
                    name=f"case_wall_latency_ms[{case_id}]",
                ),
            )
            for case_id in case_ids
        )
        object.__setattr__(self, "case_wall_latency_ms", normalized_timings)

        member_map = {member.member_id: member for member in self.configuration.members}
        expected_pairs = {(case_id, member_id) for case_id in case_ids for member_id in member_map}
        receipt_map = {(receipt.case_id, receipt.member_id): receipt for receipt in self.receipts}
        if len(receipt_map) != len(self.receipts) or set(receipt_map) != expected_pairs:
            raise ValueError("receipts must contain exactly one attempt per case/member pair")

        for case_id in case_ids:
            task = build_benchmark_task(
                case_id,
                repository=repo,
                source_revision=revision,
            )
            run = run_map[case_id]
            if run.task_sha256 != task.sha256:
                raise ValueError("council run is not bound to expected benchmark task")
            successful = {review.member_id: review for review in run.reviews}
            failed: dict[str, str] = {}
            for entry in run.failed_members:
                member_id, separator, error_class = entry.rpartition(":")
                if not separator or not member_id or not error_class:
                    raise ValueError("failed member identity is malformed")
                if member_id in failed:
                    raise ValueError("run repeats a failed reviewer identity")
                failed[member_id] = error_class
            if set(successful) | set(failed) != set(member_map):
                raise ValueError("run does not account for every configured reviewer")
            if set(successful) & set(failed):
                raise ValueError("reviewer cannot be both successful and failed")

            for member_id, member in member_map.items():
                receipt = receipt_map[(case_id, member_id)]
                if receipt.task_sha256 != task.sha256:
                    raise ValueError("receipt task identity mismatch")
                if receipt.provider_qualification_fingerprint != provider:
                    raise ValueError("receipt provider qualification mismatch")
                if (
                    receipt.role != member.role
                    or receipt.model != member.model
                    or receipt.member_prompt_sha256 != member.prompt_sha256
                ):
                    raise ValueError("receipt member configuration mismatch")
                if member_id in successful:
                    review = successful[member_id]
                    if (
                        review.role != member.role
                        or review.model != member.model
                        or review.prompt_sha256 != member.prompt_sha256
                    ):
                        raise ValueError("successful review member configuration mismatch")
                    if receipt.outcome != "success":
                        raise ValueError("successful review lacks successful receipt")
                    if receipt.parsed_response_sha256 != review.response_sha256:
                        raise ValueError("receipt parsed-response identity mismatch")
                else:
                    if receipt.outcome not in {
                        "provider_failure",
                        "review_validation_failure",
                    }:
                        raise ValueError("failed review lacks failure receipt")
                    if receipt.error_class != failed[member_id]:
                        raise ValueError("failed review error class does not match receipt")

    def _payload(self) -> dict[str, Any]:
        run_map = dict(self.runs)
        timing_map = dict(self.case_wall_latency_ms)
        return {
            "schema": "neuros.nim_swarm_live_evaluation_manifest.v1",
            "repository": self.repository,
            "source_revision": self.source_revision,
            "corpus_sha256": BENCHMARK_CORPUS_SHA256,
            "configuration": self.configuration.to_dict(),
            "configuration_sha256": self.configuration.sha256,
            "provider_qualification_fingerprint": self.provider_qualification_fingerprint,
            "case_ids": list(self.case_ids),
            "runs": {case_id: run_map[case_id].to_dict() for case_id in sorted(run_map)},
            "receipts": [receipt.to_dict() for receipt in self.receipts],
            "case_wall_latency_ms": {
                case_id: timing_map[case_id] for case_id in sorted(timing_map)
            },
            "token_counts_are_provider_reported_not_estimated": True,
            "dollar_cost_estimated": False,
            "live_review_output_is_scientific_authority": False,
            "merge_authority": False,
            "provider_execution_authority": False,
            "scientific_promotion_authority": False,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["manifest_sha256"] = canonical_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return canonical_sha256(self._payload())


def council_run_from_dict(payload: dict[str, Any]) -> CouncilRun:
    """Reconstruct and verify a provider-neutral council run for scorer-side use."""
    expected = {
        "schema",
        "task_sha256",
        "reviews",
        "failed_members",
        "majority_vote_is_authority",
        "merge_authority",
        "provider_execution_authority",
        "scientific_promotion_authority",
        "run_sha256",
    }
    if set(payload) != expected:
        raise ValueError("council run fields do not match the v1 schema")
    if payload["schema"] != "neuros.scientific_engineering_swarm_run.v1":
        raise ValueError("unexpected council run schema")
    for key in (
        "majority_vote_is_authority",
        "merge_authority",
        "provider_execution_authority",
        "scientific_promotion_authority",
    ):
        if payload[key] is not False:
            raise ValueError(f"council authority flag {key} must remain false")
    rows = payload["reviews"]
    if not isinstance(rows, list):
        raise ValueError("council reviews must be a list")
    reviews = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("council review must be a JSON object")
        review_expected = {
            "task_sha256",
            "member_id",
            "role",
            "model",
            "prompt_sha256",
            "response_sha256",
            "findings",
        }
        if set(row) != review_expected or not isinstance(row["findings"], list):
            raise ValueError("council review fields do not match the v1 schema")
        findings = tuple(
            Finding.from_dict(finding)
            for finding in row["findings"]
            if isinstance(finding, dict)
        )
        if len(findings) != len(row["findings"]):
            raise ValueError("every serialized finding must be a JSON object")
        reviews.append(
            AgentReview(
                task_sha256=row["task_sha256"],
                member_id=row["member_id"],
                role=row["role"],
                model=row["model"],
                prompt_sha256=row["prompt_sha256"],
                response_sha256=row["response_sha256"],
                findings=findings,
            )
        )
    failed_members = payload["failed_members"]
    if not isinstance(failed_members, list):
        raise ValueError("failed_members must be a list")
    run = CouncilRun(
        task_sha256=payload["task_sha256"],
        reviews=tuple(reviews),
        failed_members=tuple(str(value) for value in failed_members),
    )
    if run.sha256 != require_sha256(payload["run_sha256"], name="run_sha256"):
        raise ValueError("serialized council run fingerprint mismatch")
    return run
