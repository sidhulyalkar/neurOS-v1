"""Deterministic, provider-neutral scientific-engineering council for neurOS.

This module is advisory only. It can produce review findings and proposal artifacts,
but it has no Git, provider-execution, ExperimentEvidence, or promotion authority.
"""
from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

from ._canonical import (
    canonical_sha256,
    freeze_json,
    require_nonempty,
    require_sha256,
    thaw_json,
)

Severity = Literal["info", "low", "medium", "high", "critical"]
Category = Literal[
    "architecture",
    "scientific_validity",
    "reproducibility",
    "implementation",
    "experimental_design",
    "security",
    "performance",
]

_ALLOWED_SEVERITIES = {"info", "low", "medium", "high", "critical"}
_ALLOWED_CATEGORIES = {
    "architecture",
    "scientific_validity",
    "reproducibility",
    "implementation",
    "experimental_design",
    "security",
    "performance",
}
_FORBIDDEN_TASK_KEYS = {
    "credential",
    "credentials",
    "api_key",
    "token",
    "secret",
    "password",
    "raw_participant_data",
    "participant_identifier",
    "hidden_target",
    "private_leaderboard",
}


def _contains_forbidden_key(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).strip().lower() in _FORBIDDEN_TASK_KEYS:
                return True
            if _contains_forbidden_key(item):
                return True
    elif isinstance(value, (list, tuple)):
        return any(_contains_forbidden_key(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class SealedSwarmTask:
    repository: str
    source_revision: str
    objective: str
    claim_boundary: str
    allowed_paths: tuple[str, ...] = ()
    authority_sha256s: tuple[str, ...] = ()
    forbidden_actions: tuple[str, ...] = ()
    public_context: Mapping[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "repository", require_nonempty(self.repository, name="repository"))
        object.__setattr__(
            self,
            "source_revision",
            require_nonempty(self.source_revision, name="source_revision"),
        )
        object.__setattr__(self, "objective", require_nonempty(self.objective, name="objective"))
        object.__setattr__(
            self,
            "claim_boundary",
            require_nonempty(self.claim_boundary, name="claim_boundary"),
        )
        for name in ("allowed_paths", "forbidden_actions"):
            values = tuple(require_nonempty(v, name=name) for v in getattr(self, name))
            if len(set(values)) != len(values):
                raise ValueError(f"{name} values must be unique")
            object.__setattr__(self, name, values)
        hashes = tuple(
            require_sha256(v, name="authority_sha256") for v in self.authority_sha256s
        )
        if len(set(hashes)) != len(hashes):
            raise ValueError("authority_sha256s values must be unique")
        object.__setattr__(self, "authority_sha256s", hashes)
        context = freeze_json(self.public_context, path="swarm.public_context")
        if _contains_forbidden_key(context):
            raise ValueError("public_context contains a forbidden secret/private-data key")
        object.__setattr__(self, "public_context", context)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "neuros.scientific_engineering_swarm_task.v1",
            "repository": self.repository,
            "source_revision": self.source_revision,
            "objective": self.objective,
            "claim_boundary": self.claim_boundary,
            "allowed_paths": list(self.allowed_paths),
            "authority_sha256s": list(self.authority_sha256s),
            "forbidden_actions": list(self.forbidden_actions),
            "public_context": thaw_json(self.public_context),
            "llm_output_is_authority": False,
        }

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


@dataclass(frozen=True, slots=True)
class CouncilMember:
    member_id: str
    role: str
    model: str
    system_prompt: str

    def __post_init__(self) -> None:
        for name in ("member_id", "role", "model", "system_prompt"):
            object.__setattr__(
                self,
                name,
                require_nonempty(getattr(self, name), name=name),
            )

    @property
    def prompt_sha256(self) -> str:
        return hashlib.sha256(self.system_prompt.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Finding:
    finding_id: str
    severity: Severity
    category: Category
    claim: str
    evidence: str
    falsification_test: str
    proposed_repair: str
    confidence: float
    requires_human_judgment: bool
    reference: str = ""

    def __post_init__(self) -> None:
        for name in ("finding_id", "claim", "evidence", "falsification_test"):
            object.__setattr__(
                self,
                name,
                require_nonempty(getattr(self, name), name=name),
            )
        if self.severity not in _ALLOWED_SEVERITIES:
            raise ValueError(f"unsupported severity {self.severity!r}")
        if self.category not in _ALLOWED_CATEGORIES:
            raise ValueError(f"unsupported category {self.category!r}")
        if not 0.0 <= float(self.confidence) <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        object.__setattr__(self, "confidence", float(self.confidence))
        object.__setattr__(
            self,
            "requires_human_judgment",
            bool(self.requires_human_judgment),
        )
        object.__setattr__(self, "reference", str(self.reference).strip())
        object.__setattr__(self, "proposed_repair", str(self.proposed_repair).strip())

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Finding:
        expected = {
            "finding_id",
            "severity",
            "category",
            "claim",
            "evidence",
            "falsification_test",
            "proposed_repair",
            "confidence",
            "requires_human_judgment",
            "reference",
        }
        if set(payload) != expected:
            raise ValueError("finding fields must match the v1 schema exactly")
        return cls(**payload)  # type: ignore[arg-type]

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "severity": self.severity,
            "category": self.category,
            "claim": self.claim,
            "evidence": self.evidence,
            "falsification_test": self.falsification_test,
            "proposed_repair": self.proposed_repair,
            "confidence": self.confidence,
            "requires_human_judgment": self.requires_human_judgment,
            "reference": self.reference,
        }


@dataclass(frozen=True, slots=True)
class AgentReview:
    task_sha256: str
    member_id: str
    role: str
    model: str
    prompt_sha256: str
    response_sha256: str
    findings: tuple[Finding, ...]

    def __post_init__(self) -> None:
        for name in ("task_sha256", "prompt_sha256", "response_sha256"):
            object.__setattr__(
                self,
                name,
                require_sha256(getattr(self, name), name=name),
            )
        for name in ("member_id", "role", "model"):
            object.__setattr__(
                self,
                name,
                require_nonempty(getattr(self, name), name=name),
            )
        ids = [finding.finding_id for finding in self.findings]
        if len(ids) != len(set(ids)):
            raise ValueError("finding IDs must be unique within one review")

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_sha256": self.task_sha256,
            "member_id": self.member_id,
            "role": self.role,
            "model": self.model,
            "prompt_sha256": self.prompt_sha256,
            "response_sha256": self.response_sha256,
            "findings": [finding.to_dict() for finding in self.findings],
        }


class CouncilTransport(Protocol):
    async def review(
        self,
        task: SealedSwarmTask,
        member: CouncilMember,
    ) -> dict[str, Any]: ...


def parse_review_payload(
    payload: dict[str, Any],
    *,
    task: SealedSwarmTask,
    member: CouncilMember,
) -> AgentReview:
    if set(payload) != {"findings"} or not isinstance(payload["findings"], list):
        raise ValueError("review response must contain exactly one findings list")
    findings = tuple(
        Finding.from_dict(row) for row in payload["findings"] if isinstance(row, dict)
    )
    if len(findings) != len(payload["findings"]):
        raise ValueError("every finding must be a JSON object")
    return AgentReview(
        task_sha256=task.sha256,
        member_id=member.member_id,
        role=member.role,
        model=member.model,
        prompt_sha256=member.prompt_sha256,
        response_sha256=canonical_sha256(payload),
        findings=findings,
    )


@dataclass(frozen=True, slots=True)
class CouncilRun:
    task_sha256: str
    reviews: tuple[AgentReview, ...]
    failed_members: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "task_sha256",
            require_sha256(self.task_sha256, name="task_sha256"),
        )
        members = [review.member_id for review in self.reviews]
        if len(members) != len(set(members)):
            raise ValueError("council reviews must have unique member IDs")
        if any(review.task_sha256 != self.task_sha256 for review in self.reviews):
            raise ValueError("all reviews must bind the same sealed task")
        failed = tuple(
            require_nonempty(value, name="failed_member") for value in self.failed_members
        )
        if len(set(failed)) != len(failed) or set(failed) & set(members):
            raise ValueError(
                "failed_members must be unique and disjoint from successful reviews"
            )
        object.__setattr__(self, "failed_members", failed)

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(self.reviews, key=lambda review: review.member_id)
        payload = {
            "schema": "neuros.scientific_engineering_swarm_run.v1",
            "task_sha256": self.task_sha256,
            "reviews": [review.to_dict() for review in ordered],
            "failed_members": sorted(self.failed_members),
            "majority_vote_is_authority": False,
            "merge_authority": False,
            "provider_execution_authority": False,
            "scientific_promotion_authority": False,
        }
        payload["run_sha256"] = canonical_sha256(payload)
        return payload

    @property
    def sha256(self) -> str:
        return self.to_dict()["run_sha256"]

    @property
    def findings(self) -> tuple[Finding, ...]:
        return tuple(
            finding for review in self.reviews for finding in review.findings
        )


async def run_council(
    task: SealedSwarmTask,
    members: tuple[CouncilMember, ...],
    transport: CouncilTransport,
    *,
    require_all: bool = True,
) -> CouncilRun:
    if not members:
        raise ValueError("council must contain at least one member")
    ids = [member.member_id for member in members]
    if len(ids) != len(set(ids)):
        raise ValueError("council member IDs must be unique")

    async def one(member: CouncilMember):
        try:
            payload = await transport.review(task, member)
            return parse_review_payload(payload, task=task, member=member), None
        except Exception as exc:
            # Preserve only bounded failure identity here. Provider-specific diagnostics are
            # already retained by the qualified transport and must not leak into this authority-
            # neutral manifest.
            return None, f"{member.member_id}:{type(exc).__name__}"

    results = await asyncio.gather(*(one(member) for member in members))
    reviews = tuple(row for row, _ in results if row is not None)
    failures = tuple(error for _, error in results if error is not None)
    if require_all and failures:
        raise RuntimeError("council member failure: " + ",".join(sorted(failures)))
    return CouncilRun(
        task_sha256=task.sha256,
        reviews=reviews,
        failed_members=failures,
    )


def finding_support(run: CouncilRun) -> dict[str, tuple[str, ...]]:
    """Return support by exact finding ID, preserving minority findings without voting."""
    support: dict[str, set[str]] = {}
    for review in run.reviews:
        for finding in review.findings:
            support.setdefault(finding.finding_id, set()).add(review.member_id)
    return {
        finding_id: tuple(sorted(member_ids))
        for finding_id, member_ids in sorted(support.items())
    }
