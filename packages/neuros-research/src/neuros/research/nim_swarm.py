"""NVIDIA NIM transport for the neurOS scientific-engineering council."""
from __future__ import annotations

import asyncio
import json
from typing import Any

from ._canonical import canonical_sha256
from .nim_provider import QualifiedNvidiaNimClient
from .swarm import CouncilMember, SealedSwarmTask

ROLE_PROMPTS: dict[str, str] = {
    "architecture": (
        "Audit package boundaries, interfaces, unnecessary complexity, coupling, and long-term "
        "maintainability. Prefer simplification when it preserves capability."
    ),
    "scientific_adversary": (
        "Attack scientific validity: leakage, circularity, hidden adaptivity, confounding, "
        "invalid units of analysis, overclaiming, and unfalsifiable reasoning."
    ),
    "reproducibility": (
        "Audit provenance, determinism, exact-revision binding, replayability, stale evidence, "
        "incomplete artifacts, environment drift, and evidence-chain attacks."
    ),
    "implementation": (
        "Audit correctness, edge cases, portability, concurrency, performance, dependency "
        "assumptions, and missing executable tests."
    ),
    "experimental_design": (
        "Propose the highest-information falsification tests, controls, ablations, stopping "
        "rules, and cheap experiments before expensive compute."
    ),
}

_OUTPUT_SCHEMA = {
    "findings": [
        {
            "finding_id": "stable-short-id",
            "severity": "info | low | medium | high | critical",
            "category": (
                "architecture | scientific_validity | reproducibility | implementation | "
                "experimental_design | security | performance"
            ),
            "claim": "specific defect or testable concern",
            "evidence": "specific evidence from the sealed packet",
            "falsification_test": "executable or inspectable test that could refute the concern",
            "proposed_repair": "specific repair or empty string",
            "confidence": 0.0,
            "requires_human_judgment": False,
            "reference": "file:symbol/line or contract reference, or empty string",
        }
    ]
}


def build_nvidia_council(
    qualified_models: tuple[str, ...],
) -> tuple[CouncilMember, ...]:
    """Build five independent roles distributed across live-qualified models."""
    models = tuple(
        dict.fromkeys(
            str(model).strip()
            for model in qualified_models
            if str(model).strip()
        )
    )
    if not models:
        raise ValueError("at least one live-qualified NVIDIA model is required")

    members = []
    for index, (role, instruction) in enumerate(ROLE_PROMPTS.items()):
        model = models[index % len(models)]
        system_prompt = (
            "You are one independent neurOS scientific-engineering council member. "
            "You are advisory only: you cannot merge code, authorize provider execution, alter "
            "frozen scientific authority, or promote a scientific claim. Do not infer secret or "
            "private data. Return exactly one JSON object matching the requested schema. "
            "Do not suppress a finding because another agent may disagree. "
            + instruction
        )
        members.append(
            CouncilMember(
                member_id=f"{role}:{index + 1}",
                role=role,
                model=model,
                system_prompt=system_prompt,
            )
        )
    return tuple(members)


def _user_prompt(task: SealedSwarmTask) -> str:
    task_json = json.dumps(
        task.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    schema_json = json.dumps(
        _OUTPUT_SCHEMA,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return (
        "Audit the sealed task below independently. Report zero or more concrete findings. "
        "Every finding must include a falsification test. Finding IDs should be stable and "
        "derived from the defect concept, not confidence or severity. Do not request additional "
        "private data.\n\n"
        "SEALED_TASK_SHA256="
        + task.sha256
        + "\n\nSEALED_TASK="
        + task_json
        + "\n\nOUTPUT_SCHEMA="
        + schema_json
    )


class NvidiaCouncilTransport:
    """Concurrent council adapter over an already-qualified NVIDIA NIM client."""

    def __init__(
        self,
        client: QualifiedNvidiaNimClient,
        *,
        max_tokens: int = 2800,
    ) -> None:
        self.client = client
        self.max_tokens = int(max_tokens)
        if self.max_tokens < 512 or self.max_tokens > 8192:
            raise ValueError("max_tokens must be in [512, 8192]")

    async def review(
        self,
        task: SealedSwarmTask,
        member: CouncilMember,
    ) -> dict[str, Any]:
        user_prompt = _user_prompt(task)
        parsed, _record = await asyncio.to_thread(
            self.client.chat_json,
            role=f"swarm:{member.role}",
            model=member.model,
            system_prompt=member.system_prompt,
            user_prompt=user_prompt,
            max_tokens=self.max_tokens,
            temperature=0.1,
        )
        return parsed


def council_configuration_sha256(
    members: tuple[CouncilMember, ...],
) -> str:
    return canonical_sha256(
        [
            {
                "member_id": member.member_id,
                "role": member.role,
                "model": member.model,
                "prompt_sha256": member.prompt_sha256,
            }
            for member in sorted(members, key=lambda member: member.member_id)
        ]
    )
