"""Run a bounded NVIDIA NIM research-proposal tournament for Algonauts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from neuros.research import ExternalDispatchPolicy
from neuros.research._canonical import canonical_json, canonical_sha256
from neuros.research.nim import NvidiaNimClient, frozen_public_context, parse_proposals

_ALLOWED_DEVELOPMENT_METRICS = (
    "validation_pearson",
    "validation_mse",
    "rsa_spearman",
    "temporal_shift_null",
    "validation_stability",
    "runtime_seconds",
    "cache_gb",
    "complementarity_error_correlation",
)

_SYSTEM_PROMPT = """You are one research proposer inside neurOS. You do not control data,
splits, evaluation authority, promotion, or hidden-test access. Produce only the JSON object
requested by the user. Proposals must be falsifiable, development-only, and compatible with
the frozen information boundary. Never request raw participant data, participant identifiers,
hidden targets, private leaderboard feedback, credentials, or changing the evaluation split.
Prefer experiments that can teach us something even when they fail."""

_CRITIC_SYSTEM_PROMPT = """You are the adversarial-science critic in neurOS. You can critique
research proposals but cannot promote them or change their frozen evaluation authority.
Return only the requested JSON. Look specifically for leakage, confounding, duplicated ideas,
weak falsification, excessive compute, and claims that exceed the evidence described."""

_SYNTHESIS_SYSTEM_PROMPT = """You are a research-program synthesizer in neurOS. Recommend an
execution order for already validated proposals. You do not promote scientific claims and you
cannot change data, splits, metrics, or hidden-test policy. Return only the requested JSON."""


def _public_context() -> dict[str, Any]:
    dispatch = ExternalDispatchPolicy()
    return {
        "program": "neurOS x Algonaut-A-Mario",
        "scientific_goal": (
            "Prospectively test whether development-only neural-geometry evidence predicts "
            "which frozen representation families later generalize best to unseen game levels, "
            "cross-game transfer, and held-out subjects."
        ),
        "current_candidate_families": [
            "actions/state controls",
            "V-JEPA frozen features",
            "V-JEPA 2.1 denser temporal features",
            "train-only PCA and segment-aware diffusion neural geometry",
            "ridge/readout variants",
            "representation fusion and specialist selection",
        ],
        "frozen_rules": [
            "training and validation may be used for candidate development",
            "G2 unseen-level truth is opened only after the candidate is frozen",
            "G3 Mario-to-Shinobi truth is not model-selection feedback",
            "G4 held-subject truth is not model-selection feedback",
            "a proposer may not rewrite dataset, split, HRF, metric, or promotion authority",
            "failed and unavailable runs remain evidence",
        ],
        "allowed_payload_classes": list(dispatch.allowed_payload_classes),
        "prohibited_payload_classes": list(dispatch.prohibited_payload_classes),
        "allowed_development_metrics": list(_ALLOWED_DEVELOPMENT_METRICS),
        "priority_questions": [
            (
                "Can neural-geometry alignment prospectively screen representation families "
                "before OOD truth is opened?"
            ),
            (
                "Which temporal representation choices add signal beyond feature dimension, "
                "ordinary ridge capacity, and low-cost game-state controls?"
            ),
            (
                "When do geometry-aware or specialist readouts add complementary errors rather "
                "than merely rediscovering the global winner?"
            ),
            (
                "Which low-cost experiments maximize information gained per unit compute before "
                "expensive G2/G3/G4 evaluation?"
            ),
        ],
    }


def _generator_prompt(context: dict[str, Any]) -> str:
    schema = {
        "candidates": [
            {
                "candidate_id": "short-stable-id",
                "title": "short title",
                "statement": "falsifiable hypothesis",
                "rationale": "why this can change our understanding",
                "family": (
                    "representation | temporal_alignment | neural_geometry | readout | "
                    "fusion | generalization"
                ),
                "changed_variables": ["representation.example"],
                "required_payload_classes": ["source_code", "aggregate_metrics"],
                "development_metrics": ["validation_pearson", "rsa_spearman"],
                "falsification_test": "specific negative result that rejects or weakens the idea",
                "estimated_compute_tier": "low | medium | high",
                "expected_failure_modes": ["one concrete failure mode"],
            }
        ]
    }
    return (
        "Generate exactly 5 diverse candidate experiments. At least one must be a low-compute "
        "control, one must test the prospective neural-geometry-screening hypothesis, and one "
        "must test representation complementarity rather than raw winner score. Do not propose "
        "using OOD truth for selection. Use only payload classes and development metrics listed "
        "in the context. Avoid changing any dataset/evaluation/split authority.\n\n"
        f"PUBLIC_CONTEXT={canonical_json(context)}\n\n"
        f"OUTPUT_SCHEMA_EXAMPLE={canonical_json(schema)}"
    )


def _critic_prompt(context: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    schema = {
        "reviews": [
            {
                "candidate_id": "id",
                "verdict": "advance | revise | reject",
                "risk_flags": ["confound or leakage risk"],
                "critical_test": "single most important test",
                "revision": "specific revision or empty string",
            }
        ]
    }
    return (
        "Critique every candidate exactly once. A verdict is only planning advice, never a "
        "scientific promotion decision. Reject anything that depends on forbidden feedback, "
        "cannot be falsified, or confounds the changed variable with a major capacity/data "
        "difference. Prefer controls that isolate representation geometry, temporal density, "
        "and error complementarity.\n\n"
        f"PUBLIC_CONTEXT={canonical_json(context)}\n\n"
        f"VALIDATED_PROPOSALS={canonical_json(proposals)}\n\n"
        f"OUTPUT_SCHEMA_EXAMPLE={canonical_json(schema)}"
    )


def _synthesis_prompt(
    context: dict[str, Any],
    proposals: list[dict[str, Any]],
    reviews: dict[str, Any],
) -> str:
    schema = {
        "priority_queue": ["candidate-id-1", "candidate-id-2"],
        "rounds": [
            {
                "round": 1,
                "candidate_ids": ["candidate-id-1"],
                "reason": "why this ordering maximizes information per compute",
            }
        ],
        "stopping_rule": (
            "development-only criterion for stopping or revising before any OOD truth is opened"
        ),
    }
    return (
        "Create a 2-round development-only execution queue using only existing candidate IDs. "
        "Prioritize information gain, falsification strength, low-cost controls, and experiments "
        "whose outputs determine whether later expensive experiments are worth running. Do not "
        "claim any candidate is scientifically promoted.\n\n"
        f"PUBLIC_CONTEXT={canonical_json(context)}\n\n"
        f"VALIDATED_PROPOSALS={canonical_json(proposals)}\n\n"
        f"CRITIC_REVIEWS={canonical_json(reviews)}\n\n"
        f"OUTPUT_SCHEMA_EXAMPLE={canonical_json(schema)}"
    )


def _validate_reviews(payload: dict[str, Any], candidate_ids: set[str]) -> dict[str, Any]:
    rows = payload.get("reviews")
    if not isinstance(rows, list) or len(rows) != len(candidate_ids):
        raise ValueError("critic must review every candidate exactly once")
    observed: set[str] = set()
    normalized = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("critic review rows must be objects")
        candidate_id = str(row.get("candidate_id", "")).strip()
        if candidate_id not in candidate_ids or candidate_id in observed:
            raise ValueError("critic review candidate identity mismatch")
        observed.add(candidate_id)
        verdict = str(row.get("verdict", "")).strip()
        if verdict not in {"advance", "revise", "reject"}:
            raise ValueError(f"unsupported critic verdict {verdict!r}")
        risk_flags = row.get("risk_flags")
        if not isinstance(risk_flags, list):
            raise ValueError("critic risk_flags must be a list")
        normalized.append(
            {
                "candidate_id": candidate_id,
                "verdict": verdict,
                "risk_flags": [str(value).strip() for value in risk_flags if str(value).strip()],
                "critical_test": str(row.get("critical_test", "")).strip(),
                "revision": str(row.get("revision", "")).strip(),
            }
        )
    return {"reviews": normalized}


def _validate_synthesis(payload: dict[str, Any], candidate_ids: set[str]) -> dict[str, Any]:
    queue = payload.get("priority_queue")
    rounds = payload.get("rounds")
    if not isinstance(queue, list) or not queue:
        raise ValueError("synthesis priority_queue must be a non-empty list")
    normalized_queue = tuple(str(value).strip() for value in queue)
    if len(set(normalized_queue)) != len(normalized_queue):
        raise ValueError("synthesis priority_queue must not contain duplicates")
    if not set(normalized_queue).issubset(candidate_ids):
        raise ValueError("synthesis priority_queue contains an unknown candidate")
    if not isinstance(rounds, list) or not rounds:
        raise ValueError("synthesis rounds must be a non-empty list")
    normalized_rounds = []
    for row in rounds:
        if not isinstance(row, dict):
            raise ValueError("synthesis round must be an object")
        ids = tuple(str(value).strip() for value in row.get("candidate_ids", ()))
        if not ids or not set(ids).issubset(candidate_ids):
            raise ValueError("synthesis round contains invalid candidate IDs")
        normalized_rounds.append(
            {
                "round": int(row.get("round")),
                "candidate_ids": list(ids),
                "reason": str(row.get("reason", "")).strip(),
            }
        )
    return {
        "priority_queue": list(normalized_queue),
        "rounds": normalized_rounds,
        "stopping_rule": str(payload.get("stopping_rule", "")).strip(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("NVIDIA_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1"),
    )
    parser.add_argument("--max-tokens", type=int, default=1800)
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is required")

    client = NvidiaNimClient(api_key, endpoint=args.endpoint)
    available = client.list_models()
    models = client.select_models(available, count=3)
    role_models = {
        "generator": models[0],
        "critic": models[min(1, len(models) - 1)],
        "synthesizer": models[min(2, len(models) - 1)],
    }

    context_bundle = frozen_public_context(_public_context())
    context = context_bundle["context"]
    dispatch = ExternalDispatchPolicy()

    generator_payload, generator_call = client.chat_json(
        role="generator",
        model=role_models["generator"],
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_generator_prompt(context),
        max_tokens=args.max_tokens,
        temperature=0.35,
    )
    proposals = parse_proposals(
        generator_payload,
        allowed_payload_classes=dispatch.allowed_payload_classes,
        allowed_development_metrics=_ALLOWED_DEVELOPMENT_METRICS,
        min_candidates=5,
        max_candidates=5,
    )
    proposal_rows = [proposal.to_dict() for proposal in proposals]
    candidate_ids = {proposal.candidate_id for proposal in proposals}

    critic_payload, critic_call = client.chat_json(
        role="critic",
        model=role_models["critic"],
        system_prompt=_CRITIC_SYSTEM_PROMPT,
        user_prompt=_critic_prompt(context, proposal_rows),
        max_tokens=args.max_tokens,
        temperature=0.1,
    )
    reviews = _validate_reviews(critic_payload, candidate_ids)

    synthesis_payload, synthesis_call = client.chat_json(
        role="synthesizer",
        model=role_models["synthesizer"],
        system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
        user_prompt=_synthesis_prompt(context, proposal_rows, reviews),
        max_tokens=args.max_tokens,
        temperature=0.1,
    )
    synthesis = _validate_synthesis(synthesis_payload, candidate_ids)

    source_revision = os.environ.get("GITHUB_SHA", "local-unspecified").strip()
    result = {
        "schema_version": 1,
        "kind": "neuros_nim_research_tournament",
        "source_revision": source_revision,
        "provider": "nvidia_nim",
        "endpoint": client.endpoint,
        "model_identity_boundary": (
            "Model IDs and API responses are fingerprinted; hosted backend weight revision "
            "is not independently attested by this artifact."
        ),
        "available_model_catalog_sha256": canonical_sha256(list(available)),
        "selected_models": role_models,
        "public_context": context,
        "public_context_sha256": context_bundle["context_sha256"],
        "dispatch_policy": dispatch.to_dict(),
        "proposals": [
            {**proposal.to_dict(), "fingerprint": proposal.fingerprint} for proposal in proposals
        ],
        "critic": reviews,
        "synthesis": synthesis,
        "calls": [
            generator_call.to_dict(include_response=True),
            critic_call.to_dict(include_response=True),
            synthesis_call.to_dict(include_response=True),
        ],
        "scientific_boundary": (
            "NIM output is untrusted proposal material. It is not ExperimentEvidence and cannot "
            "promote a candidate. Real promotion requires a frozen ExperimentPacket, deterministic "
            "execution, attached evidence, and ResearchRegistry adjudication."
        ),
    }
    result["fingerprint"] = canonical_sha256(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"NIM_TOURNAMENT_SHA256={result['fingerprint']}")
    print("NIM_MODELS=" + ",".join(role_models.values()))
    print("NIM_PRIORITY_QUEUE=" + ",".join(synthesis["priority_queue"]))


if __name__ == "__main__":
    main()
