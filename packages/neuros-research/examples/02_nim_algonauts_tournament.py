"""Run a bounded, semantically typed NVIDIA NIM research tournament for Algonauts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from neuros.research import ExternalDispatchPolicy
from neuros.research._canonical import canonical_json, canonical_sha256
from neuros.research.nim import NvidiaNimClient, frozen_public_context
from neuros.research.semantics import (
    ALGORITHMIC_METRIC_REGISTRY,
    metric_registry_payload,
    parse_semantic_proposals,
)

_ALLOWED_DEVELOPMENT_METRICS = tuple(sorted(ALGORITHMIC_METRIC_REGISTRY))

_SYSTEM_PROMPT = """You are one research proposer inside neurOS. You do not control data,
splits, evaluation authority, promotion, or hidden-test access. Produce only the JSON object
requested by the user. Proposals must be falsifiable, development-only, and compatible with
the frozen information boundary. Never request raw participant data, participant identifiers,
hidden targets, private leaderboard feedback, credentials, or changes to evaluation authority.
Typed supports_if/rejects_if predicates are authoritative; prose may not contradict them."""

_CRITIC_SYSTEM_PROMPT = """You are the adversarial-science critic in neurOS. You can critique
semantically validated research proposals but cannot promote them or change frozen evaluation
authority. Return only the requested JSON. Reject or revise leakage, confounding, duplicated
ideas, weak controls, predicate/claim mismatches, metric misuse, or excessive compute."""

_SYNTHESIS_SYSTEM_PROMPT = """You are a research-program synthesizer in neurOS. Recommend a
development-only execution order for candidates that already passed deterministic semantic
validation and adversarial review. You cannot promote scientific claims, rewrite criteria,
or use hidden/OOD feedback. Return only the requested JSON."""


def _public_context() -> dict[str, Any]:
    dispatch = ExternalDispatchPolicy()
    return {
        "program": "neurOS x Algonaut-A-Mario",
        "scientific_goal": (
            "Prospectively test whether development-only neural-geometry evidence predicts "
            "which frozen representation families later generalize best, without using OOD "
            "truth as model-selection feedback."
        ),
        "research_targets": [
            "actions/state controls",
            "V-JEPA frozen features",
            "V-JEPA 2.1 denser temporal features",
            "train-only PCA and segment-aware diffusion neural geometry",
            "ridge/readout variants",
            "representation fusion and specialist selection",
        ],
        "proposal_family_enum": [
            "representation",
            "temporal_alignment",
            "neural_geometry",
            "readout",
            "fusion",
            "generalization",
        ],
        "frozen_rules": [
            "training and validation may be used for candidate development",
            "G2 unseen-level truth is opened only after a candidate is frozen",
            "G3 Mario-to-Shinobi truth is not model-selection feedback",
            "G4 held-subject truth is not model-selection feedback",
            "a proposer may not rewrite dataset, split, HRF, metric, or promotion authority",
            "failed and unavailable runs remain evidence",
            "all decision criteria must use the typed metric registry",
        ],
        "allowed_payload_classes": list(dispatch.allowed_payload_classes),
        "prohibited_payload_classes": list(dispatch.prohibited_payload_classes),
        "metric_registry": metric_registry_payload(),
        "priority_questions": [
            "Can neural-geometry alignment prospectively screen representation families?",
            "Which temporal choices add signal beyond feature dimension and ridge capacity?",
            "When do readouts add complementary errors rather than rediscovering a winner?",
            "Which low-cost controls maximize information gained per unit compute?",
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
                "development_metrics": ["validation_pearson_delta", "rsa_spearman"],
                "primary_metric": "validation_pearson_delta",
                "supports_if": [
                    {
                        "metric": "validation_pearson_delta",
                        "operator": ">=",
                        "threshold": 0.02,
                        "rationale": "predeclared development effect threshold",
                    }
                ],
                "rejects_if": [
                    {
                        "metric": "validation_pearson_delta",
                        "operator": "<=",
                        "threshold": 0.005,
                        "rationale": "predeclared practical-null threshold",
                    }
                ],
                "falsification_test": "plain-language restatement of rejects_if",
                "estimated_compute_tier": "low | medium | high",
                "expected_failure_modes": ["one concrete failure mode"],
            }
        ]
    }
    return (
        "Generate exactly 5 concise, diverse candidate experiments. At least two must be "
        "low-compute controls, one must test prospective neural-geometry screening, and one "
        "must test representation complementarity. Use ONLY metrics from metric_registry. "
        "For every metric, obey its registered direction. Every candidate must include a "
        "primary_metric plus non-overlapping supports_if and rejects_if predicates. The primary "
        "metric must appear in both predicate sets. Leave an indeterminate gap between support "
        "and rejection thresholds rather than making them overlap. For higher_is_better metrics, "
        "support uses >/>= and rejection uses </<=. For lower_is_better or neutrality metrics, "
        "support uses </<= and rejection uses >/>=. Use derived temporal_shift_drop and "
        "complementarity_score instead of ambiguous raw null/correlation wording. Do not use "
        "G2/G3/G4 outcomes or unseen-level truth as a development metric. Do not change any "
        "dataset/evaluation/split authority. The family field MUST be exactly one value from "
        "proposal_family_enum, never a research target label.\n\n"
        f"PUBLIC_CONTEXT={canonical_json(context)}\n\n"
        f"OUTPUT_SCHEMA_EXAMPLE={canonical_json(schema)}"
    )


def _repair_prompt(
    context: dict[str, Any],
    previous_payload: dict[str, Any],
    validation_error: str,
) -> str:
    return (
        "Your previous candidate JSON failed neurOS deterministic validation. Correct the JSON "
        "without weakening any rule. Preserve exactly five candidate IDs and preserve the scientific "
        "intent where possible. The family field MUST be one of proposal_family_enum exactly. "
        "All metrics and directional predicates must obey metric_registry. Do not explain the repair; "
        "return only the corrected JSON object.\n\n"
        f"VALIDATION_ERROR={validation_error}\n\n"
        f"PUBLIC_CONTEXT={canonical_json(context)}\n\n"
        f"PREVIOUS_PAYLOAD={canonical_json(previous_payload)}"
    )


def _critic_prompt(context: dict[str, Any], proposals: list[dict[str, Any]]) -> str:
    schema = {
        "reviews": [
            {
                "candidate_id": "id",
                "verdict": "advance | revise | reject",
                "risk_flags": ["specific scientific or execution risk"],
                "critical_test": "single most important pre-execution check",
                "revision": "specific revision or empty string",
            }
        ]
    }
    return (
        "Critique every candidate exactly once. The typed predicates already passed deterministic "
        "direction checks, but you must still compare each scientific statement and rationale "
        "against those predicates. Mark revise if the prose hypothesis and machine criteria test "
        "different claims. Mark reject for leakage or an unresolvable confound. An advance verdict "
        "must have an empty risk_flags list and empty revision. Prefer matched controls that isolate "
        "representation geometry, temporal density, and error complementarity. Verdicts remain "
        "planning advice, never scientific promotion.\n\n"
        f"PUBLIC_CONTEXT={canonical_json(context)}\n\n"
        f"SEMANTICALLY_VALIDATED_PROPOSALS={canonical_json(proposals)}\n\n"
        f"OUTPUT_SCHEMA_EXAMPLE={canonical_json(schema)}"
    )


def _synthesis_prompt(
    context: dict[str, Any],
    proposals: list[dict[str, Any]],
    reviews: dict[str, Any],
    advanced_ids: set[str],
) -> str:
    schema = {
        "priority_queue": ["advanced-candidate-id-1"],
        "rounds": [
            {
                "round": 1,
                "candidate_ids": ["advanced-candidate-id-1"],
                "reason": "why this ordering maximizes information per compute",
            }
        ],
        "stopping_rule": "development-only stopping or revision criterion",
    }
    return (
        "Create a maximum 2-round development-only execution queue using ONLY candidate IDs "
        "whose critic verdict is advance. Do not include revise/reject candidates. Prioritize "
        "low-cost controls, falsification strength, and experiments whose outcomes decide whether "
        "later work is worth running. Do not rewrite any typed criterion and do not claim any "
        "candidate is scientifically promoted.\n\n"
        f"PUBLIC_CONTEXT={canonical_json(context)}\n\n"
        f"ADVANCED_IDS={canonical_json(sorted(advanced_ids))}\n\n"
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
        normalized_flags = [str(value).strip() for value in risk_flags if str(value).strip()]
        revision = str(row.get("revision", "")).strip()
        critical_test = str(row.get("critical_test", "")).strip()
        if not critical_test:
            raise ValueError("critic critical_test must be non-empty")
        if verdict == "advance" and (normalized_flags or revision):
            raise ValueError("advance verdict cannot retain risk flags or a requested revision")
        if verdict == "revise" and not revision:
            raise ValueError("revise verdict must include a concrete revision")
        normalized.append(
            {
                "candidate_id": candidate_id,
                "verdict": verdict,
                "risk_flags": normalized_flags,
                "critical_test": critical_test,
                "revision": revision,
            }
        )
    return {"reviews": normalized}


def _validate_synthesis(payload: dict[str, Any], advanced_ids: set[str]) -> dict[str, Any]:
    if not advanced_ids:
        return {
            "priority_queue": [],
            "rounds": [],
            "stopping_rule": "No candidate passed adversarial review; revise before execution.",
        }
    queue = payload.get("priority_queue")
    rounds = payload.get("rounds")
    if not isinstance(queue, list) or not queue:
        raise ValueError("synthesis priority_queue must be a non-empty list")
    normalized_queue = tuple(str(value).strip() for value in queue)
    if len(set(normalized_queue)) != len(normalized_queue):
        raise ValueError("synthesis priority_queue must not contain duplicates")
    if not set(normalized_queue).issubset(advanced_ids):
        raise ValueError("synthesis priority_queue contains a non-advanced candidate")
    if not isinstance(rounds, list) or not rounds or len(rounds) > 2:
        raise ValueError("synthesis rounds must contain one or two rounds")
    normalized_rounds = []
    queued_in_rounds: set[str] = set()
    for row in rounds:
        if not isinstance(row, dict):
            raise ValueError("synthesis round must be an object")
        ids = tuple(str(value).strip() for value in row.get("candidate_ids", ()))
        if not ids or not set(ids).issubset(advanced_ids):
            raise ValueError("synthesis round contains invalid candidate IDs")
        if queued_in_rounds & set(ids):
            raise ValueError("a candidate may appear in only one synthesis round")
        queued_in_rounds.update(ids)
        reason = str(row.get("reason", "")).strip()
        if not reason:
            raise ValueError("synthesis round reason must be non-empty")
        normalized_rounds.append(
            {
                "round": int(row.get("round")),
                "candidate_ids": list(ids),
                "reason": reason,
            }
        )
    if set(normalized_queue) != queued_in_rounds:
        raise ValueError("priority_queue and synthesis rounds must contain the same candidates")
    stopping_rule = str(payload.get("stopping_rule", "")).strip()
    if not stopping_rule:
        raise ValueError("synthesis stopping_rule must be non-empty")
    return {
        "priority_queue": list(normalized_queue),
        "rounds": normalized_rounds,
        "stopping_rule": stopping_rule,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("NVIDIA_NIM_ENDPOINT", "https://integrate.api.nvidia.com/v1"),
    )
    parser.add_argument("--max-tokens", type=int, default=4000)
    args = parser.parse_args()

    api_key = os.environ.get("NVIDIA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is required")

    client = NvidiaNimClient(api_key, endpoint=args.endpoint)
    available, discovery_mode = client.discover_models()
    models = client.select_models(available, count=3)
    role_models = {
        "generator": models[0],
        "critic": models[min(1, len(models) - 1)],
        "synthesizer": models[min(2, len(models) - 1)],
    }

    context_bundle = frozen_public_context(_public_context())
    context = context_bundle["context"]
    dispatch = ExternalDispatchPolicy()
    calls = []

    generator_payload, generator_call = client.chat_json(
        role="generator",
        model=role_models["generator"],
        system_prompt=_SYSTEM_PROMPT,
        user_prompt=_generator_prompt(context),
        max_tokens=args.max_tokens,
        temperature=0.25,
    )
    calls.append(generator_call)

    proposals = None
    validation_error = ""
    for repair_index in range(3):
        try:
            proposals = parse_semantic_proposals(
                generator_payload,
                allowed_payload_classes=dispatch.allowed_payload_classes,
                allowed_development_metrics=_ALLOWED_DEVELOPMENT_METRICS,
                min_candidates=5,
                max_candidates=5,
            )
            break
        except (KeyError, TypeError, ValueError) as exc:
            validation_error = f"{type(exc).__name__}: {exc}"
            if repair_index == 2:
                raise
            generator_payload, repair_call = client.chat_json(
                role=f"generator_repair_{repair_index + 1}",
                model=role_models["generator"],
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=_repair_prompt(context, generator_payload, validation_error),
                max_tokens=args.max_tokens,
                temperature=0.0,
            )
            calls.append(repair_call)
    if proposals is None:
        raise RuntimeError(f"semantic proposal validation failed: {validation_error}")
    proposal_rows = [proposal.to_dict() for proposal in proposals]
    candidate_ids = {proposal.candidate_id for proposal in proposals}

    critic_payload, critic_call = client.chat_json(
        role="critic",
        model=role_models["critic"],
        system_prompt=_CRITIC_SYSTEM_PROMPT,
        user_prompt=_critic_prompt(context, proposal_rows),
        max_tokens=args.max_tokens,
        temperature=0.05,
    )
    calls.append(critic_call)
    reviews = _validate_reviews(critic_payload, candidate_ids)
    advanced_ids = {
        row["candidate_id"] for row in reviews["reviews"] if row["verdict"] == "advance"
    }

    if advanced_ids:
        synthesis_payload, synthesis_call = client.chat_json(
            role="synthesizer",
            model=role_models["synthesizer"],
            system_prompt=_SYNTHESIS_SYSTEM_PROMPT,
            user_prompt=_synthesis_prompt(context, proposal_rows, reviews, advanced_ids),
            max_tokens=args.max_tokens,
            temperature=0.05,
        )
        calls.append(synthesis_call)
        synthesis = _validate_synthesis(synthesis_payload, advanced_ids)
    else:
        synthesis = _validate_synthesis({}, advanced_ids)

    source_revision = os.environ.get("GITHUB_SHA", "local-unspecified").strip()
    result = {
        "schema_version": 2,
        "semantic_contract_version": 1,
        "kind": "neuros_nim_research_tournament",
        "source_revision": source_revision,
        "provider": "nvidia_nim",
        "endpoint": client.endpoint,
        "model_identity_boundary": (
            "Model IDs and API responses are fingerprinted; hosted backend weight revision "
            "is not independently attested by this artifact."
        ),
        "model_discovery_mode": discovery_mode,
        "available_model_catalog_sha256": canonical_sha256(list(available)),
        "selected_models": role_models,
        "metric_registry": metric_registry_payload(),
        "metric_registry_sha256": canonical_sha256(metric_registry_payload()),
        "public_context": context,
        "public_context_sha256": context_bundle["context_sha256"],
        "dispatch_policy": dispatch.to_dict(),
        "proposal_validation_repairs": sum(
            1 for call in calls if call.role.startswith("generator_repair_")
        ),
        "proposals": [
            {**proposal.to_dict(), "fingerprint": proposal.fingerprint} for proposal in proposals
        ],
        "critic": reviews,
        "synthesis": synthesis,
        "calls": [call.to_dict(include_response=True) for call in calls],
        "scientific_boundary": (
            "NIM output is untrusted proposal material. It is not ExperimentEvidence and cannot "
            "promote a candidate. Real execution requires a reviewed semantic proposal plus real "
            "dataset/split/preprocessing/runner bindings in an immutable ExperimentPacket."
        ),
    }
    result["fingerprint"] = canonical_sha256(result)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"NIM_TOURNAMENT_SHA256={result['fingerprint']}")
    print("NIM_MODELS=" + ",".join(role_models.values()))
    print(f"NIM_ADVANCED={len(advanced_ids)}")
    print("NIM_PRIORITY_QUEUE=" + ",".join(synthesis["priority_queue"]))


if __name__ == "__main__":
    main()
