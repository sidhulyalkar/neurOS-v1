"""Run the Algonauts NIM tournament with frozen reviewed aggregate evidence."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
from types import ModuleType
from typing import Any

from neuros.research._canonical import canonical_sha256
from neuros.research.nim_provider import QualifiedNvidiaNimClient
from neuros.research.reviewed_context import load_reviewed_aggregate_context
from neuros.research.semantics import enforce_independent_synthesis_stopping_policy

_ROOT = Path(__file__).resolve().parents[1]
_BASE_TOURNAMENT = Path(__file__).with_name("02_nim_algonauts_tournament.py")
_REVIEWED_EVIDENCE = _ROOT / "evidence" / "controlled_representation_program_v3.json"
_SEMANTIC_CONTRACT_VERSION = 2

_CLAIM_RELATION_CONTRACT = """
ADDITIONAL_TYPED_CLAIM_CONTRACT:
Every candidate MUST include two additional fields: claim_relation and control_description.
claim_relation MUST exactly equal the claim_relation registered for primary_metric in
metric_registry. Allowed relations are absolute, matched_control, temporal_null,
complementarity, stability, control_sweep, and prospective_prediction. For absolute metrics,
control_description MUST be an empty string and the statement/falsification test MUST NOT claim
superiority over a baseline/control. For every non-absolute relation, control_description MUST be
a concrete, non-empty description of the matched control, null, comparison, sweep, or frozen
prospective reveal protocol. A claim using words such as than, versus, vs, compared to, relative
to, matched-control, baseline, improvement over, gain over, or reduction versus MUST NOT use
absolute validation_pearson or validation_mse as its primary metric. Use validation_pearson_delta
or validation_mse_reduction for matched-control performance claims. Use matched_geometry_rsa_delta
for matched geometry comparisons, temporal_shift_drop for temporal-null claims, and
complementarity_score for complementarity. A claim that geometry prospectively predicts,
forecasts, screens, or anticipates later/subsequent/future validation gain MUST use
prospective_geometry_gain_spearman with claim_relation=prospective_prediction and MUST describe a
candidate set and geometry scores frozen before matched validation deltas are revealed. Descriptive
rsa_spearman alone cannot support a prospective prediction claim. These fields are deterministic
execution authority, not optional prose.
""".strip()


def _load_base_tournament() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "neuros_research_nim_algonauts_tournament",
        _BASE_TOURNAMENT,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the base NIM tournament module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _output_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output", required=True)
    args, _ = parser.parse_known_args()
    return Path(args.output)


def _write_failure_evidence(output: Path, reviewed: dict[str, Any], exc: Exception) -> None:
    client = QualifiedNvidiaNimClient.latest_instance
    if client is None:
        return
    payload = {
        "kind": "neuros_nim_tournament_failure_evidence",
        "schema_version": 1,
        "semantic_contract_version": _SEMANTIC_CONTRACT_VERSION,
        "source_revision": os.environ.get("GITHUB_SHA", "local-unspecified").strip(),
        "provider": "nvidia_nim",
        "reviewed_evidence_fingerprint": reviewed["review_fingerprint"],
        "provider_qualification": client.provider_qualification(),
        "calls": client.call_journal_payload(),
        "failure": {
            "type": type(exc).__name__,
            "message": str(exc),
        },
        "scientific_boundary": (
            "This artifact preserves untrusted proposer failures and transport evidence only. "
            "It is not ExperimentEvidence and cannot promote any candidate."
        ),
    }
    payload["fingerprint"] = canonical_sha256(payload)
    output.parent.mkdir(parents=True, exist_ok=True)
    path = output.with_name("tournament-failure.json")
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"NIM_FAILURE_EVIDENCE_SHA256={payload['fingerprint']}")


def _upgrade_semantic_contract(output: Path) -> None:
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["semantic_contract_version"] = _SEMANTIC_CONTRACT_VERSION
    payload.pop("fingerprint", None)
    payload["fingerprint"] = canonical_sha256(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _attach_provider_qualification(output: Path) -> None:
    client = QualifiedNvidiaNimClient.latest_instance
    if client is None:
        raise RuntimeError("qualified NVIDIA client instance was not retained")
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["provider_qualification"] = client.provider_qualification()
    payload.pop("fingerprint", None)
    payload["fingerprint"] = canonical_sha256(payload)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    # This validation occurs before base.main() constructs a client or performs model discovery.
    reviewed = load_reviewed_aggregate_context(_REVIEWED_EVIDENCE)
    base = _load_base_tournament()
    base_context = base._public_context
    base_generator_prompt = base._generator_prompt
    base_generator_repair_prompt = base._generator_repair_prompt
    base_validate_synthesis = base._validate_synthesis
    base.NvidiaNimClient = QualifiedNvidiaNimClient

    def evidence_informed_context():
        context = base_context()
        context["reviewed_prior_evidence"] = reviewed
        context["priority_questions"] = [
            "Can richer temporal objectives or fixed externally pretrained temporal representations "
            "add development value beyond a converged matched-capacity reconstruction control?",
            "Which geometry measurements prospectively predict later matched-control validation "
            "gain when the candidate set and geometry scores are frozen before outcome reveal?",
            "Which representation and readout families produce complementary development residuals "
            "after controlling optimization, feature dimension, and temporal alignment?",
            "What low-cost matched controls maximize information before any candidate advances to "
            "more expensive G2/G3/G4 evaluation?",
        ]
        return context

    def evidence_generator_prompt(context: dict[str, Any]) -> str:
        return base_generator_prompt(context) + "\n\n" + _CLAIM_RELATION_CONTRACT

    def evidence_generator_repair_prompt(
        context: dict[str, Any], previous_payload: dict[str, Any], validation_error: str
    ) -> str:
        return (
            base_generator_repair_prompt(context, previous_payload, validation_error)
            + "\n\n"
            + _CLAIM_RELATION_CONTRACT
        )

    def deterministic_validate_synthesis(
        payload: dict[str, Any], advanced_ids: set[str]
    ) -> dict[str, Any]:
        structurally_valid = base_validate_synthesis(payload, advanced_ids)
        return enforce_independent_synthesis_stopping_policy(structurally_valid)

    base._public_context = evidence_informed_context
    base._generator_prompt = evidence_generator_prompt
    base._generator_repair_prompt = evidence_generator_repair_prompt
    base._validate_synthesis = deterministic_validate_synthesis
    output = _output_path()
    try:
        base.main()
    except Exception as exc:
        _write_failure_evidence(output, reviewed, exc)
        raise
    _upgrade_semantic_contract(output)
    _attach_provider_qualification(output)


if __name__ == "__main__":
    main()
