"""Command-line evidence surfaces for neuros-mechint."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from typing import Any

from .adapters import integration_status
from .benchmarks import (
    read_correspondence_artifact,
    read_dose_response_artifact,
    read_evidence_pack_artifact,
    read_factorial_artifact,
    read_replication_artifact,
    run_circuit_faithfulness_benchmark,
    run_correspondence_ground_truth_benchmark,
    run_evidence_pack_generalization_benchmark,
    run_factorial_ground_truth_benchmark,
    run_ground_truth_benchmark,
    run_mechanism_emergence_benchmark,
    run_replication_ground_truth_benchmark,
    run_shared_computation_benchmark,
    run_v1_release_contract_benchmark,
)
from .core import EvidenceTier, list_method_cards, schema_catalog
from .integrations.evidence_recipes import external_evidence_recipe_dicts
from .release import default_v1_evidence_status


def _emit(payload: Any, *, as_json: bool) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return
    if isinstance(payload, list):
        for item in payload:
            print(item)
    else:
        print(payload)


def _methods(as_json: bool) -> int:
    cards = [card.to_dict() for card in list_method_cards()]
    if as_json:
        _emit(cards, as_json=True)
    else:
        for card in cards:
            print(f"{card['name']} [{card['maturity']}]")
            print("  establishes:", "; ".join(card["establishes"]))
            print("  limitations:", "; ".join(card["limitations"]))
            print("  controls:", "; ".join(card["required_controls"]))
    return 0


def _integrations(as_json: bool) -> int:
    statuses = [
        {"name": item.name, "available": item.available, "status": item.status, "role": item.role}
        for item in integration_status()
    ]
    if as_json:
        _emit(statuses, as_json=True)
    else:
        for item in statuses:
            installed = "installed" if item["available"] else "not installed"
            print(f"{item['name']}: {item['status']} ({installed}) - {item['role']}")
    return 0


def _evidence(as_json: bool) -> int:
    tiers = [{"level": int(tier), "label": tier.label, "name": tier.name} for tier in EvidenceTier]
    _emit(tiers if as_json else [f"{item['level']}. {item['label']}" for item in tiers], as_json=as_json)
    return 0


def _evidence_recipes(as_json: bool) -> int:
    recipes = external_evidence_recipe_dicts()
    if as_json:
        _emit(recipes, as_json=True)
    else:
        for recipe in recipes:
            print(f"{recipe['recipe_id']} [{recipe['ecosystem']}]")
            print(f"  model: {recipe['model_id']}")
            print(f"  discovery: {recipe['discovery_method']}")
            print(f"  target surface: {recipe['target_surface']}")
            print(f"  device: {recipe['recommended_device']}")
            print(f"  revision policy: {recipe['revision_policy']}")
    return 0


def _schemas(as_json: bool) -> int:
    payload = list(schema_catalog())
    _emit(payload, as_json=as_json)
    return 0


def _release_status(as_json: bool) -> int:
    payload = default_v1_evidence_status().to_dict()
    _emit(payload, as_json=as_json)
    return 0


def _ground_truth(as_json: bool, seed: int) -> int:
    report = run_ground_truth_benchmark(seed=seed)
    _emit(report, as_json=as_json)
    return 0 if report["localization"]["passed_separation"] else 1


def _shared_computation_ground_truth(as_json: bool) -> int:
    report = run_shared_computation_benchmark()
    _emit(report, as_json=as_json)
    return 0 if report["passed"] else 1


def _mechanism_emergence_ground_truth(as_json: bool) -> int:
    report = run_mechanism_emergence_benchmark()
    _emit(report, as_json=as_json)
    return 0 if report["passed"] else 1


def _circuit_faithfulness_ground_truth(as_json: bool, seed: int) -> int:
    report = run_circuit_faithfulness_benchmark(seed=seed)
    _emit(report, as_json=as_json)
    return 0 if report["passed"] else 1


def _evidence_pack_generalization_ground_truth(as_json: bool, seed: int) -> int:
    report = run_evidence_pack_generalization_benchmark(seed=seed)
    _emit(report, as_json=as_json)
    return 0 if report["passed"] else 1


def _factorial_ground_truth(as_json: bool) -> int:
    report = run_factorial_ground_truth_benchmark().to_dict()
    _emit(report, as_json=as_json)
    return 0 if report["passed"] else 1


def _correspondence_ground_truth(as_json: bool, seed: int) -> int:
    report = run_correspondence_ground_truth_benchmark(seed=seed).to_dict()
    _emit(report, as_json=as_json)
    return 0 if report["passed"] else 1


def _replication_ground_truth(as_json: bool, seed: int) -> int:
    report = run_replication_ground_truth_benchmark(seed=seed).to_dict()
    _emit(report, as_json=as_json)
    return 0 if report["passed"] else 1


def _v1_ground_truth(as_json: bool) -> int:
    report = run_v1_release_contract_benchmark().to_dict()
    _emit(report, as_json=as_json)
    return 0 if report["passed"] else 1


def _verify_evidence_artifact(path: str, as_json: bool) -> int:
    result = read_evidence_pack_artifact(path)
    _emit(
        {
            "artifact_valid": True,
            "pack_id": result["spec"]["pack_id"],
            "promotion": result["promotion"],
            "study_fingerprint": result["study_fingerprint"],
        },
        as_json=as_json,
    )
    return 0


def _verify_factorial_artifact(path: str, as_json: bool) -> int:
    result = read_factorial_artifact(path)
    _emit(
        {
            "artifact_valid": True,
            "estimable_contrast_ids": result["estimable_contrast_ids"],
            "nonestimable_contrast_ids": result["nonestimable_contrast_ids"],
            "study_fingerprint": result["study_fingerprint"],
            "study_id": result["spec"]["study_id"],
        },
        as_json=as_json,
    )
    return 0


def _verify_correspondence_artifact(path: str, as_json: bool) -> int:
    result = read_correspondence_artifact(path)
    _emit(
        {
            "artifact_valid": True,
            "promotion": result["promotion"],
            "source_space_id": result["spec"]["source_space"]["space_id"],
            "study_fingerprint": result["study_fingerprint"],
            "study_id": result["spec"]["study_id"],
            "target_space_id": result["spec"]["target_space"]["space_id"],
        },
        as_json=as_json,
    )
    return 0


def _verify_replication_artifact(path: str, as_json: bool) -> int:
    result = read_replication_artifact(path)
    _emit(
        {
            "artifact_valid": True,
            "claim_axis": result["spec"]["claim_axis"],
            "decision": result["decision"],
            "study_fingerprint": result["study_fingerprint"],
            "study_id": result["spec"]["study_id"],
        },
        as_json=as_json,
    )
    return 0


def _verify_dose_response_artifact(path: str, as_json: bool) -> int:
    result = read_dose_response_artifact(path)
    _emit(
        {
            "artifact_valid": True,
            "passed": result["passed"],
            "study_fingerprint": result["study_fingerprint"],
            "study_id": result["spec"]["study_id"],
        },
        as_json=as_json,
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuros-mechint",
        description="Mechanistic interpretability experiments and evidence audits.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command, help_text in (
        ("methods", "List method maturity cards and required controls."),
        ("integrations", "Show optional integration availability."),
        ("evidence", "Print the repository-wide evidence ladder."),
        ("evidence-recipes", "List maintained external-model evidence recipes."),
        ("schemas", "List frozen v1 artifact schema contracts."),
        ("release-status", "Report software readiness separately from empirical evidence closure."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--json", action="store_true", dest="as_json")

    for command, help_text in (
        ("shared-computation-ground-truth", "Verify shared-vs-architecture-specific causal maps."),
        ("mechanism-emergence-ground-truth", "Verify checkpoint mechanism-emergence inference."),
        ("factorial-ground-truth", "Verify matched architecture x tokenizer contrasts."),
        ("v1-ground-truth", "Verify v1 schema, reproduction, and evidence-closure contracts."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--json", action="store_true", dest="as_json")

    for command, help_text in (
        ("ground-truth", "Run the known-mechanism synthetic localization benchmark."),
        ("circuit-faithfulness-ground-truth", "Verify circuit necessity/sufficiency controls."),
        ("evidence-pack-generalization-ground-truth", "Verify held-out discovery rejection."),
        ("correspondence-ground-truth", "Reject predictive but noncausal correspondence."),
        ("replication-ground-truth", "Reject trial-level pseudoreplication."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--json", action="store_true", dest="as_json")

    for command, help_text in (
        ("verify-evidence-artifact", "Verify a held-out evidence-pack artifact."),
        ("verify-factorial-artifact", "Verify a factorial-study artifact."),
        ("verify-correspondence-artifact", "Verify a causal-correspondence artifact."),
        ("verify-replication-artifact", "Verify a hierarchical-replication artifact."),
        ("verify-dose-response-artifact", "Verify a v1 dose-response artifact."),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument("path")
        child.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "methods":
        return _methods(args.as_json)
    if args.command == "integrations":
        return _integrations(args.as_json)
    if args.command == "evidence":
        return _evidence(args.as_json)
    if args.command == "evidence-recipes":
        return _evidence_recipes(args.as_json)
    if args.command == "schemas":
        return _schemas(args.as_json)
    if args.command == "release-status":
        return _release_status(args.as_json)
    if args.command == "ground-truth":
        return _ground_truth(args.as_json, args.seed)
    if args.command == "shared-computation-ground-truth":
        return _shared_computation_ground_truth(args.as_json)
    if args.command == "mechanism-emergence-ground-truth":
        return _mechanism_emergence_ground_truth(args.as_json)
    if args.command == "circuit-faithfulness-ground-truth":
        return _circuit_faithfulness_ground_truth(args.as_json, args.seed)
    if args.command == "evidence-pack-generalization-ground-truth":
        return _evidence_pack_generalization_ground_truth(args.as_json, args.seed)
    if args.command == "factorial-ground-truth":
        return _factorial_ground_truth(args.as_json)
    if args.command == "correspondence-ground-truth":
        return _correspondence_ground_truth(args.as_json, args.seed)
    if args.command == "replication-ground-truth":
        return _replication_ground_truth(args.as_json, args.seed)
    if args.command == "v1-ground-truth":
        return _v1_ground_truth(args.as_json)
    if args.command == "verify-evidence-artifact":
        return _verify_evidence_artifact(args.path, args.as_json)
    if args.command == "verify-factorial-artifact":
        return _verify_factorial_artifact(args.path, args.as_json)
    if args.command == "verify-correspondence-artifact":
        return _verify_correspondence_artifact(args.path, args.as_json)
    if args.command == "verify-replication-artifact":
        return _verify_replication_artifact(args.path, args.as_json)
    if args.command == "verify-dose-response-artifact":
        return _verify_dose_response_artifact(args.path, args.as_json)
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
