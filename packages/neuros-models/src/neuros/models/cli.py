"""Command-line discovery and artifact inspection for neurOS decoders."""

from __future__ import annotations

import argparse
import importlib.util
import json

from neuros.models.artifact import verify_model_artifact
from neuros.models.artifact_store import ModelArtifactStore
from neuros.models.catalog import get_decoder_card, list_decoder_cards


def _print_cards(cards: object, as_json: bool) -> None:
    values = [card.to_dict() for card in cards]
    if as_json:
        print(json.dumps(values, indent=2))
        return
    for card in cards:
        print(f"{card.id:20} {card.backend:12} mechint={card.mechint:10} {card.family}")


def _print_artifact(manifest: object, as_json: bool) -> None:
    payload = manifest.to_dict()
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    for key in (
        "artifact_id",
        "artifact_sha256",
        "manifest_sha256",
        "weights_sha256",
        "factory_id",
        "model_type",
        "backend",
        "backend_version",
        "interpretability_manifest_sha256",
        "git_sha",
    ):
        print(f"{key}: {payload[key]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect neurOS decoder capabilities")
    sub = parser.add_subparsers(dest="command", required=True)

    list_cmd = sub.add_parser("list", help="List curated decoder families")
    list_cmd.add_argument("--mechint-ready", action="store_true")
    list_cmd.add_argument("--json", action="store_true")

    show_cmd = sub.add_parser("show", help="Show one decoder card")
    show_cmd.add_argument("model_id")
    show_cmd.add_argument("--json", action="store_true")

    artifact_cmd = sub.add_parser("artifact", help="Inspect promoted Model Artifact v1 objects")
    artifact_sub = artifact_cmd.add_subparsers(dest="artifact_command", required=True)
    verify_cmd = artifact_sub.add_parser(
        "verify", help="Verify artifact hashes and print the canonical manifest"
    )
    verify_cmd.add_argument("path")
    verify_cmd.add_argument("--json", action="store_true")
    resolve_cmd = artifact_sub.add_parser(
        "resolve", help="Resolve a content SHA or named ref in a ModelArtifactStore"
    )
    resolve_cmd.add_argument("store")
    resolve_cmd.add_argument("ref_or_sha256")
    resolve_cmd.add_argument("--json", action="store_true")

    sub.add_parser("doctor", help="Check optional research/deployment dependencies")
    args = parser.parse_args()

    if args.command == "list":
        _print_cards(list_decoder_cards(mechint_ready=args.mechint_ready), args.json)
    elif args.command == "show":
        card = get_decoder_card(args.model_id)
        if args.json:
            print(json.dumps(card.to_dict(), indent=2))
        else:
            for key, value in card.to_dict().items():
                print(f"{key}: {value}")
    elif args.command == "artifact":
        if args.artifact_command == "verify":
            _print_artifact(verify_model_artifact(args.path), args.json)
        elif args.artifact_command == "resolve":
            _path, manifest = ModelArtifactStore(args.store).resolve(args.ref_or_sha256)
            _print_artifact(manifest, args.json)
    elif args.command == "doctor":
        for package in ("torch", "safetensors", "neuros_mechint", "sklearn"):
            print(f"{package}: {'available' if importlib.util.find_spec(package) else 'missing'}")


if __name__ == "__main__":
    main()
