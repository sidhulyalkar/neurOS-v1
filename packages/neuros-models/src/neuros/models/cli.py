"""Command-line discovery for neurOS task-specific decoders."""

from __future__ import annotations

import argparse
import importlib.util
import json

from neuros.models.catalog import get_decoder_card, list_decoder_cards


def _print_cards(cards: object, as_json: bool) -> None:
    values = [card.to_dict() for card in cards]
    if as_json:
        print(json.dumps(values, indent=2))
        return
    for card in cards:
        print(f"{card.id:20} {card.backend:12} mechint={card.mechint:10} {card.family}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect neurOS decoder capabilities")
    sub = parser.add_subparsers(dest="command", required=True)
    list_cmd = sub.add_parser("list", help="List curated decoder families")
    list_cmd.add_argument("--mechint-ready", action="store_true")
    list_cmd.add_argument("--json", action="store_true")
    show_cmd = sub.add_parser("show", help="Show one decoder card")
    show_cmd.add_argument("model_id")
    show_cmd.add_argument("--json", action="store_true")
    sub.add_parser("doctor", help="Check optional research dependencies")
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
    elif args.command == "doctor":
        for package in ("torch", "neuros_mechint", "sklearn"):
            print(f"{package}: {'available' if importlib.util.find_spec(package) else 'missing'}")


if __name__ == "__main__":
    main()
