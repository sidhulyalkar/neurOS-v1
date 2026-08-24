"""Command-line discovery tools for neural models and real-world evidence sources."""

from __future__ import annotations

import argparse
import json
from typing import Any

from .real_world import find_evidence_sources
from .registry import DEFAULT_REGISTRY


def _emit_json(value: Any) -> None:
    print(json.dumps(value, indent=2, sort_keys=True))


def _compact(values: list[str] | tuple[str, ...], limit: int = 4) -> str:
    values = list(values)
    if len(values) <= limit:
        return ",".join(values)
    return ",".join(values[:limit]) + f",+{len(values) - limit}"


def _print_cards(cards: tuple[Any, ...]) -> None:
    if not cards:
        print("No models matched the requested filters.")
        return
    rows = []
    for card in cards:
        status = DEFAULT_REGISTRY.availability(card.id)
        rows.append(
            (
                card.id,
                str(card.year),
                _compact([value.value for value in card.modalities]),
                card.access.value,
                "yes" if status.available else "no",
                card.name,
            )
        )
    header = ("ID", "YEAR", "MODALITY", "ACCESS", "RUN", "NAME")
    widths = [
        max(len(row[index]) for row in rows + [header])
        for index in range(len(header))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(header)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def _print_evidence_sources(sources: tuple[Any, ...]) -> None:
    if not sources:
        print("No evidence sources matched the requested filters.")
        return
    rows = [
        (
            source.id,
            source.ecosystem,
            source.modality,
            _compact(source.roles, limit=3),
            source.title,
        )
        for source in sources
    ]
    header = ("ID", "ECOSYSTEM", "MODALITY", "ROLES", "TITLE")
    widths = [
        max(len(row[index]) for row in rows + [header])
        for index in range(len(header))
    ]
    print("  ".join(value.ljust(widths[index]) for index, value in enumerate(header)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuros-foundation",
        description=(
            "Discover, compare, and inspect neural foundation models and curated "
            "real-world evidence sources."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List/filter model cards")
    list_parser.add_argument("--modality")
    list_parser.add_argument("--task")
    list_parser.add_argument("--access")
    list_parser.add_argument("--integration")
    list_parser.add_argument("--min-year", type=int)
    list_parser.add_argument("--tag", action="append", default=[])
    list_parser.add_argument("--runnable", action="store_true")
    list_parser.add_argument("--json", action="store_true")

    show_parser = subparsers.add_parser("show", help="Show one model card")
    show_parser.add_argument("model_id")
    show_parser.add_argument("--json", action="store_true")

    compare_parser = subparsers.add_parser("compare", help="Compare model metadata")
    compare_parser.add_argument("model_ids", nargs="+")
    compare_parser.add_argument("--field", action="append", default=[])
    compare_parser.add_argument("--json", action="store_true")

    doctor_parser = subparsers.add_parser("doctor", help="Check local execution adapters")
    doctor_parser.add_argument("--json", action="store_true")

    evidence_parser = subparsers.add_parser(
        "evidence",
        help="List curated public sources selected for real-world neurOS evidence",
    )
    evidence_parser.add_argument("--modality")
    evidence_parser.add_argument("--ecosystem")
    evidence_parser.add_argument("--role")
    evidence_parser.add_argument("--json", action="store_true")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "list":
        cards = DEFAULT_REGISTRY.filter(
            modality=args.modality,
            task=args.task,
            access=args.access,
            integration=args.integration,
            min_year=args.min_year,
            tags=args.tag,
            runnable=True if args.runnable else None,
        )
        if args.json:
            _emit_json([card.to_dict() for card in cards])
        else:
            _print_cards(cards)
        return 0

    if args.command == "show":
        card = DEFAULT_REGISTRY.get_card(args.model_id)
        payload = card.to_dict()
        payload["availability"] = DEFAULT_REGISTRY.availability(args.model_id).to_dict()
        if args.json:
            _emit_json(payload)
        else:
            for key, value in payload.items():
                if isinstance(value, list):
                    value = ", ".join(str(item) for item in value)
                elif isinstance(value, dict):
                    value = json.dumps(value, sort_keys=True)
                print(f"{key}: {value}")
        return 0

    if args.command == "compare":
        rows = DEFAULT_REGISTRY.compare(args.model_ids, fields=args.field or None)
        _emit_json(rows)
        return 0

    if args.command == "doctor":
        rows = [DEFAULT_REGISTRY.availability(card.id).to_dict() for card in DEFAULT_REGISTRY.cards()]
        rows = [
            row
            for row in rows
            if row["reason"] != "catalog entry has no neurOS execution adapter yet"
        ]
        if args.json:
            _emit_json(rows)
        else:
            for row in rows:
                mark = "OK" if row["available"] else "MISSING"
                print(f"[{mark}] {row['model_id']}: {row['reason']}")
        return 0

    if args.command == "evidence":
        sources = find_evidence_sources(
            modality=args.modality,
            ecosystem=args.ecosystem,
            role=args.role,
        )
        if args.json:
            _emit_json([source.to_dict() for source in sources])
        else:
            _print_evidence_sources(sources)
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())
