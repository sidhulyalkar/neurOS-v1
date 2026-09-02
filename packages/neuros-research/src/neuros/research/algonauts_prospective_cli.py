"""Freeze and adjudicate Algonaut NG3 prospective artifacts under native neurOS authority."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._canonical import canonical_sha256
from .algonauts_prospective import (
    AlgonautProspectiveEvaluation,
    AlgonautProspectivePlanBinding,
    freeze_algonaut_prospective_plan,
    ingest_algonaut_prospective_geometry,
)
from .prospective import ProspectiveGeometryPlan

_PLAN_KIND = "neuros_algonaut_prospective_plan_binding"
_EVALUATION_KIND = "neuros_algonaut_prospective_evaluation"
_SCHEMA_VERSION = 1


def _read_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{source} must contain one JSON object")
    return payload


def _write_immutable_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    return destination


def _seal_envelope(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(payload)
    values.pop("fingerprint", None)
    values["fingerprint"] = canonical_sha256(values)
    return values


def _verify_envelope(payload: Mapping[str, Any], *, expected_kind: str) -> dict[str, Any]:
    values = dict(payload)
    fingerprint = str(values.pop("fingerprint", ""))
    if len(fingerprint) != 64 or any(ch not in "0123456789abcdef" for ch in fingerprint):
        raise ValueError("neurOS envelope fingerprint must be a lowercase SHA-256")
    if canonical_sha256(values) != fingerprint:
        raise ValueError("neurOS envelope fingerprint mismatch")
    if values.get("kind") != expected_kind or values.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unexpected neurOS Algonaut prospective envelope")
    return values


def build_frozen_plan_envelope(binding: AlgonautProspectivePlanBinding) -> dict[str, Any]:
    """Create the immutable artifact that must exist before outcome reveal."""

    payload = {
        "kind": _PLAN_KIND,
        "schema_version": _SCHEMA_VERSION,
        **binding.to_dict(),
        "scientific_boundary": (
            "This artifact freezes the native neurOS prospective plan from an Algonaut pre-reveal "
            "screen. It contains no later outcome and must exist before reveal/adjudication."
        ),
    }
    return _seal_envelope(payload)


def build_adjudication_envelope(
    frozen_plan_envelope: Mapping[str, Any],
    result: AlgonautProspectiveEvaluation,
) -> dict[str, Any]:
    """Bind an evaluation to the exact neurOS plan artifact frozen before reveal."""

    frozen = _verify_envelope(frozen_plan_envelope, expected_kind=_PLAN_KIND)
    plan_payload = frozen.get("plan")
    if not isinstance(plan_payload, Mapping):
        raise ValueError("frozen neurOS plan envelope is missing plan artifact")
    frozen_plan = ProspectiveGeometryPlan.from_artifact(dict(plan_payload))
    if frozen_plan.fingerprint != result.plan.fingerprint:
        raise ValueError("revealed evaluation does not match the pre-reveal neurOS plan")
    if frozen.get("algonaut_screen_fingerprint") != result.algonaut_screen_fingerprint:
        raise ValueError("revealed evaluation belongs to a different Algonaut screen")
    payload = {
        "kind": _EVALUATION_KIND,
        "schema_version": _SCHEMA_VERSION,
        "frozen_neuros_plan_fingerprint": frozen_plan.fingerprint,
        "frozen_plan_envelope_fingerprint": frozen_plan_envelope["fingerprint"],
        "result": result.to_dict(),
        "scientific_boundary": (
            "This is development-only prospective evidence. Eligibility reflects deterministic "
            "adversarial checks and does not authorize G2/G3/G4, biological, BCI, or clinical claims."
        ),
    }
    return _seal_envelope(payload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser(
        "freeze-plan",
        help="verify an Algonaut screen and write the canonical neurOS plan before reveal",
    )
    freeze.add_argument("screen")
    freeze.add_argument("output")

    adjudicate = subparsers.add_parser(
        "adjudicate",
        help="evaluate a reveal only against the exact neurOS plan frozen before outcome access",
    )
    adjudicate.add_argument("screen")
    adjudicate.add_argument("frozen_plan")
    adjudicate.add_argument("reveal")
    adjudicate.add_argument("output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "freeze-plan":
        screen = _read_json_object(args.screen)
        binding = freeze_algonaut_prospective_plan(screen)
        artifact = build_frozen_plan_envelope(binding)
        destination = _write_immutable_json(args.output, artifact)
        print(f"NEUROS_PROSPECTIVE_PLAN_SHA256={binding.plan.fingerprint}")
        print(f"NEUROS_PLAN_ENVELOPE_SHA256={artifact['fingerprint']}")
        print(f"NEUROS_PLAN_ARTIFACT={destination}")
        return 0

    screen = _read_json_object(args.screen)
    frozen_plan = _read_json_object(args.frozen_plan)
    reveal = _read_json_object(args.reveal)
    result = ingest_algonaut_prospective_geometry(screen, reveal)
    artifact = build_adjudication_envelope(frozen_plan, result)
    destination = _write_immutable_json(args.output, artifact)
    print(f"NEUROS_PROSPECTIVE_METRIC={result.evaluation['metric_value']}")
    print(f"NEUROS_ADJUDICATION_ELIGIBLE={str(result.eligible_for_adjudication).lower()}")
    print(f"NEUROS_EVALUATION_SHA256={artifact['fingerprint']}")
    print(f"NEUROS_EVALUATION_ARTIFACT={destination}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
