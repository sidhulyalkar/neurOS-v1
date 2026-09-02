"""Native neurOS ingestion for Algonaut NG3 prospective geometry artifacts.

Algonaut owns its competition-specific split, lineage, control campaign, and evidence files.
neurOS owns the canonical prospective plan/reveal identity and the final development-only
prospective statistic. This adapter verifies both artifact envelopes without importing the
Algonaut package or duplicating its scientific executor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ._canonical import freeze_json, require_sha256, thaw_json
from .evidence import AdversarialCheck
from .prospective import (
    ProspectiveGeometryCandidate,
    ProspectiveGeometryPlan,
    ProspectiveGeometryReveal,
    ProspectiveOutcome,
    evaluate_prospective_geometry_gain,
)


def _algonaut_canonical_sha256(payload: Any) -> str:
    """Reproduce Algonaut NG3's canonical JSON envelope identity exactly."""

    raw = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _verify_algonaut_artifact(
    payload: Mapping[str, Any],
    *,
    expected_kind: str,
    expected_schema: int,
    name: str,
) -> tuple[dict[str, Any], str]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{name} must be a mapping")
    values = dict(payload)
    fingerprint = require_sha256(
        str(values.get("fingerprint_sha256", "")),
        name=f"{name}.fingerprint_sha256",
    )
    unsigned = dict(values)
    unsigned.pop("fingerprint_sha256", None)
    if _algonaut_canonical_sha256(unsigned) != fingerprint:
        raise ValueError(f"{name} fingerprint mismatch")
    if unsigned.get("kind") != expected_kind:
        raise ValueError(f"unexpected {name} kind")
    if unsigned.get("schema_version") != expected_schema:
        raise ValueError(f"unsupported {name} schema")
    return unsigned, fingerprint


def _require_no_holdout_access(payload: Mapping[str, Any], *, name: str) -> None:
    for key in (
        "g2_validation_access",
        "g2_id_test_access",
        "g2_ood_access",
        "cross_game_test_access",
        "held_subject_access",
    ):
        if payload.get(key) != "none":
            raise ValueError(f"{name} contains forbidden access declaration {key}")


@dataclass(frozen=True, slots=True)
class AlgonautProspectiveEvaluation:
    """Verified native neurOS interpretation of one Algonaut NG3 screen/reveal pair."""

    plan: ProspectiveGeometryPlan
    reveal: ProspectiveGeometryReveal
    evaluation: Mapping[str, Any]
    checks: tuple[AdversarialCheck, ...]
    eligible_for_adjudication: bool
    algonaut_screen_fingerprint: str
    algonaut_reveal_fingerprint: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "evaluation",
            freeze_json(self.evaluation, path="algonaut_prospective.evaluation"),
        )
        check_ids = [check.check_id for check in self.checks]
        if len(set(check_ids)) != len(check_ids):
            raise ValueError("Algonaut prospective checks must have unique IDs")
        expected = all(check.status == "pass" for check in self.checks)
        if bool(self.eligible_for_adjudication) != expected:
            raise ValueError("Algonaut adjudication eligibility disagrees with adversarial checks")
        object.__setattr__(
            self,
            "algonaut_screen_fingerprint",
            require_sha256(
                self.algonaut_screen_fingerprint,
                name="algonaut_screen_fingerprint",
            ),
        )
        object.__setattr__(
            self,
            "algonaut_reveal_fingerprint",
            require_sha256(
                self.algonaut_reveal_fingerprint,
                name="algonaut_reveal_fingerprint",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan": self.plan.to_artifact(),
            "reveal": self.reveal.to_artifact(),
            "evaluation": thaw_json(self.evaluation),
            "checks": [check.to_dict() for check in self.checks],
            "eligible_for_adjudication": self.eligible_for_adjudication,
            "algonaut_screen_fingerprint": self.algonaut_screen_fingerprint,
            "algonaut_reveal_fingerprint": self.algonaut_reveal_fingerprint,
        }


def ingest_algonaut_prospective_geometry(
    screen: Mapping[str, Any],
    reveal: Mapping[str, Any],
) -> AlgonautProspectiveEvaluation:
    """Verify Algonaut NG3 artifacts, then evaluate them under native neurOS authority."""

    screen_unsigned, screen_fingerprint = _verify_algonaut_artifact(
        screen,
        expected_kind="algonaut_mario.neuros_prospective_geometry_screen",
        expected_schema=2,
        name="Algonaut prospective screen",
    )
    reveal_unsigned, reveal_fingerprint = _verify_algonaut_artifact(
        reveal,
        expected_kind="algonaut_mario.neuros_prospective_geometry_reveal",
        expected_schema=2,
        name="Algonaut prospective reveal",
    )
    _require_no_holdout_access(screen_unsigned, name="Algonaut prospective screen")
    _require_no_holdout_access(reveal_unsigned, name="Algonaut prospective reveal")
    if screen_unsigned.get("volume_isolation") != "verified_metadata":
        raise ValueError("Algonaut screen lacks verified physical-volume isolation")
    if reveal_unsigned.get("volume_isolation") != "verified_metadata":
        raise ValueError("Algonaut reveal lacks verified physical-volume isolation")
    if screen_unsigned.get("temporal_control_status") != "predeclared_not_revealed":
        raise ValueError("Algonaut screen does not prove a preregistered control boundary")
    if reveal_unsigned.get("screen_fingerprint_sha256") != screen_fingerprint:
        raise ValueError("Algonaut reveal does not reference the exact frozen screen")

    plan_input = screen_unsigned.get("neuros_plan_input")
    if not isinstance(plan_input, Mapping):
        raise ValueError("Algonaut screen is missing neuros_plan_input")
    candidate_rows = plan_input.get("candidates")
    if not isinstance(candidate_rows, list) or not all(
        isinstance(row, Mapping) for row in candidate_rows
    ):
        raise ValueError("Algonaut neuros_plan_input candidates must be a list of mappings")
    plan = ProspectiveGeometryPlan(
        plan_id=str(plan_input["plan_id"]),
        candidates=tuple(
            ProspectiveGeometryCandidate(
                candidate_id=str(row["candidate_id"]),
                geometry_score=float(row["geometry_score"]),
                geometry_evidence_sha256=str(row["geometry_evidence_sha256"]),
            )
            for row in candidate_rows
        ),
        geometry_metric=str(plan_input["geometry_metric"]),
        outcome_metric=str(plan_input["outcome_metric"]),
        evaluation_fingerprint=str(plan_input["evaluation_fingerprint"]),
        split_fingerprint=str(plan_input["split_fingerprint"]),
        preprocessing_fingerprint=str(plan_input["preprocessing_fingerprint"]),
        source_revision=str(plan_input["source_revision"]),
    )
    evaluation_authority = screen_unsigned.get("evaluation_authority")
    if not isinstance(evaluation_authority, Mapping):
        raise ValueError("Algonaut screen lacks evaluation authority")
    if plan.evaluation_fingerprint != evaluation_authority.get("fingerprint_sha256"):
        raise ValueError("native neurOS evaluator identity disagrees with Algonaut screen")
    if plan.split_fingerprint != screen_unsigned.get("g2_plan_authority_fingerprint_sha256"):
        raise ValueError("native neurOS split identity disagrees with Algonaut G2 authority")
    if plan.preprocessing_fingerprint != screen_unsigned.get(
        "preprocessing_fingerprint_sha256"
    ):
        raise ValueError("native neurOS preprocessing identity disagrees with Algonaut screen")
    if plan.source_revision != screen_unsigned.get("source_revision"):
        raise ValueError("native neurOS source revision disagrees with Algonaut screen")

    projection = reveal_unsigned.get("neuros_reveal_projection")
    if not isinstance(projection, Mapping):
        raise ValueError("Algonaut reveal is missing neuros_reveal_projection")
    if projection.get("screen_fingerprint_sha256") != screen_fingerprint:
        raise ValueError("Algonaut neurOS reveal projection references a different screen")
    if projection.get("source_revision") != plan.source_revision:
        raise ValueError("Algonaut reveal projection uses a different source revision")
    outcome_rows = projection.get("outcomes")
    if not isinstance(outcome_rows, list) or not all(
        isinstance(row, Mapping) for row in outcome_rows
    ):
        raise ValueError("Algonaut neurOS reveal outcomes must be a list of mappings")
    native_reveal = ProspectiveGeometryReveal(
        plan_fingerprint=plan.fingerprint,
        outcomes=tuple(
            ProspectiveOutcome(
                candidate_id=str(row["candidate_id"]),
                validation_pearson_delta=float(row["validation_pearson_delta"]),
                outcome_evidence_sha256=str(row["outcome_evidence_sha256"]),
            )
            for row in outcome_rows
        ),
        source_revision=str(projection["source_revision"]),
    )
    evaluation = evaluate_prospective_geometry_gain(plan, native_reveal)

    adversarial = projection.get("adversarial_checks")
    if not isinstance(adversarial, Mapping):
        raise ValueError("Algonaut reveal projection lacks adversarial checks")
    check_specs = (
        (
            "algonaut_temporal_controls",
            bool(adversarial.get("temporal_controls")),
            "Registered temporal specificity/alignment controls passed.",
        ),
        (
            "algonaut_volume_isolation",
            bool(adversarial.get("volume_isolation")),
            "Original physical-volume isolation was verified by Algonaut lineage authority.",
        ),
        (
            "algonaut_candidate_roster_unchanged",
            bool(adversarial.get("candidate_roster_unchanged")),
            "The primary prospective relation retained the complete preregistered candidate roster.",
        ),
    )
    checks = tuple(
        AdversarialCheck(
            check_id=check_id,
            status="pass" if passed else "fail",
            detail=detail,
            metadata={
                "algonaut_screen_fingerprint": screen_fingerprint,
                "algonaut_reveal_fingerprint": reveal_fingerprint,
            },
        )
        for check_id, passed, detail in check_specs
    )
    eligible = all(check.status == "pass" for check in checks)
    if bool(reveal_unsigned.get("eligible_for_neuros_adjudication")) != eligible:
        raise ValueError("Algonaut reveal eligibility disagrees with its adversarial projection")

    return AlgonautProspectiveEvaluation(
        plan=plan,
        reveal=native_reveal,
        evaluation=evaluation,
        checks=checks,
        eligible_for_adjudication=eligible,
        algonaut_screen_fingerprint=screen_fingerprint,
        algonaut_reveal_fingerprint=reveal_fingerprint,
    )
