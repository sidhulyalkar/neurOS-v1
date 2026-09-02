from __future__ import annotations

import copy

import pytest
from neuros.research.algonauts_prospective import (
    AlgonautProspectiveEvaluation,
    AlgonautProspectivePlanBinding,
)
from neuros.research.algonauts_prospective_cli import (
    _verify_envelope,
    build_adjudication_envelope,
    build_frozen_plan_envelope,
)
from neuros.research.evidence import AdversarialCheck
from neuros.research.prospective import (
    ProspectiveGeometryCandidate,
    ProspectiveGeometryPlan,
    ProspectiveGeometryReveal,
    ProspectiveOutcome,
    evaluate_prospective_geometry_gain,
)


def _plan(*, suffix: str = "") -> ProspectiveGeometryPlan:
    return ProspectiveGeometryPlan(
        plan_id="algonaut-prospective" + suffix,
        candidates=tuple(
            ProspectiveGeometryCandidate(
                candidate_id=f"candidate-{index}",
                geometry_score=float(index),
                geometry_evidence_sha256=f"{index + 10:064x}",
            )
            for index in range(5)
        ),
        geometry_metric="inner_validation_feature_geometry_spearman",
        outcome_metric="validation_pearson_delta",
        evaluation_fingerprint="1" * 64,
        split_fingerprint="2" * 64,
        preprocessing_fingerprint="3" * 64,
        source_revision="a" * 40,
    )


def _evaluation(plan: ProspectiveGeometryPlan) -> AlgonautProspectiveEvaluation:
    reveal = ProspectiveGeometryReveal(
        plan_fingerprint=plan.fingerprint,
        outcomes=tuple(
            ProspectiveOutcome(
                candidate_id=f"candidate-{index}",
                validation_pearson_delta=float(index) / 100.0,
                outcome_evidence_sha256=f"{index + 30:064x}",
            )
            for index in range(5)
        ),
        source_revision=plan.source_revision,
    )
    checks = tuple(
        AdversarialCheck(
            check_id=check_id,
            status="pass",
            detail="qualified for test",
            metadata={},
        )
        for check_id in (
            "algonaut_temporal_controls",
            "algonaut_volume_isolation",
            "algonaut_candidate_roster_unchanged",
        )
    )
    return AlgonautProspectiveEvaluation(
        plan=plan,
        reveal=reveal,
        evaluation=evaluate_prospective_geometry_gain(plan, reveal),
        checks=checks,
        eligible_for_adjudication=True,
        algonaut_screen_fingerprint="4" * 64,
        algonaut_reveal_fingerprint="5" * 64,
    )


def test_frozen_plan_envelope_is_self_verifying() -> None:
    binding = AlgonautProspectivePlanBinding(
        plan=_plan(),
        algonaut_screen_fingerprint="4" * 64,
    )
    envelope = build_frozen_plan_envelope(binding)
    verified = _verify_envelope(
        envelope,
        expected_kind="neuros_algonaut_prospective_plan_binding",
    )
    assert verified["plan"]["fingerprint"] == binding.plan.fingerprint
    assert verified["algonaut_screen_fingerprint"] == "4" * 64


def test_frozen_plan_envelope_rejects_tampering() -> None:
    binding = AlgonautProspectivePlanBinding(
        plan=_plan(),
        algonaut_screen_fingerprint="4" * 64,
    )
    envelope = build_frozen_plan_envelope(binding)
    tampered = copy.deepcopy(envelope)
    tampered["plan"]["plan_id"] = "changed-after-freeze"
    with pytest.raises(ValueError, match="fingerprint mismatch"):
        _verify_envelope(
            tampered,
            expected_kind="neuros_algonaut_prospective_plan_binding",
        )


def test_adjudication_requires_exact_pre_reveal_plan() -> None:
    plan = _plan()
    frozen = build_frozen_plan_envelope(
        AlgonautProspectivePlanBinding(
            plan=plan,
            algonaut_screen_fingerprint="4" * 64,
        )
    )
    artifact = build_adjudication_envelope(frozen, _evaluation(plan))
    assert artifact["frozen_neuros_plan_fingerprint"] == plan.fingerprint
    assert artifact["result"]["eligible_for_adjudication"] is True

    other_plan = _plan(suffix="-other")
    with pytest.raises(ValueError, match="does not match the pre-reveal neurOS plan"):
        build_adjudication_envelope(frozen, _evaluation(other_plan))


def test_adjudication_requires_same_algonaut_screen() -> None:
    plan = _plan()
    frozen = build_frozen_plan_envelope(
        AlgonautProspectivePlanBinding(
            plan=plan,
            algonaut_screen_fingerprint="4" * 64,
        )
    )
    result = _evaluation(plan)
    object.__setattr__(result, "algonaut_screen_fingerprint", "6" * 64)
    with pytest.raises(ValueError, match="different Algonaut screen"):
        build_adjudication_envelope(frozen, result)
