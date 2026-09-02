from __future__ import annotations

import copy

import pytest
from neuros.research.algonauts_prospective import (
    _algonaut_canonical_sha256,
    ingest_algonaut_prospective_geometry,
)


def _seal(payload: dict) -> dict:
    values = copy.deepcopy(payload)
    values["fingerprint_sha256"] = _algonaut_canonical_sha256(values)
    return values


def _artifacts(*, temporal_controls: bool = True) -> tuple[dict, dict]:
    source_revision = "a" * 40
    evaluation_fingerprint = "1" * 64
    split_fingerprint = "2" * 64
    preprocessing_fingerprint = "3" * 64
    candidates = [
        {
            "candidate_id": candidate_id,
            "geometry_score": 0.1 * (index + 1),
            "geometry_evidence_sha256": f"{index + 10:064x}",
        }
        for index, candidate_id in enumerate(
            (
                "pca",
                "spatial_diffusion",
                "segment_diffusion_tw0p2",
                "segment_diffusion_tw0p35",
                "segment_diffusion_tw0p6",
            )
        )
    ]
    screen = _seal(
        {
            "kind": "algonaut_mario.neuros_prospective_geometry_screen",
            "schema_version": 2,
            "outcome_access": "none",
            "g2_validation_access": "none",
            "g2_id_test_access": "none",
            "g2_ood_access": "none",
            "cross_game_test_access": "none",
            "held_subject_access": "none",
            "volume_isolation": "verified_metadata",
            "temporal_control_status": "predeclared_not_revealed",
            "source_revision": source_revision,
            "g2_plan_authority_fingerprint_sha256": split_fingerprint,
            "preprocessing_fingerprint_sha256": preprocessing_fingerprint,
            "evaluation_authority": {"fingerprint_sha256": evaluation_fingerprint},
            "neuros_plan_input": {
                "plan_id": "algonaut-mario-ng2-repeat-00_fold-00-geometry-screen-v2",
                "candidates": candidates,
                "geometry_metric": "inner_validation_feature_geometry_spearman",
                "outcome_metric": "validation_pearson_delta",
                "evaluation_fingerprint": evaluation_fingerprint,
                "split_fingerprint": split_fingerprint,
                "preprocessing_fingerprint": preprocessing_fingerprint,
                "source_revision": source_revision,
            },
        }
    )
    outcomes = [
        {
            "candidate_id": row["candidate_id"],
            "validation_pearson_delta": 0.01 * index,
            "outcome_evidence_sha256": f"{index + 30:064x}",
        }
        for index, row in enumerate(candidates)
    ]
    adversarial = {
        "temporal_controls": temporal_controls,
        "volume_isolation": True,
        "candidate_roster_unchanged": True,
    }
    reveal = _seal(
        {
            "kind": "algonaut_mario.neuros_prospective_geometry_reveal",
            "schema_version": 2,
            "screen_fingerprint_sha256": screen["fingerprint_sha256"],
            "source_revision": source_revision,
            "g2_validation_access": "none",
            "g2_id_test_access": "none",
            "g2_ood_access": "none",
            "cross_game_test_access": "none",
            "held_subject_access": "none",
            "volume_isolation": "verified_metadata",
            "eligible_for_neuros_adjudication": temporal_controls,
            "neuros_reveal_projection": {
                "screen_fingerprint_sha256": screen["fingerprint_sha256"],
                "source_revision": source_revision,
                "outcomes": outcomes,
                "adversarial_checks": adversarial,
            },
        }
    )
    return screen, reveal


def _reseal(payload: dict) -> dict:
    values = copy.deepcopy(payload)
    values.pop("fingerprint_sha256", None)
    return _seal(values)


def test_ingest_builds_native_plan_reveal_and_evaluation() -> None:
    screen, reveal = _artifacts()
    result = ingest_algonaut_prospective_geometry(screen, reveal)
    assert result.eligible_for_adjudication is True
    assert result.reveal.plan_fingerprint == result.plan.fingerprint
    assert result.evaluation["metric_name"] == "prospective_geometry_gain_spearman"
    assert result.evaluation["metric_value"] == pytest.approx(1.0)
    assert result.evaluation["permutation_evidence"]["mode"] == "exact_label_permutation"
    assert all(check.status == "pass" for check in result.checks)
    assert result.algonaut_screen_fingerprint == screen["fingerprint_sha256"]
    assert result.algonaut_reveal_fingerprint == reveal["fingerprint_sha256"]


def test_ingest_rejects_tampered_algonaut_envelope() -> None:
    screen, reveal = _artifacts()
    screen["neuros_plan_input"]["candidates"][0]["geometry_score"] = 0.99
    with pytest.raises(ValueError, match="screen fingerprint mismatch"):
        ingest_algonaut_prospective_geometry(screen, reveal)


def test_ingest_rejects_reveal_bound_to_other_screen() -> None:
    screen, reveal = _artifacts()
    reveal["screen_fingerprint_sha256"] = "9" * 64
    reveal = _reseal(reveal)
    with pytest.raises(ValueError, match="exact frozen screen"):
        ingest_algonaut_prospective_geometry(screen, reveal)


def test_failed_temporal_adversary_preserves_primary_evaluation() -> None:
    screen, reveal = _artifacts(temporal_controls=False)
    result = ingest_algonaut_prospective_geometry(screen, reveal)
    assert result.eligible_for_adjudication is False
    assert result.evaluation["metric_value"] == pytest.approx(1.0)
    checks = {check.check_id: check for check in result.checks}
    assert checks["algonaut_temporal_controls"].status == "fail"
    assert checks["algonaut_volume_isolation"].status == "pass"
    assert checks["algonaut_candidate_roster_unchanged"].status == "pass"


def test_ingest_rejects_eligibility_that_disagrees_with_checks() -> None:
    screen, reveal = _artifacts(temporal_controls=False)
    reveal["eligible_for_neuros_adjudication"] = True
    reveal = _reseal(reveal)
    with pytest.raises(ValueError, match="eligibility disagrees"):
        ingest_algonaut_prospective_geometry(screen, reveal)


def test_ingest_rejects_missing_candidate_outcome() -> None:
    screen, reveal = _artifacts()
    reveal["neuros_reveal_projection"]["outcomes"].pop()
    reveal = _reseal(reveal)
    with pytest.raises(ValueError, match="candidate set mismatch"):
        ingest_algonaut_prospective_geometry(screen, reveal)


def test_ingest_rejects_holdout_access_declaration() -> None:
    screen, reveal = _artifacts()
    reveal["g2_ood_access"] = "read"
    reveal = _reseal(reveal)
    with pytest.raises(ValueError, match="forbidden access declaration g2_ood_access"):
        ingest_algonaut_prospective_geometry(screen, reveal)


def test_ingest_rejects_native_evaluator_drift() -> None:
    screen, reveal = _artifacts()
    screen["neuros_plan_input"]["evaluation_fingerprint"] = "8" * 64
    screen = _reseal(screen)
    reveal["screen_fingerprint_sha256"] = screen["fingerprint_sha256"]
    reveal["neuros_reveal_projection"]["screen_fingerprint_sha256"] = screen[
        "fingerprint_sha256"
    ]
    reveal = _reseal(reveal)
    with pytest.raises(ValueError, match="evaluator identity disagrees"):
        ingest_algonaut_prospective_geometry(screen, reveal)
