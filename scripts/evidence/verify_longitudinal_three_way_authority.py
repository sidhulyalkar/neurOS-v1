#!/usr/bin/env python3
"""Emit deterministic evidence for the v2 three-way longitudinal authority."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

from neuros.foundation_models.longitudinal import chronological_partition
from neuros.foundation_models.longitudinal_three_way import make_three_way_calibration_split
from neuros.foundation_models.longitudinal_three_way_authority import (
    ThreeWayLongitudinalCaseAuthority,
)
from neuros.foundation_models.real_world import GroupedEvaluationData


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )
    return hashlib.sha256(raw).hexdigest()


def _fixture() -> GroupedEvaluationData:
    rng = np.random.default_rng(20260826)
    X = []
    y = []
    metadata = []
    for session in ("0", "1", "2"):
        for label in ("left", "right"):
            for trial in range(12):
                X.append(rng.normal(size=(4, 16)))
                y.append(label)
                metadata.append(
                    {
                        "subject": "synthetic-1",
                        "session": session,
                        "run": f"run-{trial // 6}",
                    }
                )
    return GroupedEvaluationData.from_moabb_result(
        (np.asarray(X), np.asarray(y), metadata),
        dataset_id="synthetic-three-way-authority-v1",
    )


def build_evidence() -> dict[str, Any]:
    data = _fixture()
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="2",
    )
    split = make_three_way_calibration_split(
        partition,
        qualification_fraction=0.25,
        final_assessment_fraction=0.25,
        seed=23,
    )
    authority = ThreeWayLongitudinalCaseAuthority.from_split(
        split,
        case_id="synthetic-1/session-2/three-way-v2",
        history_policy="prior",
        case_metadata={
            "purpose": "software-contract-evidence",
            "final_assessment_policy": "untouched until state and policy freeze",
        },
    )

    # Round-trip the serialized authority before emitting evidence. The public
    # record is therefore evidence of the replayable contract, not only the
    # in-memory constructor.
    serialized = authority.to_dict()
    restored_authority = ThreeWayLongitudinalCaseAuthority.from_dict(serialized)
    restored_split = restored_authority.restore(data)
    if restored_split.fingerprint != split.fingerprint:
        raise RuntimeError("three-way authority replay changed split identity")

    qualification_set = set(restored_authority.qualification_indices)
    final_set = set(restored_authority.final_assessment_indices)
    budgets = []
    for budget in (0, 1, 3, restored_authority.max_budget_per_class):
        calibration = restored_authority.calibration_indices(budget)
        calibration_set = set(calibration)
        if calibration_set & qualification_set:
            raise RuntimeError("calibration overlaps qualification")
        if calibration_set & final_set:
            raise RuntimeError("calibration overlaps final assessment")
        budgets.append(
            {
                "per_class": budget,
                "n_calibration_samples": len(calibration),
                "calibration_indices": list(calibration),
                "calibration_budget_sha256": restored_authority.calibration_budget_sha256(
                    budget
                ),
                "qualification_set_sha256": restored_authority.qualification_set_sha256,
                "final_assessment_set_sha256": restored_authority.final_assessment_set_sha256,
            }
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": "neuros-longitudinal-three-way-authority-evidence",
        "authority": restored_authority.to_dict(),
        "split_manifest": restored_split.manifest(),
        "calibration_budgets": budgets,
        "claim_boundary": {
            "chronological_source_history_enforced": True,
            "nested_balanced_calibration_enforced": True,
            "qualification_separate_from_calibration": True,
            "final_assessment_separate_from_calibration": True,
            "final_assessment_separate_from_state_selection": True,
            "processed_data_identity_enforced": True,
            "serialization_replay_verified": True,
            "real_dataset_qualified": False,
            "adaptation_efficacy_qualified": False,
            "calibration_reduction_qualified": False,
            "orion_superiority_qualified": False,
            "hardware_qualified": False,
            "closed_loop_qualified": False,
            "clinical_qualified": False,
        },
    }
    payload["evidence_sha256"] = _canonical_sha256(payload)
    return payload


def main() -> None:
    print(json.dumps(build_evidence(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
