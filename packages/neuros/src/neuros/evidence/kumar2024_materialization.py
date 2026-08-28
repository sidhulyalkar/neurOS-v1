"""Kumar2024 composition helpers for NSQ study materialization authority.

The generic authority types live in ``neuros-foundation``. This module is the
study-specific bridge that turns a processed participant shard and a frozen
longitudinal case into reviewer-facing observation-role evidence.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

import numpy as np

from neuros.foundation_models.longitudinal_authority import (
    LongitudinalCaseAuthority,
    processed_data_sha256,
)
from neuros.foundation_models.materialization_authority import (
    ObservationIdentityAuthority,
    ProcessedMaterializationShard,
    StudyMaterializationAuthority,
    observation_identities_from_grouped_data,
)
from neuros.foundation_models.qualification_runner import QualificationCaseResult
from neuros.foundation_models.real_world import GroupedEvaluationData


def _identity_sha256(schema: str, payload: Mapping[str, Any]) -> str:
    raw = json.dumps(
        {"schema": schema, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def qualification_observation_set_sha256(
    role: str,
    processed_sha256: str,
    indices: Sequence[int] | np.ndarray,
) -> str:
    """Recompute the public NSQ observation-set identity used by result rows."""

    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype == np.bool_:
        raise ValueError(f"{role} indices must be a one-dimensional integer sequence")
    integer = values.astype(np.int64)
    if not np.array_equal(values, integer):
        raise ValueError(f"{role} indices must be integers without coercion")
    return _identity_sha256(
        "neuros.qualification_observation_set.v1",
        {
            "role": str(role),
            "processed_data_sha256": str(processed_sha256),
            "indices": [int(value) for value in integer.tolist()],
        },
    )


def build_processed_subject_shard(
    data: GroupedEvaluationData,
    *,
    subject: int,
    preprocessing_authority_sha256: str,
) -> ProcessedMaterializationShard:
    """Build the immutable processed participant shard used by later cases."""

    if "subject" not in data.groups:
        raise ValueError("Kumar2024 processed shard requires subject group identity")
    observed = set(np.asarray(data.groups["subject"]).astype(str).tolist())
    expected = {str(int(subject))}
    if observed != expected:
        raise ValueError(
            f"processed shard subject identity mismatch: expected={expected}, observed={observed}"
        )
    return ProcessedMaterializationShard(
        shard_id=f"subject={int(subject)}",
        processed_data_sha256=processed_data_sha256(data),
        observation_identity=observation_identities_from_grouped_data(data),
        preprocessing_authority_sha256s=(preprocessing_authority_sha256,),
    )


def verify_processed_subject_shard(
    data: GroupedEvaluationData,
    shard: ProcessedMaterializationShard,
    *,
    subject: int,
) -> ObservationIdentityAuthority:
    """Fail closed if a second materialization differs from the frozen first pass."""

    expected_id = f"subject={int(subject)}"
    if shard.shard_id != expected_id:
        raise ValueError(
            f"processed shard id mismatch: expected={expected_id!r}, observed={shard.shard_id!r}"
        )
    actual_processed = processed_data_sha256(data)
    if actual_processed != shard.processed_data_sha256:
        raise ValueError(
            "second-pass processed neural array differs from frozen materialization shard"
        )
    identities = observation_identities_from_grouped_data(data)
    if identities.sha256 != shard.observation_identity.sha256:
        raise ValueError(
            "second-pass human observation identity differs from frozen materialization shard"
        )
    return identities


def _calibration_indices(
    authority: LongitudinalCaseAuthority,
    budget: int,
) -> np.ndarray:
    amount = int(budget)
    if amount < 0:
        raise ValueError("calibration budget must be non-negative")
    max_budget = min(len(values) for values in authority.calibration_order_by_class.values())
    if amount > max_budget:
        raise ValueError(
            f"budget {amount} exceeds balanced maximum {max_budget} for case {authority.case_id}"
        )
    if amount == 0:
        return np.asarray([], dtype=np.int64)
    selected = [
        np.asarray(values[:amount], dtype=np.int64)
        for _, values in sorted(authority.calibration_order_by_class.items())
    ]
    return np.sort(np.concatenate(selected))


def _fit_indices(
    authority: LongitudinalCaseAuthority,
    budget: int,
) -> np.ndarray:
    calibration = _calibration_indices(authority, budget)
    source = np.asarray(authority.source_train_indices, dtype=np.int64)
    if len(calibration) == 0:
        return source.copy()
    return np.sort(np.concatenate([source, calibration]))


def _role_payload(
    shard: ProcessedMaterializationShard,
    *,
    role: str,
    indices: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    role_authority = shard.observation_identity.role(role, indices)
    payload = role_authority.to_dict()
    payload["authority_sha256"] = role_authority.sha256
    payload["nsq_observation_set_sha256"] = qualification_observation_set_sha256(
        role,
        shard.processed_data_sha256,
        role_authority.row_indices,
    )
    return payload


def _validate_row_role_hashes(
    row: Any,
    roles: Mapping[str, Mapping[str, Any]],
) -> None:
    expected = {
        "supervised_source_history": row.source_train_indices_sha256,
        "labeled_target_calibration": row.labeled_target_indices_sha256,
        "supervised_fit": row.fit_indices_sha256,
        "untouched_final_assessment": row.evaluation_indices_sha256,
    }
    for role, row_sha in expected.items():
        observed = roles[role]["nsq_observation_set_sha256"]
        if observed != row_sha:
            raise RuntimeError(
                f"human observation role {role!r} does not reconcile with NSQ row identity"
            )


def _internal_state_roles(
    row: Any,
    shard: ProcessedMaterializationShard,
    fit_indices: np.ndarray,
) -> dict[str, Any]:
    state = row.qualification_model_state
    if state is None:
        return {}
    metadata = state.learned_state.metadata
    raw_validation = metadata.get("validation_relative_indices")
    if raw_validation is None:
        return {}
    relative = np.asarray(tuple(raw_validation), dtype=np.int64)
    if relative.ndim != 1 or len(set(relative.tolist())) != len(relative):
        raise RuntimeError("learned-state validation membership is malformed")
    if np.any(relative < 0) or np.any(relative >= len(fit_indices)):
        raise RuntimeError("learned-state validation membership escapes authorized fit set")
    validation_indices = fit_indices[relative]
    validation_set = set(int(value) for value in relative.tolist())
    training_relative = np.asarray(
        [index for index in range(len(fit_indices)) if index not in validation_set],
        dtype=np.int64,
    )
    training_indices = fit_indices[training_relative]
    return {
        "internal_model_validation": _role_payload(
            shard,
            role="internal_model_validation",
            indices=validation_indices,
        ),
        "internal_model_training": _role_payload(
            shard,
            role="internal_model_training",
            indices=training_indices,
        ),
    }


def build_case_result_observation_roles(
    *,
    authority: LongitudinalCaseAuthority,
    shard: ProcessedMaterializationShard,
    result: QualificationCaseResult,
) -> list[dict[str, Any]]:
    """Render human-auditable observation membership for every NSQ result row."""

    if authority.processed_data_sha256 != shard.processed_data_sha256:
        raise ValueError("case authority and materialization shard processed hashes differ")
    if result.case_authority_sha256 != authority.authority_sha256:
        raise ValueError("case result does not belong to supplied longitudinal authority")

    source_indices = np.asarray(authority.source_train_indices, dtype=np.int64)
    evaluation_indices = np.asarray(authority.evaluation_indices, dtype=np.int64)
    rendered: list[dict[str, Any]] = []
    for row in result.rows:
        calibration = _calibration_indices(authority, row.calibration_per_class)
        fit = _fit_indices(authority, row.calibration_per_class)
        roles: dict[str, Any] = {
            "supervised_source_history": _role_payload(
                shard,
                role="supervised_source_history",
                indices=source_indices,
            ),
            "labeled_target_calibration": _role_payload(
                shard,
                role="labeled_target_calibration",
                indices=calibration,
            ),
            "supervised_fit": _role_payload(
                shard,
                role="supervised_fit",
                indices=fit,
            ),
            "untouched_final_assessment": _role_payload(
                shard,
                role="untouched_final_assessment",
                indices=evaluation_indices,
            ),
        }
        _validate_row_role_hashes(row, roles)
        internal = _internal_state_roles(row, shard, fit)
        if internal:
            final_set = set(int(value) for value in evaluation_indices.tolist())
            for name, payload in internal.items():
                if final_set.intersection(payload["row_indices"]):
                    raise RuntimeError(
                        f"{name} overlaps untouched final assessment for case {authority.case_id}"
                    )
            roles.update(internal)

        rendered.append(
            {
                "schema_version": 1,
                "case_id": authority.case_id,
                "method_id": row.method_id,
                "calibration_per_class": row.calibration_per_class,
                "qualification_result_row_sha256": row.sha256,
                "processed_shard_id": shard.shard_id,
                "processed_shard_sha256": shard.sha256,
                "roles": roles,
            }
        )
    return rendered


def materialization_manifest(
    study: StudyMaterializationAuthority,
    *,
    raw_selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Serializable top-level authority payload stored in the study bundle."""

    return {
        "schema_version": 1,
        "study_materialization_sha256": study.sha256,
        "authority": study.to_dict(),
        "raw_selection": dict(raw_selection),
    }


__all__ = [
    "build_case_result_observation_roles",
    "build_processed_subject_shard",
    "materialization_manifest",
    "qualification_observation_set_sha256",
    "verify_processed_subject_shard",
]
