"""Run the external example through the production NSQ referee."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from neuros.foundation_models.longitudinal import (
    chronological_partition,
    make_nested_calibration_split,
)
from neuros.foundation_models.longitudinal_authority import LongitudinalCaseAuthority
from neuros.foundation_models.qualification import QualificationProtocolSpec
from neuros.foundation_models.qualification_runner import (
    DEFAULT_CLASSIFICATION_SCORECARD,
    QualificationExecutionContext,
    run_external_qualification_case,
)
from neuros.foundation_models.real_world import GroupedEvaluationData

from .method import ExternalFactory


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def synthetic_data() -> GroupedEvaluationData:
    labels = np.asarray(["left", "right"] * 24, dtype=str)
    sign = np.where(labels == "right", 1.0, -1.0).astype(np.float32)
    base = np.linspace(-0.15, 0.15, 16, dtype=np.float32).reshape(2, 8)
    X = sign[:, None, None] + base[None, :, :]
    return GroupedEvaluationData(
        dataset_id="synthetic-external-nsq-method-v1",
        X=X,
        y=labels,
        groups={
            "subject": np.asarray(["p1"] * len(labels), dtype=str),
            "session": np.repeat(np.asarray(["s1", "s2", "s3"], dtype=str), 16),
            "trial": np.asarray(
                [f"t{index:03d}" for index in range(len(labels))],
                dtype=str,
            ),
        },
    )


def build_authority(data: GroupedEvaluationData) -> LongitudinalCaseAuthority:
    partition = chronological_partition(
        data,
        split_unit="session",
        held_out_value="s3",
        order=("s1", "s2", "s3"),
    )
    split = make_nested_calibration_split(
        partition,
        evaluation_fraction=0.5,
        seed=11,
    )
    return LongitudinalCaseAuthority.from_split(
        split,
        case_id="p1:s3",
        history_policy="prior",
        observed_group_order=("s1", "s2", "s3"),
        case_metadata={"participant": "p1", "fixture": True},
    )


def run_demo() -> dict[str, Any]:
    data = synthetic_data()
    authority = build_authority(data)
    split = authority.restore(data)
    lineage_sha = _sha("synthetic-external-nsq-method-v1:dataset-lineage")
    protocol = QualificationProtocolSpec(
        protocol_id="synthetic-external-method-referee-v1",
        dataset_id=data.dataset_id,
        dataset_lineage_sha256=lineage_sha,
        task_id="synthetic-left-vs-right",
        independent_unit="participant",
        grouping_hierarchy=("participant", "session", "trial"),
        calibration_budgets_per_class=tuple(
            range(split.max_budget_per_class + 1)
        ),
        metric_scorecard_sha256=DEFAULT_CLASSIFICATION_SCORECARD.sha256,
        protocol_status="frozen",
        metadata={
            "synthetic_fixture": True,
            "numerical_result_interpretable": False,
        },
    )
    factory = ExternalFactory()
    result = run_external_qualification_case(
        data,
        authority,
        protocol,
        factory,
        execution_context=QualificationExecutionContext(
            observed_dataset_lineage_sha256=lineage_sha,
            preprocessing_authority_sha256s=(
                _sha("synthetic-external-nsq-method-v1:preprocessing"),
            ),
            calibration_authority_sha256s=(
                _sha("synthetic-external-nsq-method-v1:calibration-policy"),
            ),
        ),
    )
    return {
        "artifact_kind": "synthetic_external_method_nsq_referee_demo",
        "method_spec_sha256": factory.method_spec.sha256,
        "protocol_sha256": protocol.sha256,
        "case_authority_sha256": authority.authority_sha256,
        "case_result_sha256": result.sha256,
        "rows": [row.to_dict() for row in result.rows],
        "claim_boundary": {
            "synthetic_fixture": True,
            "production_referee_exercised": True,
            "fresh_decoder_per_budget": True,
            "learned_state_identity_exercised": True,
            "numerical_result_interpretable": False,
            "real_neural_efficacy_claim_permitted": False,
            "kumar2024_comparison_changed": False,
        },
    }


def main() -> None:
    print(json.dumps(run_demo(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
