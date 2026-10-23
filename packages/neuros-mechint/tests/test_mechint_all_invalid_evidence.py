"""All-invalid causal-normalization cases remain verifiable negative evidence.

No null (unmeasurable) effect may be mistaken for a measured zero effect.
"""

from __future__ import annotations

import json

import pytest
import torch
from torch import nn

from neuros_mechint.adapters import PyTorchAdapter
from neuros_mechint.benchmarks import (
    CircuitCandidate,
    EVIDENCE_PACK_SCHEMA,
    EVIDENCE_PACK_SCHEMA_V2,
    EvidenceExample,
    EvidencePackPolicy,
    EvidencePackSpec,
    EvidenceSplit,
    outcome_from_evidence_pack,
    read_evidence_pack_artifact,
    run_adapter_evidence_pack,
    write_evidence_pack_artifact,
)
from neuros_mechint.core import (
    OutputMetric,
    make_artifact_envelope,
    migrate_artifact_envelope,
    schema_catalog,
    validate_artifact_envelope,
)


class SplitGate(nn.Module):
    def __init__(self, mode: str):
        super().__init__()
        self.mode = mode
        self.route = nn.Identity()
        self.decoy = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        route = self.route(x)[..., :1]
        noise = self.decoy(x)[..., :1] * 0.0
        if self.mode == "flat":
            factor = torch.zeros_like(route)
        elif self.mode == "discovery":
            factor = (x[..., :1] < 3.0).to(route.dtype)
        elif self.mode == "validation":
            factor = (x[..., :1] >= 3.0).to(route.dtype)
        elif self.mode == "mixed":
            factor = (x[..., :1] < 3.5).to(route.dtype)
        else:
            raise ValueError("unknown test mode")
        return route * factor + noise


def _spec() -> EvidencePackSpec:
    return EvidencePackSpec(
        pack_id="nonestimable-negative-study",
        model_id="SplitGate",
        model_revision="synthetic-v1",
        dataset_id="synthetic:nonestimable",
        dataset_revision="synthetic-v1",
        metric_name="mean_output",
        target_universe=("route", "decoy"),
        discovery_method="preregistered-negative-fixture",
        intervention_baselines=("zero",),
        random_trials=4,
        seed=11,
    )


def _inputs() -> tuple[EvidenceExample, ...]:
    return (
        EvidenceExample("d1", torch.tensor([[1.0]]), EvidenceSplit.DISCOVERY),
        EvidenceExample("d2", torch.tensor([[2.0]]), EvidenceSplit.DISCOVERY),
        EvidenceExample("v1", torch.tensor([[3.0]]), EvidenceSplit.VALIDATION),
        EvidenceExample("v2", torch.tensor([[4.0]]), EvidenceSplit.VALIDATION),
    )


def _run(mode: str):
    return run_adapter_evidence_pack(
        spec=_spec(),
        adapter=PyTorchAdapter(SplitGate(mode).eval()),
        metric=OutputMetric(lambda x: x.mean(), name="mean_output"),
        examples=_inputs(),
        candidate=CircuitCandidate(
            name="preregistered-route", targets=("route",), source="negative-control"
        ),
        pack_policy=EvidencePackPolicy(
            min_validation_examples=2,
            require_multiple_intervention_baselines=False,
            bootstrap_samples=30,
        ),
    )


@pytest.mark.parametrize(
    ("mode", "expect_discovery", "expect_validation"),
    [
        ("flat", 0, 0),
        ("discovery", 2, 0),
        ("validation", 0, 2),
    ],
)
def test_fully_invalid_partitions_are_explicit_nonpromoted_artifacts(
    mode, expect_discovery, expect_validation, tmp_path
):
    result = _run(mode)
    assert result.discovery_aggregate.n_valid_cases == expect_discovery
    assert result.validation_aggregate.n_valid_cases == expect_validation
    assert not result.promotion.passed

    for aggregate, expected in (
        (result.discovery_aggregate, expect_discovery),
        (result.validation_aggregate, expect_validation),
    ):
        if expected == 0:
            payload = aggregate.to_dict()
            assert payload["estimable"] is False
            assert payload["nonestimable_reason"] == "no valid faithfulness normalizations"
            assert aggregate.valid_case_rate == 0.0
            assert aggregate.pass_rate is None
            assert aggregate.median_joint_faithfulness is None
            assert aggregate.joint_mean_ci95_low is None
            assert aggregate.joint_mean_ci95_high is None
        else:
            assert aggregate.estimable
            assert isinstance(aggregate.median_joint_faithfulness, float)
            assert "estimable" not in aggregate.to_dict()

    path = write_evidence_pack_artifact(result, tmp_path / (mode + ".json"))
    data = read_evidence_pack_artifact(path)
    assert data["promotion"]["passed"] is False
    assert data["schema_version"] == EVIDENCE_PACK_SCHEMA_V2
    assert data["spec"]["schema_version"] == EVIDENCE_PACK_SCHEMA_V2
    assert len(data["candidate_cases"]) == 4
    assert data["manifest"]["model_revision"] == "synthetic-v1"
    assert data["study_fingerprint"] == result.study_fingerprint
    assert all(
        item["invalid_reason"] is not None
        for item in data["candidate_cases"]
        if not item["valid"]
    )
    with pytest.raises(ValueError, match="nonestimable"):
        outcome_from_evidence_pack(result)


def test_mixed_valid_and_invalid_examples_preserve_measured_effects(tmp_path):
    result = _run("mixed")
    assert result.discovery_aggregate.n_valid_cases == 2
    assert result.validation_aggregate.n_valid_cases == 1
    assert result.validation_aggregate.n_invalid_cases == 1
    assert result.validation_aggregate.estimable
    assert result.validation_aggregate.pass_rate is not None
    assert not result.promotion.passed
    path = write_evidence_pack_artifact(result, tmp_path / "mixed.json")
    data = read_evidence_pack_artifact(path)
    assert data["schema_version"] == EVIDENCE_PACK_SCHEMA
    assert data["validation_aggregate"]["n_valid_cases"] == 1
    assert data["validation_aggregate"]["n_invalid_cases"] == 1
    assert "estimable" not in data["validation_aggregate"]


def test_tampering_with_invalid_case_reason_fails_artifact_integrity(tmp_path):
    result = _run("flat")
    artifact = write_evidence_pack_artifact(result, tmp_path / "negative.json")
    envelope = json.loads(artifact.read_text(encoding="utf-8"))
    assert envelope["result"]["candidate_cases"][0]["invalid_reason"]
    envelope["result"]["candidate_cases"][0]["invalid_reason"] = "fake negative"
    artifact.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_evidence_pack_artifact(artifact)


def test_no_baseline_faithfulness_is_not_a_numeric_zero(tmp_path):
    result = _run("flat")
    artifact = write_evidence_pack_artifact(result, tmp_path / "negative.json")
    payload = read_evidence_pack_artifact(artifact)
    assert payload["schema_version"] == EVIDENCE_PACK_SCHEMA_V2
    assert payload["validation_aggregate"]["mean_joint_faithfulness"] is None
    assert payload["validation_aggregate"]["pass_rate"] is None
    assert payload["promotion"]["validation_joint_median"] is None
    assert payload["promotion"]["joint_generalization_drop"] is None
    assert any("no estimable" in reason for reason in payload["promotion"]["reasons"])


def test_canonical_envelope_round_trip_supports_v2_without_changing_frozen_v1():
    all_invalid = _run("flat")
    valid_and_invalid = _run("mixed")
    for result, version in (
        (all_invalid, EVIDENCE_PACK_SCHEMA_V2),
        (valid_and_invalid, EVIDENCE_PACK_SCHEMA),
    ):
        envelope = make_artifact_envelope("evidence_pack", result.to_dict())
        assert envelope["contract"]["result_schema"] == version
        validated = validate_artifact_envelope(envelope, family="evidence_pack")
        assert validated["result"]["schema_version"] == version
        legacy = {
            "artifact_schema": envelope["artifact_schema"],
            "artifact_hash": envelope["artifact_hash"],
            "result": envelope["result"],
        }
        migrated = migrate_artifact_envelope(legacy, family="evidence_pack")
        assert migrated["contract"] == envelope["contract"]
    frozen_catalog = {row["family"]: row for row in schema_catalog()}
    assert frozen_catalog["evidence_pack"]["result_schema"] == EVIDENCE_PACK_SCHEMA


def test_unknown_result_schema_rejected_by_generic_envelope_reader():
    result = _run("flat").to_dict()
    result["schema_version"] = "neuros-mechint.evidence-pack.v3"
    with pytest.raises(ValueError, match="unsupported"):
        make_artifact_envelope("evidence_pack", result)
