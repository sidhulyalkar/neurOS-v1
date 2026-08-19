import json

import pytest
import torch
from torch import nn

from neuros_mechint.adapters import PyTorchAdapter
from neuros_mechint.benchmarks import (
    CircuitCandidate,
    EvidenceExample,
    EvidencePackPolicy,
    EvidencePackSpec,
    EvidenceSplit,
    read_evidence_pack_artifact,
    run_adapter_evidence_pack,
    run_evidence_pack_generalization_benchmark,
    write_evidence_pack_artifact,
)
from neuros_mechint.core import EvidenceTier, OutputMetric


class StableRouteMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.route = nn.Linear(2, 1, bias=False)
        self.nuisance_a = nn.Linear(2, 1, bias=False)
        self.nuisance_b = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.route.weight.copy_(torch.tensor([[1.0, 0.0]]))
            self.nuisance_a.weight.zero_()
            self.nuisance_b.weight.zero_()

    def forward(self, value):
        return self.route(value) + self.nuisance_a(value) + self.nuisance_b(value)


def _metric():
    return OutputMetric(lambda output: output.mean(), name="mean_output")


def _spec(*, baselines=("zero",), pinned=True):
    return EvidencePackSpec(
        pack_id="stable-route-evidence",
        model_id="StableRouteMLP",
        model_revision="synthetic-v1" if pinned else None,
        dataset_id="synthetic:stable-route",
        dataset_revision="synthetic-v1" if pinned else None,
        metric_name="mean_output",
        target_universe=("route", "nuisance_a", "nuisance_b"),
        discovery_method="precomputed-known-route",
        intervention_baselines=baselines,
        random_trials=100,
        evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
    )


def _examples(validation_values=(3.0, 4.0)):
    return (
        EvidenceExample("discover-1", torch.tensor([[1.0, 0.0]]), EvidenceSplit.DISCOVERY),
        EvidenceExample("discover-2", torch.tensor([[2.0, 0.0]]), EvidenceSplit.DISCOVERY),
        EvidenceExample(
            "validate-1",
            torch.tensor([[validation_values[0], 0.0]]),
            EvidenceSplit.VALIDATION,
        ),
        EvidenceExample(
            "validate-2",
            torch.tensor([[validation_values[1], 0.0]]),
            EvidenceSplit.VALIDATION,
        ),
    )


def _known_candidate():
    return CircuitCandidate(name="known-route", targets=("route",), source="synthetic-ground-truth")


def test_known_shift_discovery_is_rejected_on_held_out_examples():
    report = run_evidence_pack_generalization_benchmark(seed=11)
    assert report["passed"] is True
    assert report["candidate"]["targets"] == ["route_a"]
    assert report["discovery_aggregate"]["pass_rate"] == pytest.approx(1.0)
    assert report["validation_aggregate"]["pass_rate"] == pytest.approx(0.0)
    assert report["promotion"]["passed"] is False
    assert set(report["discovery_saw_example_ids"]) == {"discover-1", "discover-2"}


def test_stable_route_promotes_on_held_out_zero_ablation():
    result = run_adapter_evidence_pack(
        spec=_spec(),
        adapter=PyTorchAdapter(StableRouteMLP()),
        metric=_metric(),
        examples=_examples(),
        candidate=_known_candidate(),
        pack_policy=EvidencePackPolicy(
            require_multiple_intervention_baselines=False,
            bootstrap_samples=200,
        ),
    )
    assert result.validation_aggregate.pass_rate == pytest.approx(1.0)
    assert result.validation_aggregate.median_joint_faithfulness == pytest.approx(1.0)
    assert result.promotion.passed is True
    assert result.promotion.validation_joint_advantage_vs_magnitude == pytest.approx(0.0)
    assert result.publication_ready is True


def test_mean_donor_is_fitted_from_discovery_only():
    result = run_adapter_evidence_pack(
        spec=_spec(baselines=("zero", "mean")),
        adapter=PyTorchAdapter(StableRouteMLP()),
        metric=_metric(),
        examples=_examples(validation_values=(100.0, 200.0)),
        candidate=_known_candidate(),
        pack_policy=EvidencePackPolicy(bootstrap_samples=100),
    )
    assert result.mean_ablation_references["route"] == pytest.approx(1.5)
    assert result.mean_ablation_references["nuisance_a"] == pytest.approx(0.0)
    assert result.mean_ablation_references["nuisance_b"] == pytest.approx(0.0)
    assert result.discovery_aggregate.n_invalid_cases == 1
    assert result.promotion.passed is False
    assert any("discovery contains" in reason for reason in result.promotion.reasons)


def test_discovery_callback_receives_only_discovery_examples():
    seen = []

    def discover(adapter, examples, targets):
        del adapter, targets
        seen.extend((item.example_id, item.split) for item in examples)
        return _known_candidate()

    run_adapter_evidence_pack(
        spec=_spec(),
        adapter=PyTorchAdapter(StableRouteMLP()),
        metric=_metric(),
        examples=_examples(),
        discover_candidate=discover,
        pack_policy=EvidencePackPolicy(
            require_multiple_intervention_baselines=False,
            bootstrap_samples=50,
        ),
    )
    assert seen == [
        ("discover-1", EvidenceSplit.DISCOVERY),
        ("discover-2", EvidenceSplit.DISCOVERY),
    ]


def test_duplicate_content_across_split_is_rejected():
    duplicate = torch.tensor([[1.0, 0.0]])
    examples = (
        EvidenceExample("discover", duplicate, EvidenceSplit.DISCOVERY),
        EvidenceExample("validate", duplicate.clone(), EvidenceSplit.VALIDATION),
    )
    with pytest.raises(ValueError, match="duplicate input content"):
        run_adapter_evidence_pack(
            spec=_spec(),
            adapter=PyTorchAdapter(StableRouteMLP()),
            metric=_metric(),
            examples=examples,
            candidate=_known_candidate(),
            pack_policy=EvidencePackPolicy(
                require_multiple_intervention_baselines=False,
                bootstrap_samples=50,
            ),
        )


def test_metric_identity_must_match_frozen_spec():
    wrong_metric = OutputMetric(lambda output: output.mean(), name="different_metric")
    with pytest.raises(ValueError, match="does not match evidence spec"):
        run_adapter_evidence_pack(
            spec=_spec(),
            adapter=PyTorchAdapter(StableRouteMLP()),
            metric=wrong_metric,
            examples=_examples(),
            candidate=_known_candidate(),
        )


def test_artifact_round_trip_and_tamper_detection(tmp_path):
    result = run_adapter_evidence_pack(
        spec=_spec(),
        adapter=PyTorchAdapter(StableRouteMLP()),
        metric=_metric(),
        examples=_examples(),
        candidate=_known_candidate(),
        pack_policy=EvidencePackPolicy(
            require_multiple_intervention_baselines=False,
            bootstrap_samples=50,
        ),
    )
    artifact = write_evidence_pack_artifact(result, tmp_path / "evidence.json")
    loaded = read_evidence_pack_artifact(artifact)
    assert loaded["study_fingerprint"] == result.study_fingerprint
    assert loaded["candidate"]["targets"] == ["route"]
    assert "inputs" not in json.dumps(loaded)

    payload = json.loads(artifact.read_text(encoding="utf-8"))
    payload["result"]["candidate"]["name"] = "tampered"
    artifact.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        read_evidence_pack_artifact(artifact)


def test_unpinned_revisions_are_reported_without_blocking_negative_artifacts():
    result = run_adapter_evidence_pack(
        spec=_spec(pinned=False),
        adapter=PyTorchAdapter(StableRouteMLP()),
        metric=_metric(),
        examples=_examples(),
        candidate=_known_candidate(),
        pack_policy=EvidencePackPolicy(
            require_multiple_intervention_baselines=False,
            bootstrap_samples=50,
        ),
    )
    assert result.publication_ready is False
    assert "model_revision is not pinned" in result.publication_issues
    assert "dataset_revision is not pinned" in result.publication_issues
