from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from neuros_mechint.adapters import PyTorchAdapter
from neuros_mechint.benchmarks.correspondence import (
    CorrespondenceKind,
    CorrespondenceSplit,
    FeatureCorrespondencePolicy,
    FeatureCorrespondenceSpec,
    FeatureSpaceIdentity,
)
from neuros_mechint.benchmarks.factorial import (
    FactorialCellOutcome,
    FactorialCellSpec,
    FactorialContrastKind,
    FactorialContrastSpec,
    FactorialMechanismSpec,
    MatchedCovariate,
    analyze_factorial_mechanisms,
)
from neuros_mechint.core.metrics import OutputMetric
from neuros_mechint.integrations.correspondence import (
    AdapterFeatureSpaceView,
    AdapterPairedExampleSpec,
    TensorFeatureProjector,
    factorial_origin_from_report,
    run_adapter_feature_correspondence_study,
)


class _FeatureModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.hidden(inputs)
        return hidden[..., :1]


def _identity(space_id: str, model_id: str, architecture: str) -> FeatureSpaceIdentity:
    return FeatureSpaceIdentity(
        space_id=space_id,
        model_id=model_id,
        model_revision=f"{model_id}@sha256:123",
        representation_id="hidden",
        feature_names=("signal", "n1", "n2", "n3"),
        architecture=architecture,
        tokenizer_id="event",
        tokenizer_revision="event@v1",
        dataset_id="paired-dataset",
        dataset_revision="dataset@sha256:456",
        session_id="session-1",
        checkpoint="step-10",
        feature_semantics={"signal": "task signal"},
    )


def test_model_adapter_correspondence_executes_real_paired_interventions():
    source_identity = _identity("source", "source-model", "transformer")
    target_identity = _identity("target", "target-model", "ssm")
    source_inputs = {}
    target_inputs = {}
    example_specs = []
    values = (0.35, 0.55, 0.9, 1.4, 2.0, 3.2, 0.45, 0.7, 1.1, 1.8, 2.7, 4.1)
    for index, signal in enumerate(values):
        source_inputs[f"e{index}"] = torch.tensor(
            [signal, 0.2 + index, -0.1 * index, 1.0], dtype=torch.float32
        )
        target_inputs[f"e{index}"] = torch.tensor(
            [2.0 * signal + 0.5, -index, 0.5 * index, -1.0], dtype=torch.float32
        )
        split = CorrespondenceSplit.DISCOVERY if index < 6 else CorrespondenceSplit.VALIDATION
        example_specs.append(
            AdapterPairedExampleSpec(
                example_id=f"e{index}",
                semantic_trial_id=f"trial-{index}",
                split=split,
                partition_id="discovery" if index < 6 else "validation",
            )
        )

    metric = OutputMetric(lambda output: output.mean(), name="signal")
    source_model = _FeatureModel()
    target_model = _FeatureModel()
    source_view = AdapterFeatureSpaceView(
        identity=source_identity,
        adapter=PyTorchAdapter(source_model),
        path="hidden",
        metric=metric,
        inputs=source_inputs,
        projector=TensorFeatureProjector(feature_axis=-1),
    )
    target_view = AdapterFeatureSpaceView(
        identity=target_identity,
        adapter=PyTorchAdapter(target_model),
        path="hidden",
        metric=metric,
        inputs=target_inputs,
        projector=TensorFeatureProjector(feature_axis=-1),
    )
    spec = FeatureCorrespondenceSpec(
        study_id="adapter-correspondence",
        source_space=source_identity,
        target_space=target_identity,
        source_features=("signal",),
        target_features=("signal",),
        kind=CorrespondenceKind.ONE_TO_ONE,
        discovery_partition_id="discovery",
        validation_partition_id="validation",
        declared_context_differences=("model_id", "model_revision", "architecture"),
        random_controls=3,
        seed=3,
        policy=FeatureCorrespondencePolicy(
            min_discovery_examples=6,
            min_validation_examples=6,
            min_validation_predictive_r2=0.99,
            min_median_causal_recovery=0.95,
            min_source_effect=0.1,
            min_target_effect=0.1,
            min_random_percentile=0.9,
            min_shuffled_margin=-1.0,
            min_random_margin=0.5,
            max_discovery_validation_r2_drop=0.05,
        ),
    )
    result = run_adapter_feature_correspondence_study(
        spec,
        source=source_view,
        target=target_view,
        examples=example_specs,
    )
    assert result.promotion.passed
    assert result.validation_metrics.predictive_r2 > 0.999
    assert result.validation_metrics.median_causal_recovery > 0.999
    assert result.validation_metrics.random_control_percentile == 1.0
    assert np.isclose(result.candidate.mapping_matrix[0][0], 2.0, atol=1e-4)
    assert np.isclose(result.candidate.intercept[0], 0.5, atol=1e-4)


def _factorial_cell(cell_id: str, architecture: str, tokenizer: str) -> FactorialCellSpec:
    return FactorialCellSpec(
        cell_id=cell_id,
        architecture=architecture,
        tokenizer_id=tokenizer,
        model_id=f"{architecture}-model",
        model_revision=f"{architecture}@sha256:abc",
        tokenizer_revision=f"{tokenizer}@sha256:def",
        dataset_id="dataset",
        dataset_revision="dataset@sha256:123",
        session_id="session",
        metric_name="score",
        discovery_method="fixed",
        discovery_partition_id="discovery",
        validation_partition_id="validation",
        training_seed=0,
        checkpoint="step-10",
        checkpoint_maturity=1.0,
        target_universe=("route",),
        covariates={"token_budget": 128},
    )


def _factorial_outcome(value: float) -> FactorialCellOutcome:
    return FactorialCellOutcome(
        task_metric=0.8,
        candidate_size=1,
        validation_sufficiency=value,
        validation_necessity=value,
        validation_joint_faithfulness=value,
        validation_joint_random_percentile=1.0,
        discovery_to_validation_drop=0.0,
        intervention_baseline_sensitivity=0.0,
        promotion_passed=True,
        source_study_fingerprint=f"study-{value}",
        source_run_hash=f"run-{value}",
        evidence_protocol_fingerprint="protocol",
    )


def test_factorial_origin_requires_estimable_upstream_contrast():
    left = _factorial_cell("left", "transformer", "event")
    right = _factorial_cell("right", "ssm", "event")
    contrast = FactorialContrastSpec(
        contrast_id="architecture-effect",
        kind=FactorialContrastKind.ARCHITECTURE_MAIN,
        architectures=("transformer", "ssm"),
        tokenizers=("event",),
        fixed_axes={
            "dataset_id": "dataset",
            "session_id": "session",
            "training_seed": 0,
            "checkpoint": "step-10",
        },
    )
    spec = FactorialMechanismSpec(
        study_id="upstream",
        cells=(left, right),
        contrasts=(contrast,),
        matched_covariates=(MatchedCovariate("token_budget"),),
    )
    report = analyze_factorial_mechanisms(
        spec,
        {"left": _factorial_outcome(0.8), "right": _factorial_outcome(0.7)},
    )
    origin = factorial_origin_from_report(report, "architecture-effect")
    assert origin.factorial_study_fingerprint == report.study_fingerprint
    assert origin.cell_ids == ("left", "right")

    confounded_right = FactorialCellSpec(
        **{
            **right.to_dict(),
            "target_universe": tuple(right.target_universe),
            "covariates": {"token_budget": 64},
        }
    )
    bad_spec = FactorialMechanismSpec(
        study_id="upstream-confounded",
        cells=(left, confounded_right),
        contrasts=(contrast,),
        matched_covariates=(MatchedCovariate("token_budget"),),
    )
    bad_report = analyze_factorial_mechanisms(
        bad_spec,
        {"left": _factorial_outcome(0.8), "right": _factorial_outcome(0.7)},
    )
    with pytest.raises(ValueError, match="non-estimable"):
        factorial_origin_from_report(bad_report, "architecture-effect")
