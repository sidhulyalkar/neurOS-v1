from __future__ import annotations

import torch

from neuros_mechint.adapters import PyTorchAdapter
from neuros_mechint.benchmarks import (
    CausalEffectRecord,
    CircuitCandidate,
    EvidenceExample,
    EvidencePackPolicy,
    EvidencePackSpec,
    EvidenceSplit,
    FactorialCellSpec,
    FactorialContrastKind,
    FactorialContrastSpec,
    FactorialMechanismSpec,
    MechanismContext,
    run_adapter_evidence_pack,
)
from neuros_mechint.benchmarks.ground_truth import GroundTruthCausalMLP
from neuros_mechint.core import EvidenceTier, OutputMetric
from neuros_mechint.integrations.factorial_study import (
    FactorialEvidenceCellInput,
    run_factorial_evidence_study,
)


def _pack(architecture: str):
    model = GroundTruthCausalMLP()
    adapter = PyTorchAdapter(model)
    metric = OutputMetric(lambda output: output.mean(), name="mean_output")
    metadata = {
        "architecture": architecture,
        "checkpoint": "step:1",
        "checkpoint_maturity": 1.0,
        "discovery_partition_id": "disc-1",
        "validation_partition_id": "val-1",
        "session_id": "session-1",
        "subject_id": "subject-1",
        "training_seed": 0,
        "token_budget": 128,
        "temporal_resolution_ms": 10.0,
        "downstream_capacity": 16,
        "training_compute": 100,
    }
    spec = EvidencePackSpec(
        pack_id=f"pack-{architecture}",
        model_id=f"model-{architecture}",
        model_revision=f"model-{architecture}-rev",
        tokenizer_id="event",
        tokenizer_revision="event-rev",
        dataset_id="dataset",
        dataset_revision="dataset-rev",
        metric_name=metric.name,
        target_universe=("source", "causal", "nuisance"),
        discovery_method="fixed-candidate",
        intervention_baselines=("zero",),
        random_trials=100,
        seed=0,
        evidence_tier=EvidenceTier.SCIENTIFIC_SYNTHETIC,
        metadata=metadata,
    )
    examples = (
        EvidenceExample(
            f"{architecture}-discover-1",
            torch.tensor([[1.0, 0.0]]),
            EvidenceSplit.DISCOVERY,
        ),
        EvidenceExample(
            f"{architecture}-discover-2",
            torch.tensor([[2.0, 0.0]]),
            EvidenceSplit.DISCOVERY,
        ),
        EvidenceExample(
            f"{architecture}-validate-1",
            torch.tensor([[3.0, 0.0]]),
            EvidenceSplit.VALIDATION,
        ),
        EvidenceExample(
            f"{architecture}-validate-2",
            torch.tensor([[4.0, 0.0]]),
            EvidenceSplit.VALIDATION,
        ),
    )
    return run_adapter_evidence_pack(
        spec=spec,
        adapter=adapter,
        metric=metric,
        examples=examples,
        candidate=CircuitCandidate(
            name="known-route",
            targets=("source", "causal"),
            source="known-ground-truth",
        ),
        pack_policy=EvidencePackPolicy(
            min_validation_examples=2,
            min_validation_pass_rate=1.0,
            min_validation_joint_median=0.8,
            max_joint_generalization_drop=0.0,
            require_multiple_intervention_baselines=False,
            bootstrap_samples=20,
        ),
        include_magnitude_baseline=False,
    )


def _cell(architecture: str) -> FactorialCellSpec:
    return FactorialCellSpec(
        cell_id=f"cell-{architecture}",
        architecture=architecture,
        tokenizer_id="event",
        model_id=f"model-{architecture}",
        model_revision=f"model-{architecture}-rev",
        tokenizer_revision="event-rev",
        dataset_id="dataset",
        dataset_revision="dataset-rev",
        session_id="session-1",
        metric_name="mean_output",
        discovery_method="fixed-candidate",
        discovery_partition_id="disc-1",
        validation_partition_id="val-1",
        subject_id="subject-1",
        training_seed=0,
        checkpoint="step:1",
        checkpoint_maturity=1.0,
        target_universe=("source", "causal", "nuisance"),
        covariates={
            "token_budget": 128,
            "temporal_resolution_ms": 10.0,
            "downstream_capacity": 16,
            "training_compute": 100,
        },
    )


def _effect_record(architecture: str) -> CausalEffectRecord:
    return CausalEffectRecord(
        context=MechanismContext(
            context_id=f"ctx-{architecture}",
            architecture=architecture,
            dataset_id="dataset",
            session_id="session-1",
            subject_id="subject-1",
            checkpoint="step:1",
        ),
        baseline_metric=1.0,
        metric_name="mean_output",
        effect_map={"source": 1.0, "causal": 1.0, "nuisance": 0.0},
    )


def test_factorial_bridge_consumes_real_evidence_pack_results() -> None:
    cell_a = _cell("a")
    cell_b = _cell("b")
    contrast = FactorialContrastSpec(
        contrast_id="architecture-main",
        kind=FactorialContrastKind.ARCHITECTURE_MAIN,
        architectures=("a", "b"),
        tokenizers=("event",),
        fixed_axes={
            "dataset_id": "dataset",
            "session_id": "session-1",
            "subject_id": "subject-1",
            "training_seed": 0,
            "checkpoint": "step:1",
        },
    )
    factorial = FactorialMechanismSpec(
        study_id="evidence-pack-integration",
        cells=(cell_a, cell_b),
        contrasts=(contrast,),
    )
    report = run_factorial_evidence_study(
        factorial,
        (
            FactorialEvidenceCellInput(
                cell_id=cell_a.cell_id,
                evidence_pack=_pack("a"),
                effect_record=_effect_record("a"),
            ),
            FactorialEvidenceCellInput(
                cell_id=cell_b.cell_id,
                evidence_pack=_pack("b"),
                effect_record=_effect_record("b"),
            ),
        ),
    )
    result = report.contrasts[0]
    assert result.estimable
    assert result.effect_map_stability is not None
    assert result.task_metric_range == 0.0
    assert report.missing_cell_ids == ()
