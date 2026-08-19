from __future__ import annotations

import json

import pytest

from neuros_mechint.benchmarks import (
    FactorialAnalysisPolicy,
    FactorialCellOutcome,
    FactorialCellSpec,
    FactorialContrastKind,
    FactorialContrastSpec,
    FactorialMechanismSpec,
    analyze_factorial_mechanisms,
    read_factorial_artifact,
    run_factorial_ground_truth_benchmark,
    write_factorial_artifact,
)


def _cell(
    cell_id: str,
    *,
    architecture: str = "a",
    tokenizer: str = "event",
    checkpoint: str = "step:1",
    checkpoint_maturity: float = 1.0,
) -> FactorialCellSpec:
    return FactorialCellSpec(
        cell_id=cell_id,
        architecture=architecture,
        tokenizer_id=tokenizer,
        model_id=f"{architecture}-model",
        model_revision=f"{architecture}-rev",
        tokenizer_revision=f"{tokenizer}-rev",
        dataset_id="dataset",
        dataset_revision="dataset-rev",
        session_id="session-1",
        metric_name="score",
        discovery_method="fixed",
        discovery_partition_id="disc-1",
        validation_partition_id="val-1",
        subject_id="subject-1",
        training_seed=0,
        checkpoint=checkpoint,
        checkpoint_maturity=checkpoint_maturity,
        target_universe=("x", "y"),
    )


def _outcome(
    joint: float,
    *,
    task: float = 0.8,
    protocol: str = "protocol",
) -> FactorialCellOutcome:
    return FactorialCellOutcome(
        task_metric=task,
        candidate_size=1,
        validation_sufficiency=joint,
        validation_necessity=joint,
        validation_joint_faithfulness=joint,
        validation_joint_random_percentile=1.0,
        discovery_to_validation_drop=0.0,
        intervention_baseline_sensitivity=0.01,
        promotion_passed=True,
        source_study_fingerprint=f"study-{joint}",
        source_run_hash=f"run-{joint}",
        evidence_protocol_fingerprint=protocol,
        effect_map={"x": joint, "y": 1.0 - joint},
    )


def test_factorial_ground_truth_recovers_interaction_and_rejects_bad_designs() -> None:
    report = run_factorial_ground_truth_benchmark()
    assert report.passed
    assert report.observed_interactions == pytest.approx((-0.5, -0.5))
    assert report.cross_session_replication_ready
    assert report.confound_rejected
    assert report.missing_cell_rejected


def test_checkpoint_contrast_requires_matched_task_performance() -> None:
    early = _cell("early", checkpoint="step:1", checkpoint_maturity=0.5)
    late = _cell("late", checkpoint="step:2", checkpoint_maturity=1.0)
    contrast = FactorialContrastSpec(
        contrast_id="checkpoint",
        kind=FactorialContrastKind.CHECKPOINT,
        architectures=("a",),
        tokenizers=("event",),
        checkpoints=("step:1", "step:2"),
        fixed_axes={
            "dataset_id": "dataset",
            "session_id": "session-1",
            "subject_id": "subject-1",
            "training_seed": 0,
        },
    )
    spec = FactorialMechanismSpec(
        study_id="checkpoint-test",
        cells=(early, late),
        contrasts=(contrast,),
        policy=FactorialAnalysisPolicy(max_task_metric_delta=0.01),
    )
    result = analyze_factorial_mechanisms(
        spec,
        {
            "early": _outcome(0.8, task=0.70),
            "late": _outcome(0.8, task=0.90),
        },
    ).contrasts[0]
    assert not result.estimable
    assert any("task metric delta" in reason for reason in result.reasons)


def test_factorial_contrast_rejects_mismatched_evidence_protocol() -> None:
    left = _cell("left", architecture="a")
    right = _cell("right", architecture="b")
    contrast = FactorialContrastSpec(
        contrast_id="architecture",
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
    spec = FactorialMechanismSpec(
        study_id="protocol-test",
        cells=(left, right),
        contrasts=(contrast,),
    )
    result = analyze_factorial_mechanisms(
        spec,
        {
            "left": _outcome(0.8, protocol="p1"),
            "right": _outcome(0.8, protocol="p2"),
        },
    ).contrasts[0]
    assert not result.estimable
    assert "evidence protocol fingerprint differs" in result.reasons


def test_factorial_artifact_detects_tampering(tmp_path) -> None:
    left = _cell("left", architecture="a")
    right = _cell("right", architecture="b")
    contrast = FactorialContrastSpec(
        contrast_id="architecture",
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
    spec = FactorialMechanismSpec(
        study_id="artifact-test",
        cells=(left, right),
        contrasts=(contrast,),
    )
    report = analyze_factorial_mechanisms(
        spec,
        {"left": _outcome(0.8), "right": _outcome(0.7)},
    )
    path = tmp_path / "factorial.json"
    artifact_hash = write_factorial_artifact(report, path)
    loaded = read_factorial_artifact(path)
    assert loaded["study_fingerprint"] == report.study_fingerprint
    assert artifact_hash

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["result"]["cells"][0]["outcome"]["task_metric"] = 123.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        read_factorial_artifact(path)
