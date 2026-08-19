"""Known-ground-truth gate for v0.7 factorial mechanism science."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .factorial import (
    FactorialAnalysisPolicy,
    FactorialCellOutcome,
    FactorialCellSpec,
    FactorialContrastKind,
    FactorialContrastSpec,
    FactorialMechanismSpec,
    MatchedCovariate,
    analyze_factorial_mechanisms,
    preregister_2x2_contrasts,
)


@dataclass(frozen=True, slots=True)
class FactorialGroundTruthReport:
    expected_interaction: float
    observed_interactions: tuple[float, ...]
    max_interaction_error: float
    cross_session_replication_ready: bool
    confound_rejected: bool
    missing_cell_rejected: bool
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "confound_rejected": self.confound_rejected,
            "cross_session_replication_ready": self.cross_session_replication_ready,
            "expected_interaction": self.expected_interaction,
            "max_interaction_error": self.max_interaction_error,
            "missing_cell_rejected": self.missing_cell_rejected,
            "observed_interactions": list(self.observed_interactions),
            "passed": self.passed,
        }


def _cell(
    *,
    cell_id: str,
    architecture: str,
    tokenizer: str,
    session: str,
    token_budget: int = 128,
    available: bool = True,
    missing_reason: str | None = None,
) -> FactorialCellSpec:
    return FactorialCellSpec(
        cell_id=cell_id,
        architecture=architecture,
        tokenizer_id=tokenizer,
        model_id=f"{architecture}:{session}",
        model_revision=f"{architecture}-{session}-sha",
        tokenizer_revision=f"{tokenizer}-sha",
        dataset_id="synthetic-factorial",
        dataset_revision="dataset-sha",
        session_id=session,
        metric_name="score",
        discovery_method="synthetic-fixed-candidate",
        discovery_partition_id=f"{session}:discovery",
        validation_partition_id=f"{session}:validation",
        subject_id="subject-0",
        training_seed=0,
        checkpoint="step:100",
        checkpoint_maturity=1.0,
        target_universe=("route_a", "route_b"),
        covariates={
            "token_budget": token_budget,
            "temporal_resolution_ms": 10.0,
            "downstream_capacity": 32,
            "training_compute": 1000,
        },
        available=available,
        missing_reason=missing_reason,
    )


def _outcome(*, joint: float, task: float, switched: bool = False) -> FactorialCellOutcome:
    effect_map = {"route_a": 1.0, "route_b": 0.2}
    if switched:
        effect_map = {"route_a": 0.2, "route_b": 1.0}
    return FactorialCellOutcome(
        task_metric=task,
        candidate_size=1,
        validation_sufficiency=joint,
        validation_necessity=joint,
        validation_joint_faithfulness=joint,
        validation_joint_random_percentile=1.0,
        discovery_to_validation_drop=0.02,
        intervention_baseline_sensitivity=0.01,
        promotion_passed=joint >= 0.5,
        source_study_fingerprint=f"study-{joint}-{task}-{switched}",
        source_run_hash=f"run-{joint}-{task}-{switched}",
        evidence_protocol_fingerprint="synthetic-protocol-v1",
        effect_map=effect_map,
    )


def _base_covariates() -> tuple[MatchedCovariate, ...]:
    return (
        MatchedCovariate("token_budget"),
        MatchedCovariate("temporal_resolution_ms"),
        MatchedCovariate("downstream_capacity"),
        MatchedCovariate("training_compute"),
    )


def _primary_report():
    cells = []
    outcomes = {}
    contrasts = []
    for session, delta in (("session-1", 0.0), ("session-2", -0.02)):
        values = {
            ("arch-a", "event"): (0.90 + delta, False),
            ("arch-a", "isi"): (0.90 + delta, False),
            ("arch-b", "event"): (0.90 + delta, False),
            ("arch-b", "isi"): (0.40 + delta, True),
        }
        for (architecture, tokenizer), (joint, switched) in values.items():
            cell_id = f"{session}:{architecture}:{tokenizer}"
            cells.append(
                _cell(
                    cell_id=cell_id,
                    architecture=architecture,
                    tokenizer=tokenizer,
                    session=session,
                )
            )
            outcomes[cell_id] = _outcome(
                joint=joint,
                task=0.80 + (0.01 if tokenizer == "isi" else 0.0),
                switched=switched,
            )
        contrasts.extend(
            preregister_2x2_contrasts(
                prefix=session,
                architectures=("arch-a", "arch-b"),
                tokenizers=("event", "isi"),
                fixed_axes={
                    "dataset_id": "synthetic-factorial",
                    "session_id": session,
                    "subject_id": "subject-0",
                    "training_seed": 0,
                    "checkpoint": "step:100",
                },
                replication_namespace="synthetic",
            )
        )
    spec = FactorialMechanismSpec(
        study_id="known-architecture-tokenizer-interaction",
        cells=tuple(cells),
        contrasts=tuple(contrasts),
        matched_covariates=_base_covariates(),
        policy=FactorialAnalysisPolicy(
            max_task_metric_delta=0.02,
            max_checkpoint_maturity_delta=0.01,
            min_shared_target_fraction=1.0,
        ),
    )
    return analyze_factorial_mechanisms(spec, outcomes)


def _confounded_contrast_rejected() -> bool:
    left = _cell(
        cell_id="confound:event",
        architecture="arch-a",
        tokenizer="event",
        session="session-c",
        token_budget=128,
    )
    right = _cell(
        cell_id="confound:isi",
        architecture="arch-a",
        tokenizer="isi",
        session="session-c",
        token_budget=64,
    )
    contrast = FactorialContrastSpec(
        contrast_id="confounded-tokenizer-effect",
        kind=FactorialContrastKind.TOKENIZER_MAIN,
        architectures=("arch-a",),
        tokenizers=("event", "isi"),
        fixed_axes={
            "dataset_id": "synthetic-factorial",
            "session_id": "session-c",
            "subject_id": "subject-0",
            "training_seed": 0,
            "checkpoint": "step:100",
        },
    )
    spec = FactorialMechanismSpec(
        study_id="confound-rejection",
        cells=(left, right),
        contrasts=(contrast,),
        matched_covariates=_base_covariates(),
        policy=FactorialAnalysisPolicy(max_task_metric_delta=0.05),
    )
    report = analyze_factorial_mechanisms(
        spec,
        {
            left.cell_id: _outcome(joint=0.8, task=0.8),
            right.cell_id: _outcome(joint=0.7, task=0.8),
        },
    )
    result = report.contrasts[0]
    return (not result.estimable) and any("token_budget" in reason for reason in result.reasons)


def _missing_cell_rejected() -> bool:
    cells = (
        _cell(
            cell_id="missing:a:event",
            architecture="arch-a",
            tokenizer="event",
            session="session-m",
        ),
        _cell(
            cell_id="missing:a:isi",
            architecture="arch-a",
            tokenizer="isi",
            session="session-m",
        ),
        _cell(
            cell_id="missing:b:event",
            architecture="arch-b",
            tokenizer="event",
            session="session-m",
        ),
        _cell(
            cell_id="missing:b:isi",
            architecture="arch-b",
            tokenizer="isi",
            session="session-m",
            available=False,
            missing_reason="checkpoint unavailable",
        ),
    )
    contrast = FactorialContrastSpec(
        contrast_id="missing-interaction",
        kind=FactorialContrastKind.ARCHITECTURE_TOKENIZER_INTERACTION,
        architectures=("arch-a", "arch-b"),
        tokenizers=("event", "isi"),
        fixed_axes={
            "dataset_id": "synthetic-factorial",
            "session_id": "session-m",
            "subject_id": "subject-0",
            "training_seed": 0,
            "checkpoint": "step:100",
        },
    )
    spec = FactorialMechanismSpec(
        study_id="missing-cell-rejection",
        cells=cells,
        contrasts=(contrast,),
        matched_covariates=_base_covariates(),
    )
    outcomes = {
        cell.cell_id: _outcome(joint=0.8, task=0.8) for cell in cells if cell.available
    }
    result = analyze_factorial_mechanisms(spec, outcomes).contrasts[0]
    return (not result.estimable) and any("missing" in reason for reason in result.reasons)


def run_factorial_ground_truth_benchmark() -> FactorialGroundTruthReport:
    """Recover a known interaction while rejecting confounded/missing contrasts."""

    report = _primary_report()
    interaction_results = tuple(
        item
        for item in report.contrasts
        if item.kind is FactorialContrastKind.ARCHITECTURE_TOKENIZER_INTERACTION
    )
    expected = -0.5
    observed = tuple(
        item.outcome_effects["validation_joint_faithfulness"]
        for item in interaction_results
        if item.estimable
    )
    max_error = max(abs(value - expected) for value in observed) if observed else float("inf")
    replication = next(
        (
            item
            for item in report.replications
            if item.replication_group == "synthetic:interaction"
        ),
        None,
    )
    replication_ready = bool(replication is not None and replication.replication_ready)
    confound_rejected = _confounded_contrast_rejected()
    missing_rejected = _missing_cell_rejected()
    passed = (
        len(observed) == 2
        and max_error <= 1e-12
        and replication_ready
        and confound_rejected
        and missing_rejected
    )
    return FactorialGroundTruthReport(
        expected_interaction=expected,
        observed_interactions=observed,
        max_interaction_error=max_error,
        cross_session_replication_ready=replication_ready,
        confound_rejected=confound_rejected,
        missing_cell_rejected=missing_rejected,
        passed=passed,
    )
