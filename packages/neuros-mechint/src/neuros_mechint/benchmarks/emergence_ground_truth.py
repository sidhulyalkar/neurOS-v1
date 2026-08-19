"""Known-ground-truth checkpoint-emergence benchmark."""

from __future__ import annotations

from .emergence import CheckpointMechanismState, analyze_mechanism_emergence
from .shared_computation import CausalEffectRecord, MechanismContext


def _record(step: int, effects: dict[str, float]) -> CheckpointMechanismState:
    return CheckpointMechanismState(
        step=step,
        record=CausalEffectRecord(
            context=MechanismContext(
                context_id=f"checkpoint-{step}",
                architecture="synthetic-ssm",
                dataset_id="synthetic-emergence",
                session_id="session-1",
                subject_id="subject-1",
                checkpoint=str(step),
            ),
            baseline_metric=1.0,
            effect_map=effects,
            control_map={target: 0.0 for target in effects},
        ),
    )


def run_mechanism_emergence_benchmark() -> dict[str, object]:
    """Verify that stable causal structure is detected at the known checkpoint."""

    states = (
        _record(0, {"early": -0.05, "middle": 0.02, "late": -0.01}),
        _record(100, {"early": -0.20, "middle": -0.60, "late": -0.25}),
        _record(200, {"early": -0.60, "middle": -1.10, "late": -2.10}),
        _record(300, {"early": -1.00, "middle": -2.00, "late": -4.00}),
    )
    report = analyze_mechanism_emergence(
        states,
        effect_fraction=0.5,
        stable_spearman=0.8,
        stable_sign_agreement=1.0,
        min_shared_target_fraction=1.0,
        consecutive_checkpoints=2,
        top_k=2,
    )
    targets = {item.target: item for item in report.target_emergence}
    passed = (
        report.global_stable_step == 200
        and targets["early"].first_stable_step == 200
        and targets["middle"].first_stable_step == 200
        and targets["late"].first_stable_step == 200
    )
    return {
        "passed": passed,
        "expected_global_stable_step": 200,
        "report": report.to_dict(),
    }
