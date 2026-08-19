"""Synthetic ground-truth checks for shared-computation inference."""

from __future__ import annotations

from typing import Any

from .shared_computation import (
    CausalEffectRecord,
    MechanismContext,
    analyze_shared_computation,
)


def _record(
    context_id: str,
    architecture: str,
    session_id: str,
    effects: dict[str, float],
) -> CausalEffectRecord:
    return CausalEffectRecord(
        context=MechanismContext(
            context_id=context_id,
            architecture=architecture,
            dataset_id="shared-computation-ground-truth",
            session_id=session_id,
        ),
        baseline_metric=1.0,
        effect_map=effects,
        control_map={target: 0.0 for target in effects},
        metric_name="synthetic_score",
    )


def run_shared_computation_benchmark() -> dict[str, Any]:
    """Verify shared and architecture-specific hypothesis recovery on known maps."""

    transformer = {"w0": -1.0, "w1": -2.0, "w2": -4.0, "w3": -0.5}
    ssm_shared = {"w0": -1.1, "w1": -2.1, "w2": -4.2, "w3": -0.4}
    shared_records = [
        _record("t-s1", "transformer", "s1", transformer),
        _record(
            "t-s2",
            "transformer",
            "s2",
            {key: value * 0.9 for key, value in transformer.items()},
        ),
        _record("s-s1", "ssm", "s1", ssm_shared),
        _record(
            "s-s2",
            "ssm",
            "s2",
            {key: value * 1.1 for key, value in ssm_shared.items()},
        ),
    ]
    shared_analysis = analyze_shared_computation(shared_records, top_k=2)
    shared_ids = {item.hypothesis_id for item in shared_analysis.hypotheses}

    ssm_divergent = {"w0": -4.0, "w1": -0.5, "w2": -1.0, "w3": -2.0}
    divergent_records = [
        _record("t-s1", "transformer", "s1", transformer),
        _record(
            "t-s2",
            "transformer",
            "s2",
            {key: value * 0.9 for key, value in transformer.items()},
        ),
        _record("s-s1", "ssm", "s1", ssm_divergent),
        _record(
            "s-s2",
            "ssm",
            "s2",
            {key: value * 1.1 for key, value in ssm_divergent.items()},
        ),
    ]
    divergent_analysis = analyze_shared_computation(divergent_records, top_k=2)
    divergent_ids = {item.hypothesis_id for item in divergent_analysis.hypotheses}

    shared_passed = "shared-causal-temporal-structure" in shared_ids
    divergent_passed = "architecture-specific-implementation" in divergent_ids
    return {
        "passed": shared_passed and divergent_passed,
        "shared_mechanism": {
            "passed": shared_passed,
            "hypotheses": [item.to_dict() for item in shared_analysis.hypotheses],
            "cross_architecture": shared_analysis.comparison.axis_stability[
                "cross_architecture"
            ].to_dict(),
        },
        "architecture_specific": {
            "passed": divergent_passed,
            "hypotheses": [item.to_dict() for item in divergent_analysis.hypotheses],
            "cross_architecture": divergent_analysis.comparison.axis_stability[
                "cross_architecture"
            ].to_dict(),
        },
    }
