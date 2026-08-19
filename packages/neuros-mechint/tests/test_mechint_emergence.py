import pytest

from neuros_mechint.benchmarks import (
    CausalEffectRecord,
    CheckpointMechanismState,
    MechanismContext,
    analyze_mechanism_emergence,
    run_mechanism_emergence_benchmark,
)


def _state(step: int, session: str, effects: dict[str, float]):
    return CheckpointMechanismState(
        step=step,
        record=CausalEffectRecord(
            context=MechanismContext(
                context_id=f"{session}-{step}",
                architecture="ssm",
                dataset_id="synthetic",
                session_id=session,
                checkpoint=str(step),
            ),
            baseline_metric=1.0,
            effect_map=effects,
        ),
    )


def test_mechanism_emergence_ground_truth_gate():
    payload = run_mechanism_emergence_benchmark()
    assert payload["passed"] is True
    assert payload["report"]["global_stable_step"] == 200


def test_emergence_requires_checkpoint_to_be_only_varying_context_axis():
    states = [
        _state(0, "s1", {"a": -0.1, "b": -0.2, "c": -0.3}),
        _state(10, "s2", {"a": -1.0, "b": -2.0, "c": -3.0}),
    ]
    with pytest.raises(ValueError, match="session"):
        analyze_mechanism_emergence(states)
