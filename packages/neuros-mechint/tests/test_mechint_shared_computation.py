import pytest

from neuros_mechint.benchmarks import (
    CausalEffectRecord,
    MechanismContext,
    analyze_shared_computation,
    compare_causal_records,
)


def _record(context_id: str, architecture: str, session: str, scale: float = 1.0):
    return CausalEffectRecord(
        context=MechanismContext(
            context_id=context_id,
            architecture=architecture,
            dataset_id="synthetic",
            session_id=session,
        ),
        baseline_metric=1.0,
        effect_map={
            "w0": -1.0 * scale,
            "w1": -2.0 * scale,
            "w2": -4.0 * scale,
            "w3": -0.5 * scale,
        },
        control_map={"w0": 0.0, "w1": 0.0, "w2": 0.0, "w3": 0.0},
    )


def test_shared_computation_finds_architecture_invariant_effect_ordering():
    records = [
        _record("t-s1", "transformer", "s1", 1.0),
        _record("t-s2", "transformer", "s2", 0.9),
        _record("s-s1", "ssm", "s1", 1.1),
        _record("s-s2", "ssm", "s2", 1.0),
    ]
    analysis = analyze_shared_computation(records, top_k=2)
    cross = analysis.comparison.axis_stability["cross_architecture"]
    matched_architecture = analysis.comparison.isolated_axis_stability["architecture"]
    matched_session = analysis.comparison.isolated_axis_stability["session"]
    assert cross.pair_count == 4
    assert matched_architecture.pair_count == 2
    assert matched_session.pair_count == 2
    assert cross.median_spearman_r == pytest.approx(1.0)
    assert matched_architecture.median_spearman_r == pytest.approx(1.0)
    assert cross.median_sign_agreement == pytest.approx(1.0)
    assert cross.median_top_k_jaccard == pytest.approx(1.0)
    ids = {item.hypothesis_id for item in analysis.hypotheses}
    assert "shared-causal-temporal-structure" in ids


def test_comparison_tracks_axes_and_target_coverage():
    left = _record("a", "transformer", "s1")
    right = CausalEffectRecord(
        context=MechanismContext(
            context_id="b",
            architecture="ssm",
            dataset_id="synthetic-2",
            session_id="s2",
        ),
        baseline_metric=0.8,
        effect_map={"w0": -1.0, "w1": -2.0, "w2": -4.0, "extra": 3.0},
    )
    report = compare_causal_records([left, right], top_k=2)
    pair = report.pairwise[0]
    assert set(pair.axes_changed) == {"architecture", "dataset", "session"}
    assert pair.stability.shared_targets == 3
    assert pair.stability.union_targets == 5
    assert pair.stability.shared_target_fraction == pytest.approx(3 / 5)
    assert "architecture" not in report.isolated_axis_stability


def test_low_target_coverage_does_not_generate_shared_architecture_hypothesis():
    def low_coverage(context_id: str, architecture: str, session: str):
        unique = {f"{context_id}-u{index}": float(index + 1) for index in range(5)}
        return CausalEffectRecord(
            context=MechanismContext(
                context_id=context_id,
                architecture=architecture,
                dataset_id="synthetic",
                session_id=session,
            ),
            baseline_metric=1.0,
            effect_map={"shared0": -1.0, "shared1": -2.0, "shared2": -4.0, **unique},
        )

    analysis = analyze_shared_computation(
        [
            low_coverage("t-s1", "transformer", "s1"),
            low_coverage("t-s2", "transformer", "s2"),
            low_coverage("s-s1", "ssm", "s1"),
            low_coverage("s-s2", "ssm", "s2"),
        ],
        top_k=2,
    )
    matched = analysis.comparison.isolated_axis_stability["architecture"]
    assert matched.median_shared_target_fraction < analysis.policy.min_shared_target_fraction
    ids = {item.hypothesis_id for item in analysis.hypotheses}
    assert "shared-causal-temporal-structure" not in ids
