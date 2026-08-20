from __future__ import annotations

import numpy as np
import pytest

from neuros_sourceweigher import (
    DistanceWeigher,
    GibbsRiskWeigher,
    MMDSourceWeigher,
    OnlineSourceWeigher,
    ReliabilityWeightedFusion,
    RepresentationSourceWeigher,
    RiemannianCovarianceWeigher,
    RunningFeatureSummary,
    SourceWeigher,
    jensen_shannon_weight_shift,
    leave_one_source_out_stability,
    project_to_simplex,
    summarize_features,
)


def _old_lstsq_then_project(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    raw, *_ = np.linalg.lstsq(source.T, target, rcond=None)
    return project_to_simplex(raw)


def test_simplex_projection_is_valid() -> None:
    projected = project_to_simplex(np.array([0.7, -0.2, 1.4]))
    assert np.all(projected >= 0)
    assert projected.sum() == pytest.approx(1.0)


def test_constrained_solver_beats_old_one_shot_projection() -> None:
    # Fixed counterexample: projection(lstsq(A,b)) is not generally the solution
    # to min ||Aw-b|| subject to w in the simplex.
    source = np.array(
        [
            [0.30471708, -1.03998411, 0.75045120],
            [0.94056472, -1.95103519, -1.30217951],
            [0.12784040, -0.31624259, -0.01680116],
            [-0.85304393, 0.87939797, 0.77779194],
        ]
    )
    target = np.array([0.06603070, 1.12724121, 0.46750934])

    old_w = _old_lstsq_then_project(source, target)
    new = SourceWeigher(ridge=0.0, standardize=False).estimate(source, target)
    old_residual2 = float(np.sum((source.T @ old_w - target) ** 2))
    new_residual2 = float(np.sum((source.T @ new.weights - target) ** 2))

    assert new_residual2 < 0.5 * old_residual2
    assert np.all(new.weights >= 0)
    assert new.weights.sum() == pytest.approx(1.0)


def test_convex_hull_target_is_reconstructed() -> None:
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = np.array([0.2, 0.3])
    result = SourceWeigher(ridge=0.0, standardize=False).estimate(source, target)
    assert result.residual < 1e-6
    assert source.T @ result.weights == pytest.approx(target, abs=1e-6)


def test_nonfinite_source_is_excluded_not_silently_used() -> None:
    source = np.array([[0.0, 0.0], [np.inf, 1.0], [1.0, 1.0]])
    target = np.array([0.2, 0.2])
    result = SourceWeigher().estimate(source, target)
    assert result.weights[1] == 0.0
    assert result.diagnostics.excluded_sources == (1,)
    assert result.weights.sum() == pytest.approx(1.0)


def test_quality_score_breaks_similarity_tie() -> None:
    source = np.array([[0.0, 0.0], [0.0, 0.0]])
    target = np.array([0.0, 0.0])
    result = SourceWeigher(
        ridge=0.0,
        standardize=False,
        quality_strength=0.5,
    ).estimate(source, target, quality_scores=np.array([1.0, 0.0]))
    assert result.weights[0] > result.weights[1]


def test_distance_weigher_prefers_nearest_source() -> None:
    source = np.array([[0.0, 0.0], [5.0, 5.0], [1.0, 1.0]])
    target = np.array([0.1, 0.1])
    result = DistanceWeigher(temperature=0.2).estimate(source, target)
    assert int(np.argmax(result.weights)) == 0


def test_gibbs_risk_prefers_lowest_risk() -> None:
    result = GibbsRiskWeigher(temperature=0.05).estimate(
        np.array([0.1, 0.7, 0.3])
    )
    assert int(np.argmax(result.weights)) == 0


def test_online_weigher_limits_l1_churn() -> None:
    estimator = DistanceWeigher(temperature=0.05, standardize=False)
    online = OnlineSourceWeigher(estimator, adaptation_rate=1.0, max_l1_step=0.2)
    sources = np.array([[0.0], [10.0]])
    first = online.update(sources, np.array([0.0]))
    second = online.update(sources, np.array([10.0]))
    assert np.linalg.norm(second.weights - first.weights, ord=1) <= 0.2000001


def test_running_summary_matches_batch_summary() -> None:
    rng = np.random.default_rng(2)
    x = rng.normal(size=(50, 4))
    running = RunningFeatureSummary(4)
    running.update(x[:17]).update(x[17:])
    assert running.vector(log_std=True) == pytest.approx(
        summarize_features(x, statistics=("mean", "log_std")), abs=1e-10
    )


def test_representation_weigher_prefers_near_embedding_domain() -> None:
    rng = np.random.default_rng(1)
    target = rng.normal(loc=0.0, scale=1.0, size=(100, 5))
    sources = {
        "near": rng.normal(loc=0.05, scale=1.0, size=(100, 5)),
        "far": rng.normal(loc=4.0, scale=1.0, size=(100, 5)),
    }
    result = RepresentationSourceWeigher().estimate(sources, target)
    assert result.by_source()["near"] > result.by_source()["far"]


def test_mmd_prefers_near_distribution() -> None:
    rng = np.random.default_rng(4)
    target = rng.normal(size=(120, 3))
    sources = {
        "near": rng.normal(loc=0.05, size=(120, 3)),
        "far": rng.normal(loc=3.0, size=(120, 3)),
    }
    result = MMDSourceWeigher(temperature=0.02, max_samples=128).estimate(
        sources, target
    )
    assert result.by_source()["near"] > result.by_source()["far"]


def test_riemannian_weigher_detects_covariance_shift() -> None:
    rng = np.random.default_rng(5)
    target = rng.normal(size=(300, 2))
    near = rng.normal(size=(300, 2))
    far = rng.normal(size=(300, 2)) @ np.diag([4.0, 0.2])
    result = RiemannianCovarianceWeigher(temperature=0.5).estimate(
        {"near": near, "far": far}, target
    )
    assert result.by_source()["near"] > result.by_source()["far"]


def test_reliability_fusion_scale_concat_and_weighted_mean() -> None:
    fusion = ReliabilityWeightedFusion(
        {"a": 3.0, "b": 1.0}, mode="scale_concat"
    )
    fused = fusion.fuse({"a": np.array([2.0, 4.0]), "b": np.array([8.0])})
    assert fused == pytest.approx(np.array([1.5, 3.0, 2.0]))

    mean_fusion = ReliabilityWeightedFusion(
        {"a": 3.0, "b": 1.0}, mode="weighted_mean"
    )
    mean = mean_fusion.fuse(
        {"a": np.array([0.0, 4.0]), "b": np.array([4.0, 0.0])}
    )
    assert mean == pytest.approx(np.array([1.0, 3.0]))


def test_stability_and_weight_shift_diagnostics() -> None:
    source = np.array([[0.0, 0.0], [1.0, 1.0], [0.2, 0.2]])
    target = np.array([0.1, 0.1])
    estimator = SourceWeigher(ridge=1e-2)
    report = leave_one_source_out_stability(estimator, source, target)
    assert report.l1_change_when_removed.shape == (3,)
    assert 0 <= report.most_influential_source < 3
    assert jensen_shannon_weight_shift(
        np.array([0.5, 0.5]), np.array([0.5, 0.5])
    ) == pytest.approx(0.0)
