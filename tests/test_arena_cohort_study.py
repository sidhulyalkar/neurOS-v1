from __future__ import annotations

import numpy as np

from neuros.arena import ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, StageSpec, TransportProfile, run_scenario
from neuros.arena.studies import run_leave_one_domain_out_covariance_study


def make_world(seed: int, *, amplitude: float, noise: float):
    return run_scenario(
        ArenaScenario(
            "cohort-study",
            (
                StageSpec("rest", 1.0, None, 0.0),
                StageSpec("target", 3.0, 10.0, 0.9),
            ),
            seed=37,
        ),
        ParticipantProfile(seed=seed, ssvep_amplitude_uv=amplitude, colored_noise_uv=noise),
        DeviceProfile(sensor_noise_uv=0.05, line_noise_uv=0.0, chunk_samples=5),
        DisplayProfile(),
        TransportProfile(),
    )


def test_leave_one_domain_out_weights_transfer_to_unseen_similar_domains():
    near = make_world(31, amplitude=7.0, noise=4.0)
    far = make_world(91, amplitude=0.7, noise=13.0)
    rng = np.random.default_rng(123)
    base = near.device_output.data_uv.astype(float)
    domains = {
        f"subject-{index}": base + rng.normal(0.0, scale, size=base.shape)
        for index, scale in enumerate((0.05, 0.08, 0.10, 0.12), start=1)
    }
    result = run_leave_one_domain_out_covariance_study(
        {"near": near, "far": far},
        domains,
        temperature=0.25,
    )
    assert len(result.folds) == 4
    assert result.fraction_improved == 1.0
    assert result.mean_relative_improvement > 0
    assert result.mean_weighted_distance < result.mean_uniform_distance
    assert all(fold.best_weighted_world == "near" for fold in result.folds)
    assert "human BCI accuracy" in result.to_dict()["evidence_boundary"]


def test_leave_one_domain_out_rejects_too_few_real_domains():
    near = make_world(31, amplitude=7.0, noise=4.0)
    far = make_world(91, amplitude=0.7, noise=13.0)
    domains = {
        "a": near.device_output.data_uv,
        "b": near.device_output.data_uv,
    }
    try:
        run_leave_one_domain_out_covariance_study({"near": near, "far": far}, domains)
    except ValueError as exc:
        assert "at least three" in str(exc)
    else:
        raise AssertionError("a cross-domain study with fewer than three domains should fail")
