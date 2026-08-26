from __future__ import annotations

import numpy as np

from neuros.arena import ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, StageSpec, TransportProfile, run_scenario
from neuros.arena.reality import anchor_worlds_by_covariance, anchor_worlds_by_embeddings


def make_world(seed: int, *, amplitude: float, noise: float):
    scenario = ArenaScenario(
        "reality-anchor",
        (
            StageSpec("rest", 1.0, None, 0.0),
            StageSpec("target", 2.0, 10.0, 0.9),
        ),
        seed=17,
    )
    return run_scenario(
        scenario,
        ParticipantProfile(seed=seed, ssvep_amplitude_uv=amplitude, colored_noise_uv=noise),
        DeviceProfile(chunk_samples=5, sensor_noise_uv=0.1, line_noise_uv=0.0),
        DisplayProfile(),
        TransportProfile(),
    )


def test_covariance_anchor_prefers_identity_domain_over_distant_world():
    near = make_world(31, amplitude=6.5, noise=4.0)
    far = make_world(83, amplitude=1.0, noise=11.0)
    # The target is deliberately the exact observed domain of the near world.
    # This verifies weighting semantics, not population prevalence.
    result = anchor_worlds_by_covariance(
        {"near": near, "far": far},
        near.device_output.data_uv,
        temperature=0.4,
    )
    weights = result.by_world()
    assert weights["near"] > weights["far"]
    assert result.effective_world_count >= 1.0
    assert result.max_weight <= 1.0


def test_embedding_anchor_prefers_matching_foundation_representation():
    rng = np.random.default_rng(9)
    target = rng.normal(size=(64, 12))
    close = target + rng.normal(0.0, 0.02, size=target.shape)
    far = rng.normal(4.0, 2.0, size=target.shape)
    result = anchor_worlds_by_embeddings({"close": close, "far": far}, target)
    weights = result.by_world()
    assert weights["close"] > weights["far"]
    payload = result.to_dict()
    assert "not probabilities" in payload["evidence_boundary"]
