from __future__ import annotations

import numpy as np

from neuros.arena import (
    ArenaScenario,
    DeviceProfile,
    DisplayProfile,
    ParticipantProfile,
    StageSpec,
    TransportProfile,
    split_contiguous_recording,
    validate_covariance_anchor_held_out,
    run_scenario,
)


def make_world(seed: int, *, amplitude: float, noise: float):
    scenario = ArenaScenario(
        "scientific-validation",
        (
            StageSpec("rest", 2.0, None, 0.0),
            StageSpec("sight", 3.0, 10.0, 0.9),
            StageSpec("guard", 3.0, 12.0, 0.9),
        ),
        seed=23,
    )
    return run_scenario(
        scenario,
        ParticipantProfile(seed=seed, ssvep_amplitude_uv=amplitude, colored_noise_uv=noise),
        DeviceProfile(chunk_samples=5, sensor_noise_uv=0.1, line_noise_uv=0.0),
        DisplayProfile(),
        TransportProfile(),
    )


def test_default_run_carries_scientific_world_model_evidence_card():
    run = make_world(31, amplitude=6.0, noise=4.0)
    card = run.report["world_model_evidence"]
    assert card["model_name"] == "driven_state_space"
    assert card["evidence_level"] == "W1-causal-phenomenological"
    assert card["stimulus_causal"] is True
    assert "biophysical cortical mechanism" in card["unsupported_claims"]


def test_contiguous_split_supports_guard_interval_without_overlap():
    data = np.arange(2 * 100, dtype=float).reshape(2, 100)
    calibration, validation = split_contiguous_recording(
        data,
        calibration_fraction=0.5,
        guard_samples=5,
    )
    assert calibration.shape == (2, 45)
    assert validation.shape == (2, 45)
    assert calibration[0, -1] == 44
    assert validation[0, 0] == 55


def test_reality_anchor_is_judged_on_independent_eeg():
    near = make_world(41, amplitude=7.0, noise=3.8)
    far = make_world(97, amplitude=0.8, noise=13.0)
    calibration, validation = split_contiguous_recording(
        near.device_output.data_uv,
        calibration_fraction=0.5,
        guard_samples=10,
    )
    result = validate_covariance_anchor_held_out(
        {"near": near, "far": far},
        calibration,
        validation,
        temperature=0.25,
    )
    assert result.calibration.by_world()["near"] > result.calibration.by_world()["far"]
    assert result.weighted_validation_distance < result.uniform_validation_distance
    assert result.relative_improvement > 0
    assert result.best_calibration_world == "near"
    payload = result.to_dict()
    assert "Out-of-sample" in payload["evidence_boundary"]
