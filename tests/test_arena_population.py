from __future__ import annotations

from dataclasses import replace

import numpy as np

from neuros.arena import ArenaManifest, ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, StageSpec, TransportProfile
from neuros.arena.population import ParameterDistribution, PopulationSpec, run_population


def manifest() -> ArenaManifest:
    return ArenaManifest(
        ArenaScenario(
            "population-smoke",
            (
                StageSpec("rest", 0.75, None, 0.0),
                StageSpec("target", 1.5, 10.0, 0.9),
            ),
            seed=71,
        ),
        ParticipantProfile(seed=71, ssvep_amplitude_uv=6.0),
        DeviceProfile(chunk_samples=5, sensor_noise_uv=0.1),
        DisplayProfile(),
        TransportProfile(),
    )


def spec() -> PopulationSpec:
    return PopulationSpec(
        size=6,
        seed=101,
        parameters=(
            ParameterDistribution("participant.ssvep_amplitude_uv", "uniform", low=2.0, high=9.0),
            ParameterDistribution("participant.alpha_frequency_hz", "uniform", low=8.0, high=12.5),
            ParameterDistribution("display.frame_drop_probability", "uniform", low=0.0, high=0.08),
            ParameterDistribution("transport.jitter_ms", "uniform", low=0.0, high=15.0),
            ParameterDistribution("world_model.parameters.resonance_damping", "uniform", low=0.12, high=0.4),
        ),
    )


def test_population_is_reproducible_and_reports_coverage_quantiles():
    first = run_population(manifest(), spec())
    second = run_population(manifest(), spec())
    assert first.to_dict() == second.to_dict()
    assert len(first.trials) == 6
    assert "target_snr_db_mean" in first.summary
    snr = first.summary["target_snr_db_mean"]
    assert snr["min"] <= snr["p05"] <= snr["p50"] <= snr["p95"] <= snr["max"]


def test_population_samples_remain_within_declared_world_envelope():
    result = run_population(manifest(), spec())
    for trial in result.trials:
        assert 2.0 <= trial.sampled["participant.ssvep_amplitude_uv"] <= 9.0
        assert 8.0 <= trial.sampled["participant.alpha_frequency_hz"] <= 12.5
        assert 0.0 <= trial.sampled["display.frame_drop_probability"] <= 0.08
        assert 0.0 <= trial.sampled["transport.jitter_ms"] <= 15.0
        assert 0.12 <= trial.sampled["world_model.parameters.resonance_damping"] <= 0.4


def test_population_custom_evaluator_can_score_application_contracts():
    def evaluator(run):
        # A deliberately simple application-facing metric for the contract test.
        # Real applications can return precision, false activation, recovery, etc.
        snr = list(run.report["metrics"]["target_snr_db"].values())
        return {"application_proxy": float(np.mean(snr)) if snr else 0.0}

    result = run_population(manifest(), replace(spec(), size=3), evaluator=evaluator)
    assert all("application_proxy" in trial.metrics for trial in result.trials)
    assert "application_proxy" in result.summary
