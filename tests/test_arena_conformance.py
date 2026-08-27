from __future__ import annotations

from dataclasses import replace

from neuros.arena import ArenaManifest, ArenaScenario, DeviceProfile, DisplayProfile, ParticipantProfile, StageSpec, TransportProfile
from neuros.arena.conformance import (
    check_display_drop_monotonicity,
    check_fail_closed_degradation,
    check_transport_drop_monotonicity,
    search_counterexamples,
)
from neuros.arena.population import ParameterDistribution, PopulationSpec


def manifest() -> ArenaManifest:
    return ArenaManifest(
        ArenaScenario(
            "metamorphic",
            (
                StageSpec("rest", 0.75, None, 0.0),
                StageSpec("target", 1.5, 10.0, 0.9),
            ),
            seed=37,
        ),
        ParticipantProfile(seed=37, ssvep_amplitude_uv=6.0),
        DeviceProfile(chunk_samples=5, sensor_noise_uv=0.0, line_noise_uv=0.0),
        DisplayProfile(frame_drop_probability=0.01),
        TransportProfile(drop_probability=0.01, jitter_ms=2.0),
    )


def test_transport_and_display_fault_masks_are_metamorphically_monotone():
    world = manifest()
    transport = check_transport_drop_monotonicity(world, higher_drop_probability=0.30)
    display = check_display_drop_monotonicity(world, higher_drop_probability=0.35)
    assert transport.passed
    assert display.passed
    assert transport.mutated_value <= transport.base_value
    assert display.mutated_value >= display.base_value


def test_application_fail_closed_property_can_detect_an_authority_regression():
    world = manifest()
    degraded = replace(world, participant=replace(world.participant, ssvep_amplitude_uv=1.0))

    # This synthetic evaluator intentionally defines authority as inverse SNR,
    # producing a failing property so the counterexample machinery itself is tested.
    def bad_authority(run):
        snr = run.report["metrics"]["target_snr_db"]["10Hz"]
        return -float(snr)

    result = check_fail_closed_degradation(world, degraded, bad_authority)
    assert not result.passed
    assert result.mutated_manifest["participant"]["ssvep_amplitude_uv"] == 1.0


def test_adversarial_search_returns_portable_worst_worlds():
    world = manifest()
    spec = PopulationSpec(
        size=8,
        seed=44,
        parameters=(
            ParameterDistribution("participant.ssvep_amplitude_uv", "uniform", low=0.5, high=8.0),
            ParameterDistribution("display.frame_drop_probability", "uniform", low=0.0, high=0.25),
            ParameterDistribution("transport.drop_probability", "uniform", low=0.0, high=0.25),
        ),
    )

    def objective(run):
        snr = float(run.report["metrics"]["target_snr_db"]["10Hz"])
        packet_drop = float(run.report["metrics"]["transport"]["packet_drop_fraction"])
        # Lower is declared worse for this example.
        score = snr - 10.0 * packet_drop
        return score, {"snr": snr, "packet_drop": packet_drop}

    result = search_counterexamples(
        world,
        spec,
        objective,
        objective_name="synthetic_robustness_proxy",
        minimize=True,
        top_k=3,
    )
    assert result.evaluated == 8
    assert len(result.counterexamples) == 3
    assert result.counterexamples[0].objective <= result.counterexamples[-1].objective
    assert result.counterexamples[0].manifest["schema"] == "neuros.synthetic_bci_arena.manifest.v2"
