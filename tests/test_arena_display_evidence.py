from __future__ import annotations

import hashlib

import numpy as np
import pytest

from neuros.arena import ArenaScenario, DisplayProfile, StageSpec, compile_presentation_plan
from neuros.arena.display_qualification import (
    DISPLAY_QUALIFICATION_SCHEMA,
    EPOCH_ZERO_SEMANTICS,
    PLANNED_CLOCK_DOMAIN,
    RESIDUAL_SEMANTICS,
    DisplayObservation,
    DisplayQualificationConfig,
    TransitionDetectionConfig,
    detect_display_transitions,
    load_display_observation_csv,
    qualify_display_observation,
)


def _epoch(
    *,
    duration_s: float = 1.0,
    frequency_hz: float = 10.0,
    response_lag_ms: float = 0.0,
):
    scenario = ArenaScenario(
        "display-evidence",
        (
            StageSpec(
                "target",
                duration_s,
                frequency_hz,
                0.8,
                stimulus_id="sight-orb",
            ),
        ),
        seed=211,
    )
    display = DisplayProfile(
        refresh_hz=120.0,
        response_lag_ms=response_lag_ms,
        frame_jitter_ms=0.0,
        frame_drop_probability=0.0,
    )
    return compile_presentation_plan(scenario, display, 250.0).epochs[0], display


def _square_observation(
    frequency_hz: float = 10.0,
    *,
    duration_s: float = 1.0,
    sampling_rate_hz: float = 1000.0,
    timestamp_offset_s: float = 0.0,
    evidence_class: str = "synthetic_fixture",
) -> DisplayObservation:
    local = np.arange(0.0, duration_s, 1.0 / sampling_rate_hz)
    # Add deterministic low-amplitude analog-like noise without approaching the
    # Schmitt thresholds. The transition detector should recover the square code.
    square = np.where(np.sin(2.0 * np.pi * frequency_hz * local) >= 0.0, 4.0, 1.0)
    noise = 0.025 * np.sin(2.0 * np.pi * 37.0 * local + 0.31)
    return DisplayObservation(
        timestamps_s=local + timestamp_offset_s,
        luminance=square + noise,
        units="volts",
        source="synthetic-square",
        evidence_class=evidence_class,
    )


def _sample_planned_trace(epoch, *, sampling_rate_hz: float, timestamp_offset_s: float):
    local = np.arange(0.0, 1.0, 1.0 / sampling_rate_hz)
    trace = epoch.trace
    indices = np.searchsorted(trace.frame_times_s, local, side="right") - 1
    values = np.full(local.size, float(trace.luminance[0]), dtype=float)
    valid = indices >= 0
    values[valid] = trace.luminance[np.minimum(indices[valid], trace.luminance.size - 1)]
    return DisplayObservation(
        timestamps_s=local + timestamp_offset_s,
        luminance=values,
        units="volts",
        source="synthetic-planned-trace",
        evidence_class="synthetic_fixture",
    )


def test_schmitt_detector_recovers_noisy_ten_hz_square_wave():
    observation = _square_observation()
    trace = detect_display_transitions(observation)

    assert trace.transition_times_s.size >= 18
    assert trace.observed_frequency_hz == pytest.approx(10.0, rel=0.015)
    assert trace.contrast > 2.8
    assert set(trace.directions) == {"rising", "falling"}
    assert trace.interval_jitter_ms is not None
    assert trace.interval_jitter_ms < 2.0


def test_unaligned_capture_reports_frequency_but_refuses_timing_residuals():
    epoch, _ = _epoch()
    result = qualify_display_observation(epoch, _square_observation())
    payload = result.to_dict()

    assert payload["schema"] == DISPLAY_QUALIFICATION_SCHEMA
    assert payload["target_metrics"]["target_frequency_hz"] == 10.0
    assert payload["target_metrics"]["observed_frequency_hz"] == pytest.approx(10.0, rel=0.015)
    assert payload["epoch"]["planned_display_trace_model"] == "neuros.arena.display_trace.v2"
    assert payload["aligned_comparison"] is None
    assert "No epoch-zero alignment was supplied" in payload["evidence_boundary"]
    assert "not physical display evidence" in payload["evidence_boundary"]


def test_explicit_epoch_zero_enables_transition_timing_comparison():
    epoch, _ = _epoch()
    epoch_zero = 123.456
    observation = _sample_planned_trace(
        epoch,
        sampling_rate_hz=4000.0,
        timestamp_offset_s=epoch_zero,
    )
    result = qualify_display_observation(
        epoch,
        observation,
        DisplayQualificationConfig(
            epoch_zero_s=epoch_zero,
            transition_match_tolerance_s=0.002,
        ),
    )
    comparison = result.aligned_comparison

    assert comparison is not None
    assert comparison["clock_alignment"] == "explicit_epoch_zero"
    assert comparison["epoch_zero_semantics"] == EPOCH_ZERO_SEMANTICS
    assert comparison["planned_clock_domain"] == PLANNED_CLOCK_DOMAIN
    assert comparison["residual_semantics"] == RESIDUAL_SEMANTICS
    assert comparison["missed_transition_count"] == 0
    assert comparison["extra_transition_count"] == 0
    assert comparison["matched_transition_count"] >= 18
    assert comparison["timing_residual_p95_abs_ms"] < 0.5
    assert comparison["transition_polarity_compared"] is False


def test_qualification_names_display_model_and_declared_response_lag():
    epoch, _ = _epoch(response_lag_ms=17.0)
    epoch_zero = 80.0
    observation = _sample_planned_trace(
        epoch,
        sampling_rate_hz=4000.0,
        timestamp_offset_s=epoch_zero,
    )
    result = qualify_display_observation(
        epoch,
        observation,
        DisplayQualificationConfig(
            epoch_zero_s=epoch_zero,
            transition_match_tolerance_s=0.002,
        ),
    )
    payload = result.to_dict()

    assert payload["epoch"]["planned_display_trace_model"] == "neuros.arena.display_trace.v2"
    assert payload["epoch"]["planned_clock_reference"] == "presentation_command_epoch_zero"
    assert payload["epoch"]["planned_command_start_s"] == pytest.approx(0.0)
    assert payload["epoch"]["planned_first_emission_s"] == pytest.approx(0.017)
    assert payload["epoch"]["planned_modeled_response_lag_ms"] == pytest.approx(17.0)
    assert payload["aligned_comparison"]["timing_residual_p95_abs_ms"] < 0.5


def test_wrong_epoch_zero_is_visible_as_timing_residual_not_silently_realigned():
    epoch, _ = _epoch()
    true_zero = 50.0
    observation = _sample_planned_trace(
        epoch,
        sampling_rate_hz=4000.0,
        timestamp_offset_s=true_zero,
    )
    result = qualify_display_observation(
        epoch,
        observation,
        DisplayQualificationConfig(
            epoch_zero_s=true_zero - 0.004,
            transition_match_tolerance_s=0.010,
        ),
    )
    comparison = result.aligned_comparison
    assert comparison is not None
    assert comparison["timing_residual_mean_ms"] == pytest.approx(4.0, abs=0.35)


def test_csv_loader_preserves_source_and_content_hashes(tmp_path):
    path = tmp_path / "capture.csv"
    raw = "timestamp_s,luminance\n0.000,1\n0.010,1\n0.020,4\n0.030,4\n0.040,1\n"
    path.write_text(raw, encoding="utf-8")

    observation = load_display_observation_csv(
        path,
        units="adc_count",
        evidence_class="synthetic_fixture",
    )
    provenance = observation.provenance_dict()

    assert provenance["source_sha256"] == hashlib.sha256(raw.encode("utf-8")).hexdigest()
    assert len(provenance["content_sha256"]) == 64
    assert provenance["units"] == "adc_count"
    assert provenance["samples"] == 5


def test_observation_validation_rejects_timestamp_and_provenance_ambiguity():
    with pytest.raises(ValueError, match="strictly increasing"):
        DisplayObservation(
            np.asarray([0.0, 0.1, 0.1, 0.2]),
            np.asarray([1.0, 2.0, 1.0, 2.0]),
        ).validate()
    with pytest.raises(ValueError, match="evidence_class"):
        DisplayObservation(
            np.asarray([0.0, 0.1, 0.2, 0.3]),
            np.asarray([1.0, 2.0, 1.0, 2.0]),
            evidence_class="definitely-physical-trust-me",
        ).validate()


def test_detector_fails_closed_when_capture_has_no_resolved_low_high_states():
    times = np.arange(0.0, 0.5, 0.001)
    observation = DisplayObservation(
        timestamps_s=times,
        luminance=np.full(times.size, 2.0),
        units="volts",
        evidence_class="synthetic_fixture",
    )
    with pytest.raises(ValueError, match="insufficient low/high contrast"):
        detect_display_transitions(observation)


def test_minimum_transition_separation_suppresses_chatter():
    times = np.arange(0.0, 0.4, 0.001)
    base = np.where(np.sin(2 * np.pi * 10.0 * times) >= 0, 4.0, 1.0)
    # Two brief excursions around one true transition create chatter if no
    # minimum separation is requested.
    values = base.copy()
    values[50:52] = 1.0
    values[52:54] = 4.0
    observation = DisplayObservation(
        times,
        values,
        units="volts",
        evidence_class="synthetic_fixture",
    )
    unguarded = detect_display_transitions(observation)
    guarded = detect_display_transitions(
        observation,
        TransitionDetectionConfig(minimum_transition_separation_s=0.010),
    )
    assert guarded.transition_times_s.size <= unguarded.transition_times_s.size


def test_photodiode_evidence_boundary_is_physical_display_only():
    epoch, _ = _epoch()
    observation = _square_observation(evidence_class="measured_photodiode")
    result = qualify_display_observation(epoch, observation)
    assert "physical display-emission claims for this captured setup only" in result.evidence_boundary
    assert "does not establish human neural response" in result.evidence_boundary
