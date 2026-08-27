"""Model-aware qualification facade for measured display evidence.

The low-level :mod:`neuros.arena.display_evidence` module owns observation
provenance, transition detection, and numerical comparison. This module owns the
interpretation boundary between those measurements and Arena's synthetic display
model so every public qualification artifact states exactly which model and clock
it challenged.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .display_evidence import (
    DISPLAY_OBSERVATION_SCHEMA,
    DISPLAY_QUALIFICATION_SCHEMA,
    DISPLAY_TRANSITION_DETECTOR,
    EVIDENCE_CLASSES,
    DisplayObservation,
    DisplayQualificationConfig,
    DisplayQualificationResult,
    DisplayTransitionTrace,
    TransitionDetectionConfig,
    detect_display_transitions,
    load_display_observation_csv,
    qualify_display_observation as _qualify_display_observation,
    save_display_qualification as _save_display_qualification,
)
from .presentation import PresentationEpoch


EPOCH_ZERO_SEMANTICS = "observation_timestamp_corresponding_to_presentation_command_epoch_zero"
PLANNED_CLOCK_DOMAIN = "modeled_display_emission_seconds_relative_to_command_epoch_zero"
RESIDUAL_SEMANTICS = "measured_transition_minus_modeled_emission_transition"


def _model_summary(epoch: PresentationEpoch) -> dict[str, object]:
    trace = epoch.trace
    command_start = (
        float(trace.command_frame_times_s[0])
        if trace.command_frame_times_s.size
        else 0.0
    )
    first_emission = (
        float(trace.frame_times_s[0])
        if trace.frame_times_s.size
        else 0.0
    )
    return {
        "planned_display_trace_model": trace.model,
        "planned_clock_reference": "presentation_command_epoch_zero",
        "planned_command_start_s": command_start,
        "planned_first_emission_s": first_emission,
        "planned_modeled_response_lag_ms": float((first_emission - command_start) * 1000.0),
    }


def qualify_display_observation(
    epoch: PresentationEpoch,
    observation: DisplayObservation,
    config: DisplayQualificationConfig | None = None,
) -> DisplayQualificationResult:
    """Qualify an observation and preserve model/clock provenance.

    ``epoch_zero_s`` is interpreted as the observation-clock timestamp
    corresponding to Arena's presentation *command* epoch ``t=0``. Planned
    transition timestamps come from the synthetic display trace's modeled
    physical-emission clock. Therefore aligned residuals are

    ``measured transition - modeled emission transition``.

    They are model-comparison residuals, not independent measurements of monitor
    response latency. The physical observation remains evidence authority.
    """

    result = _qualify_display_observation(epoch, observation, config)
    epoch_summary = dict(result.epoch)
    epoch_summary.update(_model_summary(epoch))

    aligned = None
    if result.aligned_comparison is not None:
        aligned = dict(result.aligned_comparison)
        aligned.update({
            "epoch_zero_semantics": EPOCH_ZERO_SEMANTICS,
            "planned_clock_domain": PLANNED_CLOCK_DOMAIN,
            "residual_semantics": RESIDUAL_SEMANTICS,
        })

    return replace(
        result,
        epoch=epoch_summary,
        aligned_comparison=aligned,
    )


def save_display_qualification(
    result: DisplayQualificationResult,
    path: str | Path,
) -> Path:
    """Persist a model-aware qualification result."""

    return _save_display_qualification(result, path)


__all__ = [
    "DISPLAY_OBSERVATION_SCHEMA",
    "DISPLAY_QUALIFICATION_SCHEMA",
    "DISPLAY_TRANSITION_DETECTOR",
    "EPOCH_ZERO_SEMANTICS",
    "EVIDENCE_CLASSES",
    "PLANNED_CLOCK_DOMAIN",
    "RESIDUAL_SEMANTICS",
    "DisplayObservation",
    "DisplayQualificationConfig",
    "DisplayQualificationResult",
    "DisplayTransitionTrace",
    "TransitionDetectionConfig",
    "detect_display_transitions",
    "load_display_observation_csv",
    "qualify_display_observation",
    "save_display_qualification",
]
