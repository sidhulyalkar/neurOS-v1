"""Deterministic participant-state traces for Synthetic BCI Arena.

The first participant-response contract is deliberately narrow: it models
frequency-target visual-attention dynamics for SSVEP-style worlds. It does not
pretend that P300, motor-imagery, auditory or other paradigms share the same
participant dynamics merely because they use the same ``WorldInputBlock``.

The output is causal ground truth for a declared synthetic world, not a claim
that human visual attention follows this exact dynamical system.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .specs import ArenaScenario, ParticipantProfile


PARTICIPANT_RESPONSE_MODEL = "neuros.arena.frequency_target_response.v1"
PARTICIPANT_RESPONSE_SCOPE = "frequency_target_visual_attention"


@dataclass(frozen=True)
class ParticipantStateTrace:
    """Resolved per-sample frequency-target participant state.

    ``target_switch`` is retained as the compact field name for this contract,
    but its semantics are target transitions: it is true when an active frequency
    target first appears or whenever target identity changes, including
    active→rest and rest→active boundaries. Initial rest is not a transition.
    """

    attention_gain: np.ndarray
    requested_attention_gain: np.ndarray
    target_frequency_hz: np.ndarray
    target_switch: np.ndarray
    sampling_rate_hz: float
    model: str = PARTICIPANT_RESPONSE_MODEL

    def validate(self) -> None:
        arrays = (
            self.attention_gain,
            self.requested_attention_gain,
            self.target_frequency_hz,
            self.target_switch,
        )
        sizes = {np.asarray(array).size for array in arrays}
        if len(sizes) != 1:
            raise ValueError("participant trace arrays must have equal length")
        if self.sampling_rate_hz <= 0 or not np.isfinite(self.sampling_rate_hz):
            raise ValueError("sampling_rate_hz must be positive and finite")
        if not np.all(np.isfinite(self.attention_gain)):
            raise ValueError("attention_gain trace must be finite")
        if not np.all(np.isfinite(self.requested_attention_gain)):
            raise ValueError("requested_attention_gain trace must be finite")
        if np.any(self.attention_gain < 0) or np.any(self.requested_attention_gain < 0):
            raise ValueError("participant attention gains must be non-negative")
        target = np.asarray(self.target_frequency_hz, dtype=float)
        finite_or_nan = np.isfinite(target) | np.isnan(target)
        if not np.all(finite_or_nan):
            raise ValueError("target_frequency_hz trace may contain only finite values or NaN rest markers")
        transition = np.asarray(self.target_switch)
        if transition.dtype.kind not in {"b", "i", "u"}:
            raise ValueError("target_switch trace must be boolean/integer-like")

    def to_summary(self) -> dict[str, object]:
        self.validate()
        transitions = np.flatnonzero(np.asarray(self.target_switch, dtype=bool))
        target = np.asarray(self.target_frequency_hz, dtype=float)
        transition_samples = [int(value) for value in transitions.tolist()]
        return {
            "model": self.model,
            "scope": PARTICIPANT_RESPONSE_SCOPE,
            "samples": int(np.asarray(self.attention_gain).size),
            "sampling_rate_hz": float(self.sampling_rate_hz),
            "target_transition_samples": transition_samples,
            "target_switch_samples": transition_samples,
            "peak_requested_attention_gain": float(np.max(self.requested_attention_gain, initial=0.0)),
            "peak_effective_attention_gain": float(np.max(self.attention_gain, initial=0.0)),
            "mean_effective_attention_gain": float(np.mean(self.attention_gain)) if self.attention_gain.size else 0.0,
            "active_target_fraction": float(np.mean(np.isfinite(target))) if target.size else 0.0,
        }


def _stage_sample_count(duration_s: float, sampling_rate_hz: float) -> int:
    return max(1, int(round(float(duration_s) * float(sampling_rate_hz))))


def _delay_sample_count(delay_s: float, sampling_rate_hz: float) -> int:
    """Return the first eligible sample count at-or-after a declared delay."""

    # Delay is a causal lower bound. `round` could make a non-grid delay shorter
    # than requested, e.g. 41 ms at 250 Hz -> 10 samples = 40 ms.
    return max(0, int(np.ceil(float(delay_s) * float(sampling_rate_hz) - 1e-12)))


def compile_participant_state_trace(
    scenario: ArenaScenario,
    participant: ParticipantProfile,
    sampling_rate_hz: float,
) -> ParticipantStateTrace:
    """Compile frequency-target response dynamics onto the source sample clock.

    Policy:

    * only stages with ``target_frequency_hz`` participate in this v1 response
      model; non-frequency paradigms receive zero values from this specific
      stream and remain free to define their own participant semantics;
    * stage ``attention_gain`` is the requested control magnitude;
    * ``gaze_duty_cycle`` is represented as a deterministic attenuation factor,
      not as a claim about literal eye-open sample occupancy;
    * global response attenuation scales that request with elapsed source time;
    * frequency-target identity changes reset effective attention and start the
      declared response delay;
    * the first non-zero response sample can occur only at-or-after that delay;
    * after the delay, a first-order state approaches the current request using
      ``switch_time_constant_s``;
    * adjacent stages with the same target preserve response state rather than
      restarting it because of scenario segmentation;
    * rest/no-frequency-target samples revoke this response immediately.

    These are transparent synthetic assumptions, not physiological population
    estimates or a general participant model for all BCI paradigms.
    """

    scenario.validate()
    participant.validate()
    fs = float(sampling_rate_hz)
    if fs <= 0 or not np.isfinite(fs):
        raise ValueError("sampling_rate_hz must be positive and finite")

    total_samples = sum(_stage_sample_count(stage.duration_s, fs) for stage in scenario.stages)
    effective = np.zeros(total_samples, dtype=float)
    requested = np.zeros(total_samples, dtype=float)
    target_trace = np.full(total_samples, np.nan, dtype=float)
    transition_trace = np.zeros(total_samples, dtype=bool)

    delay_samples = _delay_sample_count(participant.response_delay_s, fs)
    alpha = min(1.0, 1.0 / (fs * participant.switch_time_constant_s))
    current_target: float | None = None
    response_state = 0.0
    delay_remaining = 0
    cursor = 0
    initialized = False

    for stage in scenario.stages:
        count = _stage_sample_count(stage.duration_s, fs)
        target = None if stage.target_frequency_hz is None else float(stage.target_frequency_hz)
        for local_index in range(count):
            sample_index = cursor + local_index
            if not initialized:
                initialized = True
                current_target = target
                if target is not None:
                    response_state = 0.0
                    delay_remaining = delay_samples
                    transition_trace[sample_index] = True
            elif target != current_target:
                current_target = target
                response_state = 0.0
                delay_remaining = delay_samples if target is not None else 0
                transition_trace[sample_index] = True

            if target is None:
                requested_gain = 0.0
                response_state = 0.0
            else:
                global_time_s = sample_index / fs
                attenuation = max(
                    0.0,
                    1.0
                    - participant.response_attenuation_per_minute
                    * (global_time_s / 60.0),
                )
                requested_gain = (
                    float(stage.attention_gain)
                    * float(participant.gaze_duty_cycle)
                    * attenuation
                )
                if delay_remaining > 0:
                    response_state = 0.0
                    delay_remaining -= 1
                else:
                    response_state += (requested_gain - response_state) * alpha

            requested[sample_index] = requested_gain
            effective[sample_index] = response_state
            if target is not None:
                target_trace[sample_index] = target
        cursor += count

    trace = ParticipantStateTrace(
        attention_gain=effective.astype(np.float32),
        requested_attention_gain=requested.astype(np.float32),
        target_frequency_hz=target_trace,
        target_switch=transition_trace,
        sampling_rate_hz=fs,
    )
    trace.validate()
    return trace
