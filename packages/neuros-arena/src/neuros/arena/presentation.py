"""Physical stimulus-epoch compilation for Synthetic BCI Arena.

Scenario stages are authoring/task structure. They are not automatically physical
stimulus boundaries. This module compiles adjacent stages onto explicit display
presentation epochs so a label split cannot restart display lag, frame RNG or
coded phase unless the physical stimulus identity changes or the scenario asks
for a retrigger.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulators import StimulusTrace, simulate_stimulus
from .specs import ArenaScenario, DisplayProfile, StageSpec


PRESENTATION_EPOCH_MODEL = "neuros.arena.presentation_epochs.v1"


@dataclass(frozen=True)
class PresentationEpoch:
    """One uninterrupted physical display presentation."""

    index: int
    start_sample: int
    end_sample: int
    target_frequency_hz: float | None
    stimulus_id: str | None
    stage_indices: tuple[int, ...]
    trace: StimulusTrace

    @property
    def sample_count(self) -> int:
        return self.end_sample - self.start_sample

    def to_summary(self, scenario: ArenaScenario, sampling_rate_hz: float) -> dict[str, object]:
        command_start = (
            float(self.trace.command_frame_times_s[0])
            if self.trace.command_frame_times_s.size
            else 0.0
        )
        first_emission = (
            float(self.trace.frame_times_s[0])
            if self.trace.frame_times_s.size
            else 0.0
        )
        return {
            "epoch_index": int(self.index),
            "model": PRESENTATION_EPOCH_MODEL,
            "display_trace_model": self.trace.model,
            "start_sample": int(self.start_sample),
            "end_sample": int(self.end_sample),
            "resolved_start_s": float(self.start_sample / sampling_rate_hz),
            "resolved_end_s": float(self.end_sample / sampling_rate_hz),
            "resolved_duration_s": float(self.sample_count / sampling_rate_hz),
            "command_start_s": command_start,
            "first_emission_s": first_emission,
            "modeled_response_lag_ms": float((first_emission - command_start) * 1000.0),
            "target_frequency_hz": self.target_frequency_hz,
            "stimulus_id": self.stimulus_id,
            "stage_indices": [int(value) for value in self.stage_indices],
            "stages": [scenario.stages[index].label for index in self.stage_indices],
            "observed_frequency_hz": float(self.trace.observed_frequency_hz),
            "frequency_error_hz": (
                0.0
                if self.target_frequency_hz is None
                else float(abs(self.trace.observed_frequency_hz - self.target_frequency_hz))
            ),
            "frame_drop_fraction": float(self.trace.frame_drop_fraction),
            "interval_jitter_ms": float(self.trace.interval_jitter_ms),
        }


@dataclass(frozen=True)
class PresentationPlan:
    """Resolved presentation epochs plus a stage-to-epoch lookup."""

    epochs: tuple[PresentationEpoch, ...]
    stage_epoch_index: tuple[int, ...]
    sampling_rate_hz: float
    model: str = PRESENTATION_EPOCH_MODEL

    def validate(self, scenario: ArenaScenario) -> None:
        if len(self.stage_epoch_index) != len(scenario.stages):
            raise ValueError("presentation plan must map every scenario stage")
        if self.sampling_rate_hz <= 0 or not np.isfinite(self.sampling_rate_hz):
            raise ValueError("presentation sampling rate must be positive and finite")
        if not self.epochs:
            raise ValueError("presentation plan requires at least one epoch")
        cursor = 0
        for expected_index, epoch in enumerate(self.epochs):
            if epoch.index != expected_index:
                raise ValueError("presentation epoch indices must be contiguous")
            if epoch.start_sample != cursor or epoch.end_sample <= epoch.start_sample:
                raise ValueError("presentation epochs must tile the source sample clock")
            if not epoch.stage_indices:
                raise ValueError("presentation epoch must own at least one stage")
            cursor = epoch.end_sample
        if any(index < 0 or index >= len(self.epochs) for index in self.stage_epoch_index):
            raise ValueError("stage-to-presentation mapping contains an invalid epoch index")

    def epoch_for_stage(self, stage_index: int) -> PresentationEpoch:
        return self.epochs[self.stage_epoch_index[stage_index]]

    def to_summary(self, scenario: ArenaScenario) -> dict[str, object]:
        self.validate(scenario)
        return {
            "model": self.model,
            "epoch_count": len(self.epochs),
            "stage_epoch_index": [int(value) for value in self.stage_epoch_index],
            "epochs": [
                epoch.to_summary(scenario, self.sampling_rate_hz)
                for epoch in self.epochs
            ],
        }


def _stage_sample_count(stage: StageSpec, sampling_rate_hz: float) -> int:
    return max(1, int(round(float(stage.duration_s) * float(sampling_rate_hz))))


def _same_frequency(first: float | None, second: float | None) -> bool:
    if first is None or second is None:
        return first is None and second is None
    return bool(np.isclose(float(first), float(second), rtol=0.0, atol=1e-12))


def _same_physical_stimulus(
    frequency_hz: float | None,
    stimulus_id: str | None,
    stage: StageSpec,
) -> bool:
    return _same_frequency(frequency_hz, stage.target_frequency_hz) and stimulus_id == stage.stimulus_id


def compile_presentation_plan(
    scenario: ArenaScenario,
    display: DisplayProfile,
    sampling_rate_hz: float,
) -> PresentationPlan:
    """Compile task stages into physical presentation epochs.

    A new epoch begins when:

    * target frequency changes;
    * ``stimulus_id`` changes; or
    * ``stimulus_retrigger=True`` explicitly requests a new presentation.

    Otherwise adjacent stages share the same display trace. Their labels,
    attention gain, task metadata and artifact schedule may still change without
    causing a physical display restart.

    The display trace duration is resolved from the source sample clock, not the
    requested floating stage duration. Epoch RNG seeds depend on physical epoch
    order rather than authoring-stage count, making label-only stage splitting
    presentation invariant when it preserves the same resolved sample timeline.
    """

    scenario.validate()
    display.validate()
    fs = float(sampling_rate_hz)
    if fs <= 0 or not np.isfinite(fs):
        raise ValueError("sampling_rate_hz must be positive and finite")

    raw_epochs: list[dict[str, object]] = []
    stage_epoch_index: list[int] = []
    global_sample = 0

    for stage_index, stage in enumerate(scenario.stages):
        samples = _stage_sample_count(stage, fs)
        new_epoch = not raw_epochs
        if raw_epochs:
            current = raw_epochs[-1]
            new_epoch = bool(stage.stimulus_retrigger) or not _same_physical_stimulus(
                current["target_frequency_hz"],
                current["stimulus_id"],
                stage,
            )
        if new_epoch:
            raw_epochs.append({
                "start_sample": global_sample,
                "end_sample": global_sample + samples,
                "target_frequency_hz": stage.target_frequency_hz,
                "stimulus_id": stage.stimulus_id,
                "stage_indices": [stage_index],
            })
        else:
            current = raw_epochs[-1]
            current["end_sample"] = global_sample + samples
            current["stage_indices"].append(stage_index)
        stage_epoch_index.append(len(raw_epochs) - 1)
        global_sample += samples

    epochs: list[PresentationEpoch] = []
    for epoch_index, raw in enumerate(raw_epochs):
        start_sample = int(raw["start_sample"])
        end_sample = int(raw["end_sample"])
        target_frequency_hz = raw["target_frequency_hz"]
        duration_s = (end_sample - start_sample) / fs
        trace = simulate_stimulus(
            target_frequency_hz,
            duration_s,
            display,
            seed=scenario.seed * 1009 + epoch_index,
        )
        epochs.append(PresentationEpoch(
            index=epoch_index,
            start_sample=start_sample,
            end_sample=end_sample,
            target_frequency_hz=target_frequency_hz,
            stimulus_id=raw["stimulus_id"],
            stage_indices=tuple(int(value) for value in raw["stage_indices"]),
            trace=trace,
        ))

    plan = PresentationPlan(
        epochs=tuple(epochs),
        stage_epoch_index=tuple(stage_epoch_index),
        sampling_rate_hz=fs,
    )
    plan.validate(scenario)
    return plan
