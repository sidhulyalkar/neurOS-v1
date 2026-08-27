"""Versionable manifests for the Synthetic BCI Arena."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


def _validate_scalar_mapping(name: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError(f"{name}.{key} must be a JSON scalar")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError(f"{name}.{key} must be finite")


@dataclass(frozen=True)
class ParticipantProfile:
    name: str = "strong-responder"
    seed: int = 7
    colored_noise_uv: float = 4.5
    white_noise_uv: float = 1.25
    alpha_frequency_hz: float = 9.4
    alpha_amplitude_uv: float = 2.8
    ssvep_amplitude_uv: float = 8.0
    first_harmonic_ratio: float = 0.34
    response_delay_s: float = 0.18
    switch_time_constant_s: float = 0.35
    response_attenuation_per_minute: float = 0.0
    gaze_duty_cycle: float = 1.0
    artifact_gain: float = 1.0

    def validate(self) -> None:
        if not self.name:
            raise ValueError("participant name is required")
        numeric = (
            self.colored_noise_uv,
            self.white_noise_uv,
            self.alpha_frequency_hz,
            self.alpha_amplitude_uv,
            self.ssvep_amplitude_uv,
            self.first_harmonic_ratio,
            self.response_delay_s,
            self.switch_time_constant_s,
            self.response_attenuation_per_minute,
            self.gaze_duty_cycle,
            self.artifact_gain,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("participant numeric parameters must be finite")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
            raise ValueError("participant seed must be a non-negative integer")
        if self.colored_noise_uv < 0 or self.white_noise_uv < 0:
            raise ValueError("noise amplitudes must be non-negative")
        if self.alpha_frequency_hz <= 0 or self.alpha_amplitude_uv < 0:
            raise ValueError("alpha parameters are invalid")
        if self.ssvep_amplitude_uv < 0:
            raise ValueError("ssvep amplitude must be non-negative")
        if not 0 <= self.first_harmonic_ratio <= 2:
            raise ValueError("first_harmonic_ratio must be in [0, 2]")
        if self.response_delay_s < 0 or self.switch_time_constant_s <= 0:
            raise ValueError("response timing must be non-negative/positive")
        if not 0 <= self.response_attenuation_per_minute <= 1:
            raise ValueError("response attenuation must be in [0, 1]")
        if not 0 <= self.gaze_duty_cycle <= 1:
            raise ValueError("gaze_duty_cycle must be in [0, 1]")
        if self.artifact_gain < 0:
            raise ValueError("artifact_gain must be non-negative")


@dataclass(frozen=True)
class WorldModelProfile:
    """Selects the neural dynamics implementation independently of the world."""

    name: str = "driven_state_space"
    parameters: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name:
            raise ValueError("world model name is required")
        _validate_scalar_mapping("world_model.parameters", self.parameters)


@dataclass(frozen=True)
class DeviceProfile:
    name: str = "unicorn-like"
    sampling_rate_hz: float = 250.0
    channel_names: tuple[str, ...] = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
    adc_bits: int = 24
    input_range_uv: float = 450_000.0
    sensor_noise_uv: float = 0.35
    line_frequency_hz: float = 60.0
    line_noise_uv: float = 0.8
    clock_offset_ms: float = 0.0
    clock_drift_ppm: float = 0.0
    timestamp_jitter_ms: float = 0.0
    chunk_samples: int = 5

    def validate(self) -> None:
        numeric = (
            self.sampling_rate_hz,
            self.input_range_uv,
            self.sensor_noise_uv,
            self.line_frequency_hz,
            self.line_noise_uv,
            self.clock_offset_ms,
            self.clock_drift_ppm,
            self.timestamp_jitter_ms,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("device numeric parameters must be finite")
        if self.sampling_rate_hz <= 0 or not self.channel_names:
            raise ValueError("device sample rate/channels are invalid")
        if any(not isinstance(name, str) or not name for name in self.channel_names):
            raise ValueError("device channel names must be non-empty strings")
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("device channel names must be unique")
        if isinstance(self.adc_bits, (bool, np.bool_)) or not isinstance(self.adc_bits, (int, np.integer)):
            raise ValueError("adc_bits must be an integer")
        if self.adc_bits < 8 or self.adc_bits > 32:
            raise ValueError("adc_bits must be in [8, 32]")
        if isinstance(self.chunk_samples, (bool, np.bool_)) or not isinstance(self.chunk_samples, (int, np.integer)):
            raise ValueError("chunk_samples must be an integer")
        if self.input_range_uv <= 0 or self.sensor_noise_uv < 0 or self.line_noise_uv < 0:
            raise ValueError("device amplitude parameters are invalid")
        if self.line_frequency_hz <= 0 or self.chunk_samples <= 0:
            raise ValueError("device line frequency/chunk size are invalid")
        if self.timestamp_jitter_ms < 0:
            raise ValueError("timestamp_jitter_ms must be non-negative")


@dataclass(frozen=True)
class DisplayProfile:
    name: str = "120hz-fixed"
    refresh_hz: float = 120.0
    response_lag_ms: float = 3.0
    frame_jitter_ms: float = 0.0
    frame_drop_probability: float = 0.0
    low_luminance: float = 0.18
    high_luminance: float = 1.0

    def validate(self) -> None:
        numeric = (
            self.refresh_hz,
            self.response_lag_ms,
            self.frame_jitter_ms,
            self.frame_drop_probability,
            self.low_luminance,
            self.high_luminance,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("display numeric parameters must be finite")
        if self.refresh_hz <= 0 or self.response_lag_ms < 0 or self.frame_jitter_ms < 0:
            raise ValueError("display timing parameters are invalid")
        if not 0 <= self.frame_drop_probability < 1:
            raise ValueError("frame_drop_probability must be in [0, 1)")
        if not 0 <= self.low_luminance < self.high_luminance <= 1:
            raise ValueError("luminance levels must satisfy 0 <= low < high <= 1")


@dataclass(frozen=True)
class TransportProfile:
    name: str = "localhost-clean"
    drop_probability: float = 0.0
    jitter_ms: float = 0.0
    reorder_probability: float = 0.0
    silence_windows: tuple[tuple[float, float], ...] = ()
    clock_correction_offset_error_ms: float = 0.0
    clock_correction_drift_error_ppm: float = 0.0
    clock_correction_noise_ms: float = 0.0

    def validate(self) -> None:
        numeric = (
            self.drop_probability,
            self.jitter_ms,
            self.reorder_probability,
            self.clock_correction_offset_error_ms,
            self.clock_correction_drift_error_ppm,
            self.clock_correction_noise_ms,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("transport numeric parameters must be finite")
        if not 0 <= self.drop_probability < 1 or not 0 <= self.reorder_probability < 1:
            raise ValueError("transport probabilities must be in [0, 1)")
        if self.jitter_ms < 0 or self.clock_correction_noise_ms < 0:
            raise ValueError("transport and clock-correction jitter must be non-negative")
        for start, duration in self.silence_windows:
            if not np.isfinite(start) or not np.isfinite(duration):
                raise ValueError("silence windows must be finite")
            if start < 0 or duration <= 0:
                raise ValueError("silence windows require start >= 0 and duration > 0")


@dataclass(frozen=True)
class ArtifactEvent:
    at_s: float
    kind: str
    duration_s: float = 0.35
    severity: float = 1.0
    event_id: str | None = None
    channels: tuple[str, ...] | None = None
    seed: int | None = None

    def validate(self, stage_duration_s: float) -> None:
        if not all(np.isfinite(value) for value in (self.at_s, self.duration_s, self.severity)):
            raise ValueError("artifact timing/severity must be finite")
        if not 0 <= self.at_s < stage_duration_s:
            raise ValueError("artifact onset must fall inside its stage")
        if self.duration_s <= 0 or self.severity < 0:
            raise ValueError("artifact duration/severity are invalid")
        if not isinstance(self.kind, str) or not self.kind:
            raise ValueError("artifact kind must be a non-empty string")
        if self.event_id is not None and (not isinstance(self.event_id, str) or not self.event_id.strip()):
            raise ValueError("artifact event_id must be a non-empty string when supplied")
        if self.channels is not None:
            if not self.channels or any(not isinstance(name, str) or not name for name in self.channels):
                raise ValueError("artifact channels must contain non-empty channel names")
            if len(set(self.channels)) != len(self.channels):
                raise ValueError("artifact channels must be unique")
        if self.seed is not None:
            if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
                raise ValueError("artifact seed must be a non-negative integer")


@dataclass(frozen=True)
class StageSpec:
    label: str
    duration_s: float
    target_frequency_hz: float | None = None
    attention_gain: float = 1.0
    artifacts: tuple[ArtifactEvent, ...] = ()
    target: dict[str, Any] = field(default_factory=dict)
    task_state: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.label or not np.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("stage label/duration are required")
        if self.target_frequency_hz is not None:
            if not np.isfinite(self.target_frequency_hz) or self.target_frequency_hz <= 0:
                raise ValueError("target frequency must be positive and finite")
        if not np.isfinite(self.attention_gain) or self.attention_gain < 0:
            raise ValueError("attention gain must be non-negative and finite")
        for event in self.artifacts:
            event.validate(self.duration_s)
        _validate_scalar_mapping("stage.target", self.target)
        _validate_scalar_mapping("stage.task_state", self.task_state)

        # `frequency_hz` inside generic target metadata is a reserved mirror of
        # the typed frequency-target authority. Allowing it without the typed
        # field would let rich world models see a target while participant state
        # and Arena ground truth still describe rest.
        if "frequency_hz" in self.target:
            if self.target_frequency_hz is None:
                raise ValueError(
                    "stage.target.frequency_hz is reserved; set authoritative target_frequency_hz instead"
                )
            metadata_frequency = self.target["frequency_hz"]
            if isinstance(metadata_frequency, bool) or not isinstance(metadata_frequency, (int, float, np.integer, np.floating)):
                raise ValueError("stage.target.frequency_hz must be numeric when target_frequency_hz is set")
            if not np.isfinite(float(metadata_frequency)) or not np.isclose(
                float(metadata_frequency),
                float(self.target_frequency_hz),
                rtol=0.0,
                atol=1e-12,
            ):
                raise ValueError(
                    "stage.target.frequency_hz conflicts with authoritative target_frequency_hz"
                )


@dataclass(frozen=True)
class ArenaScenario:
    name: str
    stages: tuple[StageSpec, ...]
    seed: int = 7
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or not self.stages:
            raise ValueError("scenario name and at least one stage are required")
        if isinstance(self.seed, (bool, np.bool_)) or not isinstance(self.seed, (int, np.integer)) or int(self.seed) < 0:
            raise ValueError("scenario seed must be a non-negative integer")
        for stage in self.stages:
            stage.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ArenaScenario":
        stages = []
        for stage_raw in raw.get("stages", []):
            artifacts = []
            for event_raw in stage_raw.get("artifacts", []):
                event_values = dict(event_raw)
                if event_values.get("channels") is not None:
                    event_values["channels"] = tuple(str(value) for value in event_values["channels"])
                artifacts.append(ArtifactEvent(**event_values))
            stages.append(StageSpec(
                label=stage_raw["label"],
                duration_s=float(stage_raw["duration_s"]),
                target_frequency_hz=(None if stage_raw.get("target_frequency_hz") is None else float(stage_raw["target_frequency_hz"])),
                attention_gain=float(stage_raw.get("attention_gain", 1.0)),
                artifacts=tuple(artifacts),
                target=dict(stage_raw.get("target", {})),
                task_state=dict(stage_raw.get("task_state", {})),
            ))
        scenario = cls(name=str(raw["name"]), stages=tuple(stages), seed=int(raw.get("seed", 7)), metadata=dict(raw.get("metadata", {})))
        scenario.validate()
        return scenario
