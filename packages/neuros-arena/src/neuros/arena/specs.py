"""Versionable manifests for the Synthetic BCI Arena."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def _validate_scalar_mapping(name: str, values: dict[str, Any]) -> None:
    for key, value in values.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{name} keys must be non-empty strings")
        if not isinstance(value, (str, int, float, bool)) and value is not None:
            raise ValueError(f"{name}.{key} must be a JSON scalar")


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
        if self.colored_noise_uv < 0 or self.white_noise_uv < 0:
            raise ValueError("noise amplitudes must be non-negative")
        if self.alpha_frequency_hz <= 0 or self.alpha_amplitude_uv < 0:
            raise ValueError("alpha parameters are invalid")
        if self.ssvep_amplitude_uv < 0:
            raise ValueError("ssvep amplitude must be non-negative")
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
        if self.sampling_rate_hz <= 0 or not self.channel_names:
            raise ValueError("device sample rate/channels are invalid")
        if self.adc_bits < 8 or self.adc_bits > 32:
            raise ValueError("adc_bits must be in [8, 32]")
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
        if not 0 <= self.drop_probability < 1 or not 0 <= self.reorder_probability < 1:
            raise ValueError("transport probabilities must be in [0, 1)")
        if self.jitter_ms < 0 or self.clock_correction_noise_ms < 0:
            raise ValueError("transport and clock-correction jitter must be non-negative")
        for start, duration in self.silence_windows:
            if start < 0 or duration <= 0:
                raise ValueError("silence windows require start >= 0 and duration > 0")


@dataclass(frozen=True)
class ArtifactEvent:
    at_s: float
    kind: str
    duration_s: float = 0.35
    severity: float = 1.0

    def validate(self, stage_duration_s: float) -> None:
        if not 0 <= self.at_s < stage_duration_s:
            raise ValueError("artifact onset must fall inside its stage")
        if self.duration_s <= 0 or self.severity < 0:
            raise ValueError("artifact duration/severity are invalid")


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
        if not self.label or self.duration_s <= 0:
            raise ValueError("stage label/duration are required")
        if self.target_frequency_hz is not None and self.target_frequency_hz <= 0:
            raise ValueError("target frequency must be positive")
        if self.attention_gain < 0:
            raise ValueError("attention gain must be non-negative")
        for event in self.artifacts:
            event.validate(self.duration_s)
        _validate_scalar_mapping("stage.target", self.target)
        _validate_scalar_mapping("stage.task_state", self.task_state)


@dataclass(frozen=True)
class ArenaScenario:
    name: str
    stages: tuple[StageSpec, ...]
    seed: int = 7
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.name or not self.stages:
            raise ValueError("scenario name and at least one stage are required")
        for stage in self.stages:
            stage.validate()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "ArenaScenario":
        stages = []
        for stage_raw in raw.get("stages", []):
            artifacts = tuple(ArtifactEvent(**event) for event in stage_raw.get("artifacts", []))
            stages.append(StageSpec(
                label=stage_raw["label"],
                duration_s=float(stage_raw["duration_s"]),
                target_frequency_hz=(None if stage_raw.get("target_frequency_hz") is None else float(stage_raw["target_frequency_hz"])),
                attention_gain=float(stage_raw.get("attention_gain", 1.0)),
                artifacts=artifacts,
                target=dict(stage_raw.get("target", {})),
                task_state=dict(stage_raw.get("task_state", {})),
            ))
        scenario = cls(name=str(raw["name"]), stages=tuple(stages), seed=int(raw.get("seed", 7)), metadata=dict(raw.get("metadata", {})))
        scenario.validate()
        return scenario
