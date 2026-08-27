"""Pluggable neural world models for closed-loop BCI simulation.

World models own the latent neural dynamics that map an emitted stimulus and
participant state into source/sensor-space EEG. Device/display/transport physics
remain separate Arena layers so a model can be swapped without changing the
rest of the systems test.

The built-in models are deliberately phenomenological. They provide known
causal ground truth for software qualification; they are not physiological
human digital twins.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from neuros.drivers.synthetic_eeg import (
    ArtifactEvent as DriverArtifactEvent,
    SyntheticEEGConfig,
    SyntheticEEGGenerator,
)

from .specs import ParticipantProfile
from .world_input import WorldInputBlock

SOURCE_CHANNELS = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
POSTERIOR_WEIGHTS = np.asarray([0.05, 0.05, 0.08, 0.05, 0.45, 0.85, 1.0, 0.85], dtype=float)
CENTRAL_WEIGHTS = np.asarray([0.18, 0.95, 1.0, 0.95, 0.25, 0.10, 0.08, 0.10], dtype=float)
FRONTAL_WEIGHTS = np.asarray([1.0, 0.30, 0.55, 0.30, 0.22, 0.12, 0.10, 0.12], dtype=float)


@dataclass(frozen=True)
class WorldModelEmission:
    """One emitted EEG block plus inspectable latent summaries."""

    data_uv: np.ndarray
    latent: dict[str, float]


class NeuralWorldModel(Protocol):
    """Minimal compatibility floor for Arena-compatible neural generators."""

    name: str
    channel_names: tuple[str, ...]

    def inject_artifact(self, kind: str, duration_seconds: float, severity: float) -> None: ...

    def render(
        self,
        sample_times_s: np.ndarray,
        emitted_stimulus: np.ndarray,
        target_frequency_hz: float | None,
        attention_gain: float,
    ) -> WorldModelEmission: ...


class LegacySyntheticWorldModel:
    """Compatibility adapter around the original neurOS synthetic EEG driver."""

    name = "legacy_synthetic"
    channel_names = SOURCE_CHANNELS

    def __init__(
        self,
        *,
        participant: ParticipantProfile,
        sampling_rate_hz: float,
        seed: int,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        del parameters
        self.participant = participant
        self.generator = SyntheticEEGGenerator(SyntheticEEGConfig(
            sampling_rate_hz=sampling_rate_hz,
            channel_names=SOURCE_CHANNELS,
            colored_noise_uv=participant.colored_noise_uv,
            white_noise_uv=participant.white_noise_uv,
            alpha_frequency_hz=participant.alpha_frequency_hz,
            alpha_amplitude_uv=participant.alpha_amplitude_uv,
            ssvep_amplitude_uv=participant.ssvep_amplitude_uv,
            first_harmonic_ratio=participant.first_harmonic_ratio,
            seed=seed,
        ))

    def inject_artifact(self, kind: str, duration_seconds: float, severity: float) -> None:
        self.generator.inject_artifact(kind, duration_seconds=duration_seconds, severity=severity)

    def schedule_artifact(
        self,
        kind: str,
        *,
        event_id: str,
        start_sample: int,
        duration_seconds: float,
        severity: float,
        channels: str | int | Sequence[str | int] | None = None,
        seed: int | None = None,
    ) -> DriverArtifactEvent:
        return self.generator.schedule_artifact(
            kind,
            event_id=event_id,
            start_sample=start_sample,
            duration_seconds=duration_seconds,
            severity=severity,
            channels=channels,
            seed=seed,
        )

    def render(
        self,
        sample_times_s: np.ndarray,
        emitted_stimulus: np.ndarray,
        target_frequency_hz: float | None,
        attention_gain: float,
    ) -> WorldModelEmission:
        del emitted_stimulus
        if target_frequency_hz is None:
            self.generator.set_attention(None)
        else:
            self.generator.set_attention(target_frequency_hz, attention_gain)
        block = self.generator.render(int(sample_times_s.size))
        return WorldModelEmission(
            data_uv=block.data_uv,
            latent={
                "attention_gain": float(attention_gain),
                "entrainment": float(attention_gain if target_frequency_hz is not None else 0.0),
                "stimulus_coupling": 0.0,
                "participant_stream_coupling": 0.0,
            },
        )


class _ArtifactEngine:
    """Arena adapter over the canonical v3 sample-indexed artifact renderer."""

    def __init__(self, sampling_rate_hz: float, seed: int) -> None:
        self.fs = float(sampling_rate_hz)
        self.generator = SyntheticEEGGenerator(SyntheticEEGConfig(
            sampling_rate_hz=self.fs,
            channel_names=SOURCE_CHANNELS,
            colored_noise_uv=0.0,
            white_noise_uv=0.0,
            alpha_amplitude_uv=0.0,
            ssvep_amplitude_uv=0.0,
            seed=int(seed),
        ))

    def inject(self, kind: str, duration_seconds: float, severity: float) -> None:
        self.generator.inject_artifact(kind, duration_seconds, severity)

    def schedule(
        self,
        kind: str,
        *,
        event_id: str,
        start_sample: int,
        duration_seconds: float,
        severity: float,
        channels: str | int | Sequence[str | int] | None = None,
        seed: int | None = None,
    ) -> DriverArtifactEvent:
        return self.generator.schedule_artifact(
            kind,
            event_id=event_id,
            start_sample=start_sample,
            duration_seconds=duration_seconds,
            severity=severity,
            channels=channels,
            seed=seed,
        )

    def render(
        self,
        times_s: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, tuple[DriverArtifactEvent, ...]]:
        samples = int(times_s.size)
        if samples == 0:
            return np.empty((8, 0), dtype=float), np.empty((8, 0), dtype=bool), ()
        block_start = self.generator.sample_index
        block = self.generator.render(samples)
        dropout_mask = np.zeros((8, samples), dtype=bool)
        for event in block.artifact_events:
            if event.kind != "dropout":
                continue
            overlap_start = max(block_start, event.start_sample)
            overlap_end = min(block_start + samples, event.end_sample)
            if overlap_start >= overlap_end:
                continue
            channels = event.channel_indices or (6,)
            dropout_mask[
                list(channels),
                overlap_start - block_start : overlap_end - block_start,
            ] = True
        return block.data_uv.astype(float), dropout_mask, block.artifact_events


class DrivenStateSpaceWorldModel:
    """Causal stochastic EEG state-space model driven by emitted luminance.

    The participant attention drive may be supplied per source sample through
    `WorldInputBlock.participant_streams`. The older scalar `render(...)` path is
    retained for compatibility and expands its scalar across the requested block.
    """

    name = "driven_state_space"
    channel_names = SOURCE_CHANNELS

    def __init__(
        self,
        *,
        participant: ParticipantProfile,
        sampling_rate_hz: float,
        seed: int,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        self.participant = participant
        self.fs = float(sampling_rate_hz)
        if self.fs <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        params = dict(parameters or {})
        self.resonance_damping = float(params.get("resonance_damping", 0.22))
        self.background_persistence = float(params.get("background_persistence", 0.985))
        self.entrainment_tau_s = float(params.get("entrainment_tau_s", 0.18))
        if not 0.02 <= self.resonance_damping <= 2.0:
            raise ValueError("resonance_damping must be in [0.02, 2]")
        if not 0.0 <= self.background_persistence < 1.0:
            raise ValueError("background_persistence must be in [0, 1)")
        if self.entrainment_tau_s <= 0:
            raise ValueError("entrainment_tau_s must be positive")
        self.rng = np.random.default_rng(seed)
        self._background = np.zeros(8, dtype=float)
        self._alpha_phase = self.rng.uniform(0.0, 2 * np.pi)
        self._alpha_channel_phase = self.rng.normal(0.0, 0.12, size=8)
        self._resonator_x = 0.0
        self._resonator_v = 0.0
        self._entrainment = 0.0
        self._artifact = _ArtifactEngine(self.fs, seed + 991)

    def inject_artifact(self, kind: str, duration_seconds: float, severity: float) -> None:
        self._artifact.inject(kind, duration_seconds, severity)

    def schedule_artifact(
        self,
        kind: str,
        *,
        event_id: str,
        start_sample: int,
        duration_seconds: float,
        severity: float,
        channels: str | int | Sequence[str | int] | None = None,
        seed: int | None = None,
    ) -> DriverArtifactEvent:
        return self._artifact.schedule(
            kind,
            event_id=event_id,
            start_sample=start_sample,
            duration_seconds=duration_seconds,
            severity=severity,
            channels=channels,
            seed=seed,
        )

    def _render_with_attention_stream(
        self,
        sample_times_s: np.ndarray,
        emitted_stimulus: np.ndarray,
        target_frequency_hz: float | None,
        attention_gain: np.ndarray,
    ) -> WorldModelEmission:
        times = np.asarray(sample_times_s, dtype=float)
        drive = np.asarray(emitted_stimulus, dtype=float)
        gain = np.asarray(attention_gain, dtype=float)
        if times.ndim != 1 or drive.shape != times.shape or gain.shape != times.shape:
            raise ValueError("time, emitted stimulus and attention gain must be matching 1-D arrays")
        if not np.all(np.isfinite(gain)) or np.any(gain < 0):
            raise ValueError("attention gain stream must be finite and non-negative")
        samples = times.size
        if samples == 0:
            return WorldModelEmission(
                np.empty((8, 0), dtype=np.float32),
                {"entrainment": self._entrainment, "participant_stream_coupling": 1.0},
            )
        dt = 1.0 / self.fs
        data = np.empty((8, samples), dtype=float)
        target_gain = gain if target_frequency_hz is not None else np.zeros(samples, dtype=float)
        alpha_omega = 2.0 * np.pi * self.participant.alpha_frequency_hz
        target_omega = 0.0 if target_frequency_hz is None else 2.0 * np.pi * float(target_frequency_hz)
        innovation_scale = np.sqrt(max(1.0 - self.background_persistence**2, 1e-9))
        for i in range(samples):
            self._background = (
                self.background_persistence * self._background
                + innovation_scale * self.rng.normal(size=8)
            )
            self._entrainment += (float(target_gain[i]) - self._entrainment) * min(1.0, dt / self.entrainment_tau_s)
            alpha = np.sin(alpha_omega * times[i] + self._alpha_phase + self._alpha_channel_phase)
            if target_frequency_hz is None:
                self._resonator_x *= 0.98
                self._resonator_v *= 0.95
            else:
                forcing = float(np.clip(drive[i], -1.0, 1.0)) * self._entrainment
                acceleration = (
                    target_omega * target_omega * (forcing - self._resonator_x)
                    - 2.0 * self.resonance_damping * target_omega * self._resonator_v
                )
                self._resonator_v += acceleration * dt
                self._resonator_x += self._resonator_v * dt
            harmonic = np.sin(2.0 * target_omega * times[i] + 0.4) if target_frequency_hz is not None else 0.0
            data[:, i] = (
                self.participant.colored_noise_uv * self._background
                + self.rng.normal(0.0, self.participant.white_noise_uv, size=8)
                + POSTERIOR_WEIGHTS * self.participant.alpha_amplitude_uv * alpha
                + POSTERIOR_WEIGHTS
                * self.participant.ssvep_amplitude_uv
                * (self._resonator_x + self.participant.first_harmonic_ratio * self._entrainment * harmonic)
            )
        artifact, dropout_mask, _artifact_events = self._artifact.render(times)
        data += artifact
        data[dropout_mask] = 0.0
        return WorldModelEmission(
            data_uv=data.astype(np.float32),
            latent={
                "attention_gain": float(target_gain[-1]) if target_gain.size else 0.0,
                "entrainment": float(self._entrainment),
                "resonator_x": float(self._resonator_x),
                "resonator_v": float(self._resonator_v),
                "stimulus_coupling": 1.0,
                "participant_stream_coupling": 1.0,
            },
        )

    def render_world(self, block: WorldInputBlock) -> WorldModelEmission:
        block.validate()
        raw_frequency = block.target.get("frequency_hz")
        target_frequency_hz = None if raw_frequency is None else float(raw_frequency)
        return self._render_with_attention_stream(
            block.sample_times_s,
            block.visual_luminance,
            target_frequency_hz,
            block.attention_gain,
        )

    def render(
        self,
        sample_times_s: np.ndarray,
        emitted_stimulus: np.ndarray,
        target_frequency_hz: float | None,
        attention_gain: float,
    ) -> WorldModelEmission:
        times = np.asarray(sample_times_s, dtype=float)
        gain = np.full(times.size, max(0.0, float(attention_gain)), dtype=float)
        return self._render_with_attention_stream(
            times,
            emitted_stimulus,
            target_frequency_hz,
            gain,
        )
