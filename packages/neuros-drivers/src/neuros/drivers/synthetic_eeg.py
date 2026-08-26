"""Protocol-grade synthetic EEG generator for acquisition and BCI stress tests.

The generator is deliberately not a physiological digital twin. It creates a
controlled signal with realistic nuisance structure so downstream systems can
be tested against weak SSVEPs, endogenous alpha, contact loss, movement and EMG
without requiring physical hardware for every iteration.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ArtifactKind = Literal["blink", "jaw", "controller", "motion", "saturation", "dropout"]


@dataclass(frozen=True)
class SyntheticEEGConfig:
    sampling_rate_hz: float = 250.0
    channel_names: tuple[str, ...] = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
    colored_noise_uv: float = 4.5
    white_noise_uv: float = 1.25
    alpha_frequency_hz: float = 9.4
    alpha_amplitude_uv: float = 2.8
    ssvep_amplitude_uv: float = 8.0
    first_harmonic_ratio: float = 0.34
    seed: int = 7

    def validate(self) -> None:
        if self.sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if len(self.channel_names) != 8:
            raise ValueError("the default synthetic EEG profile expects exactly 8 channels")
        if self.colored_noise_uv < 0 or self.white_noise_uv < 0:
            raise ValueError("noise amplitudes must be non-negative")
        if self.alpha_frequency_hz <= 0 or self.alpha_amplitude_uv < 0:
            raise ValueError("alpha parameters must be positive/non-negative")
        if self.ssvep_amplitude_uv < 0:
            raise ValueError("ssvep_amplitude_uv must be non-negative")
        if not 0 <= self.first_harmonic_ratio <= 2:
            raise ValueError("first_harmonic_ratio must be in [0, 2]")


@dataclass(frozen=True)
class SyntheticEEGBlock:
    data_uv: np.ndarray
    timestamps_s: np.ndarray
    target_frequency_hz: float | None
    attention_gain: float
    artifact: str | None


class SyntheticEEGGenerator:
    """Stateful eight-channel EEG source with controllable SSVEP and artifacts."""

    posterior_weights = np.asarray([0.05, 0.05, 0.08, 0.05, 0.45, 0.85, 1.0, 0.85])
    central_weights = np.asarray([0.18, 0.95, 1.0, 0.95, 0.25, 0.10, 0.08, 0.10])
    frontal_weights = np.asarray([1.0, 0.30, 0.55, 0.30, 0.22, 0.12, 0.10, 0.12])

    def __init__(self, config: SyntheticEEGConfig | None = None) -> None:
        self.config = config or SyntheticEEGConfig()
        self.config.validate()
        self.rng = np.random.default_rng(self.config.seed)
        self.sample_index = 0
        self.target_frequency_hz: float | None = None
        self.attention_gain = 0.0
        self.channel_gain = np.ones(8, dtype=float)
        self._colored_state = np.zeros((8, 4), dtype=float)
        self._phase = self.rng.uniform(0, 2 * np.pi, 8)
        self._alpha_phase = self.rng.uniform(0, 2 * np.pi, 8)
        self._artifact_kind: str | None = None
        self._artifact_remaining = 0
        self._artifact_total = 0
        self._artifact_severity = 1.0

    def set_attention(self, frequency_hz: float | None, gain: float = 1.0) -> None:
        if frequency_hz is not None and frequency_hz <= 0:
            raise ValueError("frequency_hz must be positive")
        self.target_frequency_hz = None if frequency_hz is None else float(frequency_hz)
        self.attention_gain = float(np.clip(gain, 0.0, 1.5)) if frequency_hz is not None else 0.0

    def set_channel_gain(self, channel: str | int, gain: float) -> None:
        index = self.config.channel_names.index(channel) if isinstance(channel, str) else int(channel)
        if not 0 <= index < 8:
            raise IndexError("channel index out of range")
        self.channel_gain[index] = max(0.0, float(gain))

    def inject_artifact(self, kind: ArtifactKind, duration_seconds: float = 0.35, severity: float = 1.0) -> None:
        if kind not in {"blink", "jaw", "controller", "motion", "saturation", "dropout"}:
            raise ValueError(f"unsupported artifact kind: {kind}")
        if duration_seconds <= 0:
            raise ValueError("duration_seconds must be positive")
        self._artifact_kind = kind
        self._artifact_total = max(1, int(round(duration_seconds * self.config.sampling_rate_hz)))
        self._artifact_remaining = self._artifact_total
        self._artifact_severity = max(0.0, float(severity))

    def _colored_noise(self, samples: int) -> np.ndarray:
        alphas = np.asarray([0.70, 0.90, 0.975, 0.995])
        weights = np.asarray([0.55, 0.42, 0.30, 0.20])
        out = np.empty((8, samples), dtype=float)
        for index in range(samples):
            innovation = self.rng.normal(size=(8, 4))
            self._colored_state = alphas * self._colored_state + np.sqrt(1.0 - alphas**2) * innovation
            out[:, index] = (self._colored_state * weights).sum(axis=1)
        out /= max(float(np.std(out)), 1e-8)
        return out * self.config.colored_noise_uv

    def _render_artifact(self, samples: int, time_s: np.ndarray) -> np.ndarray:
        if self._artifact_kind is None or self._artifact_remaining <= 0:
            return np.zeros((8, samples), dtype=float)
        count = min(samples, self._artifact_remaining)
        start = self._artifact_total - self._artifact_remaining
        phase = (np.arange(count) + start) / max(1, self._artifact_total - 1)
        output = np.zeros((8, samples), dtype=float)
        severity = self._artifact_severity
        kind = self._artifact_kind
        if kind == "blink":
            pulse = np.sin(np.pi * np.clip(phase, 0, 1)) ** 2
            output[:, :count] += self.frontal_weights[:, None] * (120 * severity * pulse)
        elif kind == "jaw":
            t = time_s[:count]
            high_frequency = np.sin(2 * np.pi * 38 * t) + 0.8 * np.sin(2 * np.pi * 53 * t + 0.7) + 0.55 * np.sin(2 * np.pi * 71 * t + 1.1)
            output[:, :count] += (0.55 * self.central_weights + 0.35)[:, None] * (48 * severity * high_frequency)
        elif kind == "controller":
            t = time_s[:count]
            controller_emg = np.sin(2 * np.pi * 31 * t) + 0.55 * np.sin(2 * np.pi * 46 * t + 0.4) + 0.30 * self.rng.normal(size=count)
            output[:, :count] += self.central_weights[:, None] * (24 * severity * controller_emg)
        elif kind == "motion":
            drift = np.sin(np.pi * np.clip(phase, 0, 1)) * np.sign(np.sin(2 * np.pi * 2.2 * time_s[:count]))
            output[:, :count] += (0.60 + 0.40 * self.frontal_weights)[:, None] * (55 * severity * drift)
        elif kind == "saturation":
            output[6, :count] += 480 * severity
        self._artifact_remaining -= count
        if self._artifact_remaining <= 0:
            self._artifact_kind = None
        return output

    def render(self, samples: int) -> SyntheticEEGBlock:
        if samples <= 0:
            raise ValueError("samples must be positive")
        fs = self.config.sampling_rate_hz
        sample_index = self.sample_index + np.arange(samples)
        time_s = sample_index / fs
        data = self._colored_noise(samples)
        data += self.rng.normal(0, self.config.white_noise_uv, size=(8, samples))
        alpha = np.sin(2 * np.pi * self.config.alpha_frequency_hz * time_s[None, :] + self._alpha_phase[:, None])
        data += self.posterior_weights[:, None] * self.config.alpha_amplitude_uv * alpha
        if self.target_frequency_hz is not None and self.attention_gain > 0:
            frequency = self.target_frequency_hz
            fundamental = np.sin(2 * np.pi * frequency * time_s[None, :] + self._phase[:, None])
            harmonic = np.sin(2 * np.pi * 2 * frequency * time_s[None, :] + 0.5 * self._phase[:, None])
            ssvep = fundamental + self.config.first_harmonic_ratio * harmonic
            data += self.posterior_weights[:, None] * self.config.ssvep_amplitude_uv * self.attention_gain * ssvep
        active_artifact = self._artifact_kind
        data += self._render_artifact(samples, time_s)
        data *= self.channel_gain[:, None]
        if active_artifact == "dropout":
            data[6, :] = 0.0
        self.sample_index += samples
        return SyntheticEEGBlock(data.astype(np.float32), time_s.astype(float), self.target_frequency_hz, self.attention_gain, active_artifact)
