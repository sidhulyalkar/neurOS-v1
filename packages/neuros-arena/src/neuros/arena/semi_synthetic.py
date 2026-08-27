"""Semi-synthetic EEG world model using real or recorded background data.

The model replays an observed EEG background and injects a known response driven
by Arena's emitted display waveform. Recorded background does not imply that the
synthetic injected response occurred in the recorded participant.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neuros.drivers.synthetic_eeg import ArtifactEvent as DriverArtifactEvent

from .specs import ParticipantProfile
from .world_input import WorldInputBlock
from .world_models import POSTERIOR_WEIGHTS, SOURCE_CHANNELS, WorldModelEmission, _ArtifactEngine


class SemiSyntheticReplayWorldModel:
    name = "semi_synthetic_replay"
    channel_names = SOURCE_CHANNELS

    def __init__(
        self,
        *,
        participant: ParticipantProfile,
        sampling_rate_hz: float,
        seed: int,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        params = dict(parameters or {})
        if not params.get("path"):
            raise ValueError("semi_synthetic_replay requires world_model.parameters.path")
        path = Path(str(params["path"])).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            if "data_uv" not in payload or "sampling_rate_hz" not in payload or "channel_names" not in payload:
                raise ValueError("baseline NPZ requires data_uv, sampling_rate_hz and channel_names")
            data = np.asarray(payload["data_uv"], dtype=float)
            source_fs = float(np.asarray(payload["sampling_rate_hz"]).reshape(-1)[0])
            names = tuple(str(value) for value in np.asarray(payload["channel_names"]).tolist())
        if data.ndim != 2 or data.shape[1] < 32:
            raise ValueError("baseline data_uv must be channels x samples with at least 32 samples")
        if source_fs <= 0:
            raise ValueError("baseline sampling_rate_hz must be positive")
        missing = [name for name in SOURCE_CHANNELS if name not in names]
        if missing:
            raise ValueError(f"baseline is missing required Arena channels: {missing}")
        selected = np.asarray([data[names.index(name)] for name in SOURCE_CHANNELS], dtype=float)
        self.fs = float(sampling_rate_hz)
        self.participant = participant
        self.rng = np.random.default_rng(seed)
        self._baseline = self._resample(selected, source_fs, self.fs)
        if bool(params.get("demean", True)):
            self._baseline = self._baseline - np.mean(self._baseline, axis=1, keepdims=True)
        self._cursor = int(self.rng.integers(0, self._baseline.shape[1])) if bool(params.get("random_offset", True)) else 0
        self._entrainment = 0.0
        self._response_state = 0.0
        self._tau_s = float(params.get("entrainment_tau_s", 0.18))
        self._response_scale = float(params.get("response_scale", 1.0))
        if self._tau_s <= 0 or self._response_scale < 0:
            raise ValueError("semi-synthetic entrainment_tau_s/response_scale are invalid")
        self._artifact = _ArtifactEngine(self.fs, seed + 1171)

    @staticmethod
    def _resample(data: np.ndarray, source_fs: float, target_fs: float) -> np.ndarray:
        if abs(source_fs - target_fs) < 1e-9:
            return data.astype(np.float32)
        duration = (data.shape[1] - 1) / source_fs
        count = max(32, int(round(duration * target_fs)) + 1)
        old_t = np.arange(data.shape[1], dtype=float) / source_fs
        new_t = np.arange(count, dtype=float) / target_fs
        return np.vstack([np.interp(new_t, old_t, channel) for channel in data]).astype(np.float32)

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

    def _next_background(self, samples: int) -> np.ndarray:
        indices = (self._cursor + np.arange(samples)) % self._baseline.shape[1]
        block = self._baseline[:, indices].copy()
        self._cursor = int((self._cursor + samples) % self._baseline.shape[1])
        return block

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
        if drive.shape != times.shape or gain.shape != times.shape:
            raise ValueError("sample times, emitted stimulus and attention gain must match")
        if not np.all(np.isfinite(gain)) or np.any(gain < 0):
            raise ValueError("attention gain stream must be finite and non-negative")
        samples = times.size
        data = self._next_background(samples).astype(float)
        dt = 1.0 / self.fs
        target_gain = gain if target_frequency_hz is not None else np.zeros(samples, dtype=float)
        response = np.zeros(samples, dtype=float)
        entrainment_trace = np.zeros(samples, dtype=float)
        for i in range(samples):
            self._entrainment += (float(target_gain[i]) - self._entrainment) * min(1.0, dt / self._tau_s)
            entrainment_trace[i] = self._entrainment
            desired = float(np.clip(drive[i], -1.0, 1.0)) * self._entrainment
            self._response_state += (desired - self._response_state) * min(1.0, dt / 0.035)
            response[i] = self._response_state
        if target_frequency_hz is not None:
            harmonic = np.sin(4.0 * np.pi * float(target_frequency_hz) * times + 0.4)
            response = response + self.participant.first_harmonic_ratio * entrainment_trace * harmonic
        data += (
            POSTERIOR_WEIGHTS[:, None]
            * self.participant.ssvep_amplitude_uv
            * self._response_scale
            * response[None, :]
        )
        artifact, dropout_mask, _artifact_events = self._artifact.render(times)
        data += artifact
        data[dropout_mask] = 0.0
        return WorldModelEmission(
            data_uv=data.astype(np.float32),
            latent={
                "attention_gain": float(target_gain[-1]) if target_gain.size else 0.0,
                "entrainment": float(self._entrainment),
                "stimulus_coupling": 1.0,
                "participant_stream_coupling": 1.0,
                "baseline_replay": 1.0,
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
        return self._render_with_attention_stream(
            times,
            emitted_stimulus,
            target_frequency_hz,
            np.full(times.size, max(0.0, float(attention_gain)), dtype=float),
        )
