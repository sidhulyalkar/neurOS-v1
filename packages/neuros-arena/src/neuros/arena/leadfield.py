"""Portable lead-field world model and optional MNE export adapter.

The expensive anatomy/research step is separated from Arena execution:

1. MNE reads a forward solution and exports a small ``.npz`` bundle containing
   sensor-space topographies for explicitly selected cortical sources.
2. Arena loads that bundle dependency-light and drives the topographies from the
   actual emitted stimulus waveform.

This preserves the causal display -> neural response -> scalp projection chain
without requiring every creative developer or CI worker to install MNE or carry
a full FreeSurfer subject.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np

from neuros.drivers.synthetic_eeg import ArtifactEvent as DriverArtifactEvent

from .specs import ParticipantProfile
from .world_models import WorldModelEmission, _ArtifactEngine


BUNDLE_SCHEMA = "neuros.arena.leadfield_bundle.v1"


def save_leadfield_bundle(
    path: str | Path,
    *,
    channel_names: Sequence[str],
    visual_topography: np.ndarray,
    nuisance_topographies: np.ndarray | None = None,
    metadata: dict[str, str] | None = None,
) -> None:
    """Write a portable sensor-topography bundle used by the world model."""
    channels = tuple(str(name) for name in channel_names)
    visual = np.asarray(visual_topography, dtype=float).reshape(-1)
    if not channels or visual.size != len(channels):
        raise ValueError("visual_topography must have one value per channel")
    scale = float(np.max(np.abs(visual)))
    if scale <= 0 or not np.all(np.isfinite(visual)):
        raise ValueError("visual_topography must be finite and non-zero")
    visual = visual / scale
    if nuisance_topographies is None:
        nuisance = np.empty((0, len(channels)), dtype=float)
    else:
        nuisance = np.asarray(nuisance_topographies, dtype=float)
        if nuisance.ndim != 2 or nuisance.shape[1] != len(channels) or not np.all(np.isfinite(nuisance)):
            raise ValueError("nuisance_topographies must be finite sources x channels")
        norms = np.max(np.abs(nuisance), axis=1, keepdims=True)
        nuisance = nuisance / np.maximum(norms, 1e-12)
    payload_meta = metadata or {}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray(BUNDLE_SCHEMA),
        channel_names=np.asarray(channels),
        visual_topography=visual.astype(np.float32),
        nuisance_topographies=nuisance.astype(np.float32),
        metadata_keys=np.asarray(list(payload_meta), dtype=str),
        metadata_values=np.asarray([payload_meta[key] for key in payload_meta], dtype=str),
    )


def export_mne_forward_bundle(
    forward_path: str | Path,
    output_path: str | Path,
    *,
    visual_source_indices: Sequence[int],
    visual_source_weights: Sequence[float] | None = None,
    eeg_channel_names: Sequence[str] | None = None,
    nuisance_source_indices: Sequence[int] = (),
) -> None:
    """Export an MNE forward solution into Arena's portable topography bundle.

    Source selection is deliberately explicit. Arena will not guess which
    cortical vertices represent the developer's visual or other task generator.
    A research pipeline can obtain indices from anatomical labels and then freeze
    the resulting bundle for portable tests.
    """
    try:
        import mne
    except ImportError as exc:  # pragma: no cover - optional research dependency
        raise ImportError("MNE export requires `neuros-arena[real]` or mne>=1.6") from exc

    forward = mne.read_forward_solution(str(forward_path), verbose=False)
    forward = mne.convert_forward_solution(
        forward,
        surf_ori=True,
        force_fixed=True,
        use_cps=True,
        verbose=False,
    )
    gain = np.asarray(forward["sol"]["data"], dtype=float)
    row_names = tuple(str(name) for name in forward["sol"]["row_names"])
    if eeg_channel_names is None:
        picks = mne.pick_types(forward["info"], meg=False, eeg=True, exclude=[])
        selected_names = tuple(forward["info"]["ch_names"][int(index)] for index in picks)
    else:
        selected_names = tuple(str(name) for name in eeg_channel_names)
    row_indices = []
    for name in selected_names:
        if name not in row_names:
            raise ValueError(f"EEG channel {name!r} is unavailable from the forward solution")
        row_indices.append(row_names.index(name))
    selected_gain = gain[np.asarray(row_indices, dtype=int)]

    visual_idx = np.asarray(tuple(int(value) for value in visual_source_indices), dtype=int)
    if visual_idx.size == 0 or np.any(visual_idx < 0) or np.any(visual_idx >= selected_gain.shape[1]):
        raise ValueError("visual_source_indices must contain valid fixed-orientation source columns")
    if visual_source_weights is None:
        weights = np.ones(visual_idx.size, dtype=float) / visual_idx.size
    else:
        weights = np.asarray(tuple(float(value) for value in visual_source_weights), dtype=float)
        if weights.shape != visual_idx.shape or not np.all(np.isfinite(weights)) or np.allclose(weights, 0.0):
            raise ValueError("visual_source_weights must be finite and match visual_source_indices")
        weights = weights / np.sum(np.abs(weights))
    visual_topography = selected_gain[:, visual_idx] @ weights

    nuisance = []
    for index in nuisance_source_indices:
        idx = int(index)
        if not 0 <= idx < selected_gain.shape[1]:
            raise ValueError(f"nuisance source index out of bounds: {idx}")
        nuisance.append(selected_gain[:, idx])
    save_leadfield_bundle(
        output_path,
        channel_names=selected_names,
        visual_topography=visual_topography,
        nuisance_topographies=(np.asarray(nuisance) if nuisance else None),
        metadata={
            "source": str(forward_path),
            "projection": "mne-fixed-orientation-forward",
            "visual_sources": ",".join(str(value) for value in visual_idx.tolist()),
        },
    )


class LeadFieldDrivenWorldModel:
    """Display-driven neural dynamics projected through a frozen lead field."""

    name = "leadfield_driven"

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
            raise ValueError("leadfield_driven requires world_model.parameters.path")
        path = Path(str(params["path"])).expanduser()
        if not path.exists():
            raise FileNotFoundError(path)
        with np.load(path, allow_pickle=False) as payload:
            schema = str(np.asarray(payload["schema"]).reshape(-1)[0])
            if schema != BUNDLE_SCHEMA:
                raise ValueError(f"expected lead-field bundle schema {BUNDLE_SCHEMA!r}")
            self.channel_names = tuple(str(value) for value in np.asarray(payload["channel_names"]).tolist())
            self.visual_topography = np.asarray(payload["visual_topography"], dtype=float).reshape(-1)
            self.nuisance_topographies = np.asarray(payload["nuisance_topographies"], dtype=float)
        if self.visual_topography.size != len(self.channel_names):
            raise ValueError("lead-field visual topography/channel mismatch")
        self.participant = participant
        self.fs = float(sampling_rate_hz)
        self.rng = np.random.default_rng(seed)
        self._entrainment = 0.0
        self._response_state = 0.0
        self._alpha_phase = self.rng.uniform(0.0, 2 * np.pi)
        self._nuisance_state = np.zeros(self.nuisance_topographies.shape[0], dtype=float)
        self._tau_s = float(params.get("entrainment_tau_s", 0.18))
        self._response_tau_s = float(params.get("response_tau_s", 0.035))
        if self._tau_s <= 0 or self._response_tau_s <= 0:
            raise ValueError("lead-field time constants must be positive")
        self._artifact = _ArtifactEngine(self.fs, seed + 1777) if len(self.channel_names) == 8 else None

    def _require_standard_artifact_montage(self) -> _ArtifactEngine:
        if self._artifact is None:
            raise ValueError(
                "generic artifact overlay currently requires the standard 8-channel Arena montage"
            )
        return self._artifact

    def inject_artifact(self, kind: str, duration_seconds: float, severity: float) -> None:
        self._require_standard_artifact_montage().inject(kind, duration_seconds, severity)

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
        return self._require_standard_artifact_montage().schedule(
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
        times = np.asarray(sample_times_s, dtype=float)
        drive = np.asarray(emitted_stimulus, dtype=float)
        if times.ndim != 1 or drive.shape != times.shape:
            raise ValueError("sample_times_s and emitted_stimulus must match")
        n_channels = len(self.channel_names)
        data = self.rng.normal(0.0, self.participant.white_noise_uv, size=(n_channels, times.size))
        dt = 1.0 / self.fs
        target_gain = max(0.0, float(attention_gain)) if target_frequency_hz is not None else 0.0
        visual_response = np.zeros(times.size, dtype=float)
        for i in range(times.size):
            self._entrainment += (target_gain - self._entrainment) * min(1.0, dt / self._tau_s)
            desired = float(np.clip(drive[i], -1.0, 1.0)) * self._entrainment
            self._response_state += (desired - self._response_state) * min(1.0, dt / self._response_tau_s)
            visual_response[i] = self._response_state
        data += (
            self.visual_topography[:, None]
            * self.participant.ssvep_amplitude_uv
            * visual_response[None, :]
        )
        # Use frozen lead-field nuisance topographies for low-frequency background
        # sources, then add a global endogenous alpha component with channel gain
        # derived from the absolute visual topography.
        if self.nuisance_topographies.size:
            persistence = 0.995
            innovation = np.sqrt(1.0 - persistence**2)
            for i in range(times.size):
                self._nuisance_state = persistence * self._nuisance_state + innovation * self.rng.normal(size=self._nuisance_state.size)
                data[:, i] += self.participant.colored_noise_uv * (self._nuisance_state @ self.nuisance_topographies)
        else:
            data += self.rng.normal(0.0, self.participant.colored_noise_uv, size=data.shape)
        alpha_gain = np.abs(self.visual_topography)
        alpha_gain /= max(float(np.max(alpha_gain)), 1e-12)
        data += (
            alpha_gain[:, None]
            * self.participant.alpha_amplitude_uv
            * np.sin(2 * np.pi * self.participant.alpha_frequency_hz * times + self._alpha_phase)[None, :]
        )
        if self._artifact is not None:
            artifact, dropout_mask, _artifact_events = self._artifact.render(times)
            data += artifact
            data[dropout_mask] = 0.0
        return WorldModelEmission(
            data_uv=data.astype(np.float32),
            latent={
                "attention_gain": target_gain,
                "entrainment": float(self._entrainment),
                "stimulus_coupling": 1.0,
                "leadfield_projection": 1.0,
            },
        )
