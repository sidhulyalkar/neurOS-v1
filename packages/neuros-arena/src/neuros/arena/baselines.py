"""Portable recorded-background helpers for semi-synthetic Arena worlds."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .recording import RecordingMetadata, recording_sidecar_path, save_recording_metadata

BASELINE_SCHEMA = "neuros.arena.eeg_baseline.v1"


def save_eeg_baseline(
    path: str | Path,
    *,
    data_uv: np.ndarray,
    sampling_rate_hz: float,
    channel_names: Sequence[str],
    metadata: dict[str, str] | None = None,
    recording_metadata: RecordingMetadata | None = None,
) -> None:
    """Save a finite channels×samples EEG background with explicit units.

    ``recording_metadata`` writes a traceable JSON sidecar next to the compact
    NPZ. The sidecar is BIDS-aligned provenance, not a replacement for a
    canonical BIDS dataset.
    """
    data = np.asarray(data_uv, dtype=float)
    channels = tuple(str(name) for name in channel_names)
    if data.ndim != 2 or data.shape[0] != len(channels) or data.shape[1] < 32:
        raise ValueError("data_uv must be channels x samples with one channel name per row")
    if not np.all(np.isfinite(data)):
        raise ValueError("baseline EEG must be finite")
    if sampling_rate_hz <= 0 or not channels:
        raise ValueError("sampling_rate_hz and channel_names are required")
    if len(set(channels)) != len(channels):
        raise ValueError("channel_names must be unique")
    info = metadata or {}
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        schema=np.asarray(BASELINE_SCHEMA),
        data_uv=data.astype(np.float32),
        sampling_rate_hz=np.asarray(float(sampling_rate_hz)),
        channel_names=np.asarray(channels),
        metadata_keys=np.asarray(list(info), dtype=str),
        metadata_values=np.asarray([str(info[key]) for key in info], dtype=str),
    )
    if recording_metadata is not None:
        save_recording_metadata(
            recording_metadata,
            recording_sidecar_path(output),
            channel_names=channels,
        )


def _mne_channel_types(raw: Any) -> dict[str, str]:
    """Extract MNE channel types without depending on private Info fields."""
    try:
        types = raw.get_channel_types()
    except AttributeError:  # pragma: no cover - old external MNE object
        return {}
    return {str(name): str(kind) for name, kind in zip(raw.ch_names, types, strict=True)}


def export_mne_raw_baseline(
    raw: Any,
    path: str | Path,
    *,
    channel_names: Sequence[str],
    tmin_s: float = 0.0,
    duration_s: float | None = None,
    target_sampling_rate_hz: float | None = None,
    metadata: dict[str, str] | None = None,
    recording_metadata: RecordingMetadata | None = None,
) -> None:
    """Export a selected MNE Raw window to Arena's semi-synthetic baseline.

    The function uses only MNE's public Raw methods/properties and does not
    download datasets. MOABB or any other dataset package can remain upstream.
    EEG values returned by MNE are volts and are converted explicitly to µV.

    If no ``recording_metadata`` object is supplied, Arena records only values
    that can be obtained safely from the MNE object itself. Dataset/subject/task
    provenance should normally be supplied by the caller because it belongs to
    the upstream BIDS/MOABB dataset, not to signal inference inside Arena.
    """
    try:
        import mne  # noqa: F401
    except ImportError as exc:  # pragma: no cover - optional research dependency
        raise ImportError("MNE baseline export requires `neuros-arena[real]` or mne>=1.6") from exc
    if tmin_s < 0 or (duration_s is not None and duration_s <= 0):
        raise ValueError("tmin_s must be >= 0 and duration_s must be positive")
    work = raw.copy().pick(list(channel_names))
    preprocessing: list[str] = []
    if target_sampling_rate_hz is not None:
        if target_sampling_rate_hz <= 0:
            raise ValueError("target_sampling_rate_hz must be positive")
        work.resample(float(target_sampling_rate_hz))
        preprocessing.append(f"resampled_to_{float(target_sampling_rate_hz):g}Hz")
    sfreq = float(work.info["sfreq"])
    start = int(round(tmin_s * sfreq))
    stop = None if duration_s is None else start + int(round(duration_s * sfreq))
    if start >= work.n_times:
        raise ValueError("tmin_s is beyond the recording")
    stop = work.n_times if stop is None else min(stop, work.n_times)
    data_v = np.asarray(work.get_data(start=start, stop=stop), dtype=float)
    if recording_metadata is None:
        line_frequency = work.info.get("line_freq")
        recording_metadata = RecordingMetadata(
            source_format="mne.Raw",
            line_frequency_hz=(None if line_frequency is None else float(line_frequency)),
            channel_units={name: "uV" for name in work.ch_names},
            channel_types=_mne_channel_types(work),
            preprocessing=tuple(preprocessing),
            notes=(
                "Arena baseline values were converted from MNE SI volts to microvolts.",
                "Dataset/task/subject provenance was not inferred; supply it explicitly when available.",
            ),
        )
    elif preprocessing:
        recording_metadata = replace(
            recording_metadata,
            preprocessing=tuple(recording_metadata.preprocessing) + tuple(preprocessing),
        )
    save_eeg_baseline(
        path,
        data_uv=data_v * 1e6,
        sampling_rate_hz=sfreq,
        channel_names=tuple(work.ch_names),
        metadata={"source": "mne.Raw", **(metadata or {})},
        recording_metadata=recording_metadata,
    )
