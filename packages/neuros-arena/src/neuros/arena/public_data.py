"""Thin public-dataset adapters for Arena studies.

Core Arena remains dataset-library agnostic. This module intentionally relies on
MOABB's documented ``subject -> session -> run -> MNE Raw`` contract rather than
hard-coding individual dataset layouts or silently preprocessing task data.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .baselines import export_mne_raw_baseline
from .recording import RecordingMetadata


@dataclass(frozen=True)
class MOABBDomainRun:
    dataset: str
    subject: str
    session: str
    run: str
    raw: Any

    @property
    def domain_id(self) -> str:
        return f"{self.dataset}:sub-{self.subject}:ses-{self.session}:run-{self.run}"


def _dataset_name(dataset: Any) -> str:
    for attr in ("code", "name"):
        value = getattr(dataset, attr, None)
        if value:
            return str(value)
    return dataset.__class__.__name__


def iter_moabb_raw_runs(dataset: Any, *, subjects: Sequence[int] | None = None) -> Iterable[MOABBDomainRun]:
    """Yield MOABB raw runs without applying hidden Arena preprocessing."""
    getter = getattr(dataset, "get_data", None)
    if not callable(getter):
        raise TypeError("dataset must expose MOABB-compatible get_data(subjects=...)")
    data = getter(subjects=None if subjects is None else list(subjects))
    if not isinstance(data, dict) or not data:
        raise ValueError("MOABB get_data returned no subject data")
    dataset_name = _dataset_name(dataset)
    for subject, sessions in data.items():
        if not isinstance(sessions, dict):
            raise TypeError("MOABB subject data must map sessions to runs")
        for session, runs in sessions.items():
            if not isinstance(runs, dict):
                raise TypeError("MOABB session data must map run ids to Raw objects")
            for run, raw in runs.items():
                if not hasattr(raw, "get_data") or not hasattr(raw, "info") or not hasattr(raw, "ch_names"):
                    raise TypeError("MOABB run does not expose the expected MNE Raw surface")
                yield MOABBDomainRun(
                    dataset=dataset_name,
                    subject=str(subject),
                    session=str(session),
                    run=str(run),
                    raw=raw,
                )


def export_moabb_run_window(
    domain: MOABBDomainRun,
    path: str | Path,
    *,
    channel_names: Sequence[str],
    tmin_s: float,
    duration_s: float,
    target_sampling_rate_hz: float | None = None,
    task: str = "",
    acquisition: str = "",
    source_license: str = "",
    reference: str = "",
    preprocessing_notes: Sequence[str] = (),
) -> Path:
    """Export one explicitly selected MOABB run window with traceable provenance.

    Arena does not infer whether the window is resting state, task baseline or
    evoked activity. The study author must make that semantic choice and record
    it through ``task``/notes and the surrounding protocol.
    """
    if tmin_s < 0 or duration_s <= 0:
        raise ValueError("tmin_s must be >= 0 and duration_s must be positive")
    output = Path(path)
    metadata = RecordingMetadata(
        dataset=domain.dataset,
        subject=domain.subject,
        session=domain.session,
        run=domain.run,
        task=task,
        acquisition=acquisition,
        source_locator=domain.domain_id,
        source_format="MOABB -> MNE Raw",
        source_license=source_license,
        reference=reference,
        preprocessing=tuple(str(item) for item in preprocessing_notes),
        notes=(
            f"Explicit run window: tmin={float(tmin_s):g}s duration={float(duration_s):g}s.",
            "Window semantics are declared by the study author; Arena did not infer baseline/task meaning.",
        ),
    )
    export_mne_raw_baseline(
        domain.raw,
        output,
        channel_names=channel_names,
        tmin_s=tmin_s,
        duration_s=duration_s,
        target_sampling_rate_hz=target_sampling_rate_hz,
        recording_metadata=metadata,
    )
    return output
