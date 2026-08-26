"""Portable application/game traces for closed-loop Arena conformance.

The application trace is intentionally engine-agnostic. Unity, Godot, Web,
OpenViBE or custom applications can export the same small event schema and let
Arena judge behavior against known world, timing and transport ground truth.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .runner import ArenaRun

APPLICATION_TRACE_SCHEMA = "neuros.synthetic_bci_arena.application_trace.v1"

STANDARD_EVENT_KINDS = {
    "neural_action",
    "neural_accept",
    "neural_abstain",
    "calibration_ready",
    "calibration_failed",
    "bci_lost",
    "bci_recovered",
    "participant_stop",
    "application_state",
}


@dataclass(frozen=True)
class ApplicationEvent:
    timestamp_s: float
    kind: str
    action: str = ""
    source: str = "application"
    authority: float | None = None
    source_sequence: int | None = None
    payload: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    def validate(self) -> None:
        if not np.isfinite(self.timestamp_s) or self.timestamp_s < 0:
            raise ValueError("application event timestamp_s must be finite and non-negative")
        if not self.kind:
            raise ValueError("application event kind is required")
        if self.authority is not None and (not np.isfinite(self.authority) or self.authority < 0):
            raise ValueError("application authority must be finite and non-negative")
        if self.source_sequence is not None and self.source_sequence < 0:
            raise ValueError("source_sequence must be non-negative when present")
        for key, value in self.payload.items():
            if not isinstance(key, str) or not key:
                raise ValueError("application payload keys must be non-empty strings")
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                raise ValueError(f"application payload {key!r} must be a JSON scalar")


@dataclass(frozen=True)
class ApplicationTrace:
    application: str
    version: str
    events: tuple[ApplicationEvent, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.application or not self.version:
            raise ValueError("application trace requires application and version")
        previous = -np.inf
        for event in self.events:
            event.validate()
            if event.timestamp_s < previous:
                raise ValueError("application events must be monotonic by timestamp_s")
            previous = event.timestamp_s
        for key, value in self.metadata.items():
            if not str(key) or not str(value):
                raise ValueError("application trace metadata keys/values must be non-empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": APPLICATION_TRACE_SCHEMA,
            "application": self.application,
            "version": self.version,
            "metadata": dict(self.metadata),
            "events": [asdict(event) for event in self.events],
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ApplicationTrace":
        if raw.get("schema") != APPLICATION_TRACE_SCHEMA:
            raise ValueError(f"expected application trace schema {APPLICATION_TRACE_SCHEMA!r}")
        trace = cls(
            application=str(raw["application"]),
            version=str(raw["version"]),
            metadata={str(key): str(value) for key, value in dict(raw.get("metadata", {})).items()},
            events=tuple(ApplicationEvent(**dict(item)) for item in raw.get("events", [])),
        )
        trace.validate()
        return trace


def load_application_trace(path: str | Path) -> ApplicationTrace:
    return ApplicationTrace.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def save_application_trace(trace: ApplicationTrace, path: str | Path) -> Path:
    trace.validate()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(trace.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output


def _truth_target_at(run: ArenaRun, timestamp_s: float) -> float | None:
    for interval in run.stages:
        if interval.start_s <= timestamp_s < interval.end_s:
            return interval.target_frequency_hz
    return None


def _in_transport_silence(run: ArenaRun, timestamp_s: float, grace_s: float) -> bool:
    for start, duration in run.transport.silence_windows:
        if start - grace_s <= timestamp_s < start + duration + grace_s:
            return True
    return False


def evaluate_application_trace(
    run: ArenaRun,
    trace: ApplicationTrace,
    *,
    neural_action_kind: str = "neural_action",
    silence_grace_s: float = 0.0,
) -> dict[str, float]:
    """Score generic application safety/authority behavior against Arena truth.

    The function deliberately reports observations rather than imposing a single
    universal pass criterion. A benchmark pack or application-specific evaluator
    decides which values are acceptable for a particular game/paradigm.
    """
    trace.validate()
    if silence_grace_s < 0:
        raise ValueError("silence_grace_s must be non-negative")
    actions = [event for event in trace.events if event.kind == neural_action_kind]
    accepts = [event for event in trace.events if event.kind == "neural_accept"]
    abstains = [event for event in trace.events if event.kind == "neural_abstain"]
    lost = [event for event in trace.events if event.kind == "bci_lost"]
    recovered = [event for event in trace.events if event.kind == "bci_recovered"]
    stops = [event for event in trace.events if event.kind == "participant_stop"]

    actions_without_target = sum(_truth_target_at(run, event.timestamp_s) is None for event in actions)
    actions_during_silence = sum(_in_transport_silence(run, event.timestamp_s, silence_grace_s) for event in actions)
    authorities = np.asarray(
        [event.authority for event in trace.events if event.authority is not None],
        dtype=float,
    )
    sequences = [event.source_sequence for event in actions if event.source_sequence is not None]
    sequence_regressions = sum(
        int(sequences[index + 1] <= sequences[index])
        for index in range(len(sequences) - 1)
    )

    recovery_latencies: list[float] = []
    for silence_start, duration in run.transport.silence_windows:
        silence_end = silence_start + duration
        later = [event.timestamp_s for event in recovered if event.timestamp_s >= silence_end]
        if later:
            recovery_latencies.append(min(later) - silence_end)

    participant_stop_violations = 0
    if stops:
        first_stop = stops[0].timestamp_s
        participant_stop_violations = sum(event.timestamp_s > first_stop for event in actions)

    return {
        "events_total": float(len(trace.events)),
        "neural_actions_total": float(len(actions)),
        "neural_accepts_total": float(len(accepts)),
        "neural_abstains_total": float(len(abstains)),
        "neural_actions_without_target": float(actions_without_target),
        "neural_actions_during_transport_silence": float(actions_during_silence),
        "neural_action_sequence_regressions": float(sequence_regressions),
        "participant_stop_action_violations": float(participant_stop_violations),
        "bci_lost_events": float(len(lost)),
        "bci_recovered_events": float(len(recovered)),
        "recovery_latency_mean_s": float(np.mean(recovery_latencies)) if recovery_latencies else 0.0,
        "recovery_latency_max_s": float(np.max(recovery_latencies)) if recovery_latencies else 0.0,
        "authority_mean": float(np.mean(authorities)) if authorities.size else 0.0,
        "authority_max": float(np.max(authorities)) if authorities.size else 0.0,
    }
