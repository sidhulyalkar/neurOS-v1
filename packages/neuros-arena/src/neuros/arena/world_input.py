"""Paradigm-neutral causal input blocks for neural world models.

Arena manifests remain SSVEP-friendly, but the model boundary should not be. A
world model may consume this richer block through ``render_world`` while legacy
models continue to implement the original ``render`` method.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


_JSON_SCALAR = str | int | float | bool | None


@dataclass(frozen=True)
class WorldInputBlock:
    """One causal chunk presented to a paradigm-neutral neural world model.

    ``emitted_streams`` contains physical/simulated sensory streams after
    presentation effects. ``participant_streams`` contains resolved synthetic
    participant state on the same sample clock. Scalar ``participant_state`` is
    retained as a compatibility/summary surface for older plugins.
    """

    sample_times_s: np.ndarray
    paradigm: str
    stage_label: str
    emitted_streams: dict[str, np.ndarray]
    target: dict[str, _JSON_SCALAR] = field(default_factory=dict)
    task_state: dict[str, _JSON_SCALAR] = field(default_factory=dict)
    participant_state: dict[str, _JSON_SCALAR] = field(default_factory=dict)
    participant_streams: dict[str, np.ndarray] = field(default_factory=dict)

    def validate(self) -> None:
        times = np.asarray(self.sample_times_s, dtype=float)
        if times.ndim != 1 or not np.all(np.isfinite(times)):
            raise ValueError("sample_times_s must be a finite 1-D array")
        if not self.paradigm or not self.stage_label:
            raise ValueError("paradigm and stage_label are required")
        for mapping_name, streams in (
            ("emitted_streams", self.emitted_streams),
            ("participant_streams", self.participant_streams),
        ):
            for name, values in streams.items():
                if not name:
                    raise ValueError(f"{mapping_name} names must be non-empty")
                array = np.asarray(values, dtype=float)
                if array.shape != times.shape or not np.all(np.isfinite(array)):
                    raise ValueError(
                        f"{mapping_name}.{name} must be finite and match sample_times_s"
                    )
        for mapping_name, mapping in (
            ("target", self.target),
            ("task_state", self.task_state),
            ("participant_state", self.participant_state),
        ):
            for key, value in mapping.items():
                if not isinstance(key, str) or not key:
                    raise ValueError(f"{mapping_name} keys must be non-empty strings")
                if not isinstance(value, (str, int, float, bool)) and value is not None:
                    raise ValueError(f"{mapping_name}.{key} must be a JSON scalar")
                if isinstance(value, (float, np.floating)) and not np.isfinite(value):
                    raise ValueError(f"{mapping_name}.{key} must be finite")

    @property
    def visual_luminance(self) -> np.ndarray:
        values = self.emitted_streams.get("visual_luminance")
        if values is None:
            return np.zeros(np.asarray(self.sample_times_s).size, dtype=float)
        return np.asarray(values, dtype=float)

    @property
    def attention_gain(self) -> np.ndarray:
        values = self.participant_streams.get("attention_gain")
        if values is not None:
            return np.asarray(values, dtype=float)
        scalar = float(self.participant_state.get("attention_gain", 0.0) or 0.0)
        return np.full(np.asarray(self.sample_times_s).size, scalar, dtype=float)
