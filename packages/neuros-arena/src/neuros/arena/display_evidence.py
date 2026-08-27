"""Measured display-observation qualification for Synthetic BCI Arena.

This module is an evidence bridge, not another display simulator. It ingests a
measured or synthetic luminance observation, detects transitions with an explicit
hysteresis policy, and compares the observation with one declared
``PresentationEpoch``.

The observation's evidence class and clock alignment are explicit. In
particular, an arbitrary CSV is never silently promoted to physical photodiode
evidence, and onset/phase residuals are not reported unless the caller supplies
a timestamp corresponding to presentation-epoch time zero.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .presentation import PresentationEpoch


DISPLAY_OBSERVATION_SCHEMA = "neuros.arena.display_observation.v1"
DISPLAY_QUALIFICATION_SCHEMA = "neuros.arena.display_qualification.v1"
DISPLAY_TRANSITION_DETECTOR = "neuros.arena.schmitt_transition_detector.v1"

EVIDENCE_CLASSES = frozenset({
    "unverified_observation",
    "synthetic_fixture",
    "measured_photodiode",
    "measured_other",
})


@dataclass(frozen=True)
class DisplayObservation:
    """Timestamped luminance-like observation with explicit provenance.

    ``luminance`` may be physical luminance, photodiode voltage, ADC counts, or a
    synthetic fixture. ``units`` and ``evidence_class`` keep those cases distinct.
    Arena only relies on relative low/high structure for transition timing.
    """

    timestamps_s: np.ndarray
    luminance: np.ndarray
    units: str = "arbitrary"
    source: str = "in-memory"
    evidence_class: str = "unverified_observation"
    source_sha256: str | None = None
    metadata: Mapping[str, str] = field(default_factory=dict)
    schema: str = DISPLAY_OBSERVATION_SCHEMA

    def validate(self) -> None:
        times = np.asarray(self.timestamps_s, dtype=float)
        values = np.asarray(self.luminance, dtype=float)
        if self.schema != DISPLAY_OBSERVATION_SCHEMA:
            raise ValueError(f"expected observation schema {DISPLAY_OBSERVATION_SCHEMA!r}")
        if times.ndim != 1 or values.ndim != 1 or times.shape != values.shape:
            raise ValueError("display observation requires equal-length 1-D timestamp/luminance arrays")
        if times.size < 4:
            raise ValueError("display observation requires at least four samples")
        if not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
            raise ValueError("display observation timestamps/luminance must be finite")
        if np.any(np.diff(times) <= 0):
            raise ValueError("display observation timestamps must be strictly increasing")
        if not isinstance(self.units, str) or not self.units.strip():
            raise ValueError("display observation units must be a non-empty string")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("display observation source must be a non-empty string")
        if self.evidence_class not in EVIDENCE_CLASSES:
            raise ValueError(f"evidence_class must be one of {sorted(EVIDENCE_CLASSES)!r}")
        if self.source_sha256 is not None:
            if not isinstance(self.source_sha256, str) or len(self.source_sha256) != 64:
                raise ValueError("source_sha256 must be a 64-character SHA-256 hex digest")
            try:
                int(self.source_sha256, 16)
            except ValueError as exc:
                raise ValueError("source_sha256 must be hexadecimal") from exc
        for key, value in self.metadata.items():
            if not isinstance(key, str) or not key or not isinstance(value, str):
                raise ValueError("display observation metadata must be non-empty string keys and string values")

    @property
    def duration_s(self) -> float:
        self.validate()
        return float(np.asarray(self.timestamps_s, dtype=float)[-1] - np.asarray(self.timestamps_s, dtype=float)[0])

    def content_sha256(self) -> str:
        """Hash the normalized numeric evidence independently of CSV formatting."""
        self.validate()
        digest = hashlib.sha256()
        digest.update(np.asarray(self.timestamps_s, dtype="<f8").tobytes(order="C"))
        digest.update(np.asarray(self.luminance, dtype="<f8").tobytes(order="C"))
        digest.update(self.units.encode("utf-8"))
        return digest.hexdigest()

    def provenance_dict(self) -> dict[str, Any]:
        self.validate()
        times = np.asarray(self.timestamps_s, dtype=float)
        intervals = np.diff(times)
        median_interval = float(np.median(intervals))
        return {
            "schema": self.schema,
            "source": self.source,
            "source_sha256": self.source_sha256,
            "content_sha256": self.content_sha256(),
            "evidence_class": self.evidence_class,
            "units": self.units,
            "metadata": dict(self.metadata),
            "samples": int(times.size),
            "start_timestamp_s": float(times[0]),
            "end_timestamp_s": float(times[-1]),
            "sample_span_s": float(times[-1] - times[0]),
            "median_sample_interval_ms": median_interval * 1000.0,
            "estimated_sample_rate_hz": (0.0 if median_interval <= 0 else 1.0 / median_interval),
            "sample_interval_jitter_rms_ms": float(np.sqrt(np.mean((intervals - median_interval) ** 2)) * 1000.0),
        }


@dataclass(frozen=True)
class TransitionDetectionConfig:
    """Transparent low/high and hysteresis policy for photodiode-like traces."""

    low_quantile: float = 0.10
    high_quantile: float = 0.90
    hysteresis_fraction: float = 0.20
    minimum_contrast: float = 1e-9
    minimum_transition_separation_s: float = 0.0

    def validate(self) -> None:
        numeric = (
            self.low_quantile,
            self.high_quantile,
            self.hysteresis_fraction,
            self.minimum_contrast,
            self.minimum_transition_separation_s,
        )
        if not all(np.isfinite(value) for value in numeric):
            raise ValueError("transition-detection parameters must be finite")
        if not 0.0 <= self.low_quantile < self.high_quantile <= 1.0:
            raise ValueError("transition quantiles must satisfy 0 <= low < high <= 1")
        if not 0.0 <= self.hysteresis_fraction < 1.0:
            raise ValueError("hysteresis_fraction must be in [0, 1)")
        if self.minimum_contrast < 0 or self.minimum_transition_separation_s < 0:
            raise ValueError("transition minimum contrast/separation must be non-negative")


@dataclass(frozen=True)
class DisplayTransitionTrace:
    transition_times_s: np.ndarray
    directions: tuple[str, ...]
    low_level: float
    high_level: float
    midpoint: float
    low_threshold: float
    high_threshold: float
    contrast: float
    observed_frequency_hz: float
    median_half_period_s: float | None
    interval_jitter_ms: float | None
    detector: str = DISPLAY_TRANSITION_DETECTOR

    def to_dict(self) -> dict[str, Any]:
        return {
            "detector": self.detector,
            "transition_times_s": [float(value) for value in np.asarray(self.transition_times_s, dtype=float)],
            "directions": list(self.directions),
            "transition_count": int(np.asarray(self.transition_times_s).size),
            "low_level": float(self.low_level),
            "high_level": float(self.high_level),
            "midpoint": float(self.midpoint),
            "low_threshold": float(self.low_threshold),
            "high_threshold": float(self.high_threshold),
            "contrast": float(self.contrast),
            "observed_frequency_hz": float(self.observed_frequency_hz),
            "median_half_period_s": (None if self.median_half_period_s is None else float(self.median_half_period_s)),
            "interval_jitter_ms": (None if self.interval_jitter_ms is None else float(self.interval_jitter_ms)),
        }


@dataclass(frozen=True)
class DisplayQualificationConfig:
    detection: TransitionDetectionConfig = field(default_factory=TransitionDetectionConfig)
    epoch_zero_s: float | None = None
    transition_match_tolerance_s: float | None = None

    def validate(self) -> None:
        self.detection.validate()
        if self.epoch_zero_s is not None and not np.isfinite(self.epoch_zero_s):
            raise ValueError("epoch_zero_s must be finite when supplied")
        if self.transition_match_tolerance_s is not None:
            if not np.isfinite(self.transition_match_tolerance_s) or self.transition_match_tolerance_s <= 0:
                raise ValueError("transition_match_tolerance_s must be positive and finite when supplied")


@dataclass(frozen=True)
class DisplayQualificationResult:
    epoch: dict[str, Any]
    observation: dict[str, Any]
    detection_config: dict[str, Any]
    detected: dict[str, Any]
    target_metrics: dict[str, Any]
    aligned_comparison: dict[str, Any] | None
    evidence_boundary: str
    schema: str = DISPLAY_QUALIFICATION_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "epoch": self.epoch,
            "observation": self.observation,
            "detection_config": self.detection_config,
            "detected": self.detected,
            "target_metrics": self.target_metrics,
            "aligned_comparison": self.aligned_comparison,
            "evidence_boundary": self.evidence_boundary,
        }


def load_display_observation_csv(
    path: str | Path,
    *,
    timestamp_column: str = "timestamp_s",
    luminance_column: str = "luminance",
    units: str = "arbitrary",
    evidence_class: str = "unverified_observation",
    source: str | None = None,
    metadata: Mapping[str, str] | None = None,
) -> DisplayObservation:
    """Load a two-column timestamp/luminance observation with byte provenance."""
    input_path = Path(path)
    raw = input_path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("display observation CSV must be UTF-8 text") from exc
    reader = csv.DictReader(io.StringIO(text))
    fieldnames = reader.fieldnames or []
    if timestamp_column not in fieldnames or luminance_column not in fieldnames:
        raise ValueError(
            f"display observation CSV requires columns {timestamp_column!r} and {luminance_column!r}"
        )
    timestamps: list[float] = []
    luminance: list[float] = []
    for row_index, row in enumerate(reader, start=2):
        try:
            timestamps.append(float(row[timestamp_column]))
            luminance.append(float(row[luminance_column]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid numeric display observation at CSV row {row_index}") from exc
    observation = DisplayObservation(
        timestamps_s=np.asarray(timestamps, dtype=float),
        luminance=np.asarray(luminance, dtype=float),
        units=units,
        source=(str(input_path) if source is None else source),
        evidence_class=evidence_class,
        source_sha256=hashlib.sha256(raw).hexdigest(),
        metadata={} if metadata is None else dict(metadata),
    )
    observation.validate()
    return observation


def _interpolate_crossing_time(
    t0: float,
    t1: float,
    y0: float,
    y1: float,
    threshold: float,
) -> float:
    delta = y1 - y0
    if abs(delta) <= 1e-15:
        return float(t1)
    fraction = float(np.clip((threshold - y0) / delta, 0.0, 1.0))
    return float(t0 + fraction * (t1 - t0))


def detect_display_transitions(
    observation: DisplayObservation,
    config: TransitionDetectionConfig | None = None,
) -> DisplayTransitionTrace:
    """Detect low/high transitions using a robust quantile Schmitt trigger."""
    observation.validate()
    policy = config or TransitionDetectionConfig()
    policy.validate()
    times = np.asarray(observation.timestamps_s, dtype=float)
    values = np.asarray(observation.luminance, dtype=float)

    low = float(np.quantile(values, policy.low_quantile))
    high = float(np.quantile(values, policy.high_quantile))
    contrast = high - low
    if contrast <= policy.minimum_contrast:
        raise ValueError(
            "display observation has insufficient low/high contrast for transition detection"
        )
    midpoint = 0.5 * (low + high)
    half_hysteresis = 0.5 * policy.hysteresis_fraction * contrast
    low_threshold = midpoint - half_hysteresis
    high_threshold = midpoint + half_hysteresis

    state_high = bool(values[0] >= midpoint)
    transition_times: list[float] = []
    directions: list[str] = []
    last_transition = -np.inf
    for index in range(1, times.size):
        crossed = False
        direction = ""
        if not state_high and values[index] >= high_threshold:
            crossed = True
            direction = "rising"
        elif state_high and values[index] <= low_threshold:
            crossed = True
            direction = "falling"
        if not crossed:
            continue
        crossing = _interpolate_crossing_time(
            float(times[index - 1]),
            float(times[index]),
            float(values[index - 1]),
            float(values[index]),
            midpoint,
        )
        # Even when a crossing is suppressed as too close to the previous one,
        # update Schmitt state so the detector cannot repeatedly rediscover the
        # same excursion on every following sample.
        state_high = direction == "rising"
        if crossing - last_transition + 1e-15 < policy.minimum_transition_separation_s:
            continue
        transition_times.append(crossing)
        directions.append(direction)
        last_transition = crossing

    detected = np.asarray(transition_times, dtype=float)
    if detected.size >= 2:
        intervals = np.diff(detected)
        half_period = float(np.median(intervals))
        observed_frequency = 0.0 if half_period <= 0 else 1.0 / (2.0 * half_period)
        interval_jitter_ms = float(np.sqrt(np.mean((intervals - half_period) ** 2)) * 1000.0)
    else:
        half_period = None
        observed_frequency = 0.0
        interval_jitter_ms = None

    return DisplayTransitionTrace(
        transition_times_s=detected,
        directions=tuple(directions),
        low_level=low,
        high_level=high,
        midpoint=midpoint,
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        contrast=contrast,
        observed_frequency_hz=float(observed_frequency),
        median_half_period_s=half_period,
        interval_jitter_ms=interval_jitter_ms,
    )


def _planned_transition_times(epoch: PresentationEpoch) -> np.ndarray:
    trace = epoch.trace
    if trace.luminance.size < 2:
        return np.asarray([], dtype=float)
    changed = np.flatnonzero(np.diff(np.asarray(trace.luminance, dtype=float)) != 0) + 1
    return np.asarray(trace.frame_times_s, dtype=float)[changed]


def _default_match_tolerance(epoch: PresentationEpoch) -> float:
    planned = _planned_transition_times(epoch)
    if planned.size >= 2:
        half_period = float(np.median(np.diff(planned)))
        return max(1e-6, 0.35 * half_period)
    if epoch.target_frequency_hz is not None and epoch.target_frequency_hz > 0:
        return 0.35 / (2.0 * float(epoch.target_frequency_hz))
    return 0.010


def _match_transition_times(
    planned_s: np.ndarray,
    measured_s: np.ndarray,
    tolerance_s: float,
) -> dict[str, Any]:
    """Order-preserving transition matching without inventing clock alignment."""
    i = 0
    j = 0
    residuals: list[float] = []
    matched_pairs: list[tuple[float, float]] = []
    missed = 0
    extra = 0
    while i < planned_s.size and j < measured_s.size:
        delta = float(measured_s[j] - planned_s[i])
        if abs(delta) <= tolerance_s:
            residuals.append(delta)
            matched_pairs.append((float(planned_s[i]), float(measured_s[j])))
            i += 1
            j += 1
        elif measured_s[j] < planned_s[i] - tolerance_s:
            extra += 1
            j += 1
        else:
            missed += 1
            i += 1
    missed += int(planned_s.size - i)
    extra += int(measured_s.size - j)
    residual = np.asarray(residuals, dtype=float)
    return {
        "transition_match_tolerance_ms": float(tolerance_s * 1000.0),
        "planned_transition_count": int(planned_s.size),
        "measured_transition_count_in_epoch_window": int(measured_s.size),
        "matched_transition_count": int(residual.size),
        "missed_transition_count": int(missed),
        "extra_transition_count": int(extra),
        "match_fraction_of_planned": (
            1.0 if planned_s.size == 0 and measured_s.size == 0
            else float(residual.size / max(planned_s.size, 1))
        ),
        "timing_residual_mean_ms": (None if residual.size == 0 else float(np.mean(residual) * 1000.0)),
        "timing_residual_rmse_ms": (None if residual.size == 0 else float(np.sqrt(np.mean(residual ** 2)) * 1000.0)),
        "timing_residual_p95_abs_ms": (None if residual.size == 0 else float(np.percentile(np.abs(residual), 95) * 1000.0)),
        "timing_residual_max_abs_ms": (None if residual.size == 0 else float(np.max(np.abs(residual)) * 1000.0)),
        "first_matched_transition_residual_ms": (None if residual.size == 0 else float(residual[0] * 1000.0)),
        "matched_transition_pairs_s": [
            {"planned": planned, "measured": measured}
            for planned, measured in matched_pairs
        ],
        "transition_polarity_compared": False,
    }


def _evidence_boundary(evidence_class: str, aligned: bool) -> str:
    timing = (
        "Aligned transition timing is reported because an explicit epoch-zero timestamp was supplied. "
        if aligned
        else "No epoch-zero alignment was supplied, so onset/phase timing residuals are intentionally omitted. "
    )
    if evidence_class == "measured_photodiode":
        return (
            "Measured photodiode evidence may support physical display-emission claims for this captured setup only. "
            + timing
            + "It does not establish human neural response, decoder accuracy, closed-loop efficacy, or clinical validity."
        )
    if evidence_class == "synthetic_fixture":
        return (
            "Synthetic fixture evidence validates the display-observation analysis software only; it is not physical display evidence. "
            + timing
            + "It does not establish human neural or application performance."
        )
    return (
        "Observation provenance is not sufficient to treat this result as qualified photodiode evidence. "
        + timing
        + "It does not establish human neural response, decoder accuracy, closed-loop efficacy, or clinical validity."
    )


def qualify_display_observation(
    epoch: PresentationEpoch,
    observation: DisplayObservation,
    config: DisplayQualificationConfig | None = None,
) -> DisplayQualificationResult:
    """Compare one observation with one declared Arena presentation epoch."""
    observation.validate()
    policy = config or DisplayQualificationConfig()
    policy.validate()
    detected = detect_display_transitions(observation, policy.detection)

    target_frequency = epoch.target_frequency_hz
    observed_frequency = detected.observed_frequency_hz
    if target_frequency is None:
        absolute_error = None
        relative_error = None
    else:
        absolute_error = float(abs(observed_frequency - float(target_frequency)))
        relative_error = float(absolute_error / float(target_frequency))
    target_metrics = {
        "target_frequency_hz": target_frequency,
        "observed_frequency_hz": float(observed_frequency),
        "absolute_frequency_error_hz": absolute_error,
        "relative_frequency_error": relative_error,
        "frequency_error_ppm": (None if relative_error is None else float(relative_error * 1e6)),
        "observed_transition_count": int(np.asarray(detected.transition_times_s).size),
        "observed_interval_jitter_ms": detected.interval_jitter_ms,
        "low_high_contrast": float(detected.contrast),
    }

    aligned: dict[str, Any] | None = None
    if policy.epoch_zero_s is not None:
        local_measured = np.asarray(detected.transition_times_s, dtype=float) - float(policy.epoch_zero_s)
        tolerance = policy.transition_match_tolerance_s or _default_match_tolerance(epoch)
        # Only transitions plausibly belonging to this epoch participate in the
        # alignment comparison. The tolerance margin admits boundary jitter while
        # preventing neighboring presentation events from being counted as extras.
        epoch_duration = epoch.sample_count
        # PresentationEpoch does not carry fs directly. Its trace duration is the
        # physical epoch duration for this comparison.
        if epoch.trace.frame_times_s.size:
            trace_end = float(epoch.trace.frame_times_s[-1])
        else:
            trace_end = 0.0
        in_window = (local_measured >= -tolerance) & (local_measured <= trace_end + tolerance)
        aligned = {
            "clock_alignment": "explicit_epoch_zero",
            "epoch_zero_timestamp_s": float(policy.epoch_zero_s),
            "planned_clock_domain": "presentation_epoch_local_seconds",
            "observation_clock_domain": "observation_timestamp_seconds_shifted_by_epoch_zero",
            **_match_transition_times(
                _planned_transition_times(epoch),
                local_measured[in_window],
                float(tolerance),
            ),
        }
    epoch_summary = {
        "presentation_epoch_index": int(epoch.index),
        "start_sample": int(epoch.start_sample),
        "end_sample": int(epoch.end_sample),
        "sample_count": int(epoch.sample_count),
        "target_frequency_hz": epoch.target_frequency_hz,
        "stimulus_id": epoch.stimulus_id,
        "stage_indices": [int(value) for value in epoch.stage_indices],
        "planned_observed_frequency_hz": float(epoch.trace.observed_frequency_hz),
        "planned_frame_drop_fraction": float(epoch.trace.frame_drop_fraction),
        "planned_interval_jitter_ms": float(epoch.trace.interval_jitter_ms),
    }
    return DisplayQualificationResult(
        epoch=epoch_summary,
        observation=observation.provenance_dict(),
        detection_config=asdict(policy.detection) | {
            "epoch_zero_s": policy.epoch_zero_s,
            "transition_match_tolerance_s": policy.transition_match_tolerance_s,
        },
        detected=detected.to_dict(),
        target_metrics=target_metrics,
        aligned_comparison=aligned,
        evidence_boundary=_evidence_boundary(
            observation.evidence_class,
            policy.epoch_zero_s is not None,
        ),
    )


def save_display_qualification(
    result: DisplayQualificationResult,
    path: str | Path,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    return output
