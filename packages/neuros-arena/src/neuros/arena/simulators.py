"""Display, acquisition-device and transport simulators with explicit ground truth."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .specs import DeviceProfile, DisplayProfile, TransportProfile


DISPLAY_TRACE_MODEL = "neuros.arena.display_trace.v2"


@dataclass(frozen=True)
class StimulusTrace:
    frequency_hz: float | None
    # Command/scheduler clock: when the application requested each frame state.
    command_frame_times_s: np.ndarray
    # Modeled physical-emission clock: command time plus display response lag.
    frame_times_s: np.ndarray
    luminance: np.ndarray
    dropped_frames: np.ndarray
    observed_frequency_hz: float
    frame_drop_fraction: float
    interval_jitter_ms: float
    model: str = DISPLAY_TRACE_MODEL


@dataclass(frozen=True)
class DeviceOutput:
    data_uv: np.ndarray
    timestamps_s: np.ndarray
    channel_names: tuple[str, ...]
    lsb_uv: float
    clipped_fraction: float
    ground_truth_timestamps_s: np.ndarray


@dataclass(frozen=True)
class TransportPacket:
    sequence: int
    generated_s: float
    arrival_s: float
    timestamps_s: np.ndarray
    data_uv: np.ndarray
    source_timestamps_s: np.ndarray
    ground_truth_timestamps_s: np.ndarray


def simulate_stimulus(
    frequency_hz: float | None,
    duration_s: float,
    profile: DisplayProfile,
    seed: int,
) -> StimulusTrace:
    """Simulate commanded display frames and their delayed physical emission.

    ``command_frame_times_s`` is the application/scheduler clock. The coded
    luminance value is evaluated on that clock. ``frame_times_s`` is the modeled
    emission clock after applying the declared constant response lag. Therefore
    a non-zero response lag produces a persistent phase delay:

    ``emitted(t + lag) = commanded(t)``

    rather than merely delaying the first frame and then snapping back onto an
    undelayed global phase. Frame jitter belongs to the command/frame cadence;
    dropped frames hold the previously emitted luminance value.

    This remains a synthetic display model. Physical monitor timing requires
    photodiode or equivalent observation.
    """
    profile.validate()
    if duration_s <= 0:
        raise ValueError("duration_s must be positive")
    if frequency_hz is not None and frequency_hz <= 0:
        raise ValueError("frequency_hz must be positive")
    rng = np.random.default_rng(seed)
    frames = max(2, int(np.ceil(duration_s * profile.refresh_hz)))
    nominal = 1.0 / profile.refresh_hz
    jitter = rng.normal(0.0, profile.frame_jitter_ms / 1000.0, size=frames)
    intervals = np.maximum(nominal * 0.25, nominal + jitter)
    command_frame_times = np.cumsum(intervals) - intervals[0]
    response_lag_s = profile.response_lag_ms / 1000.0
    frame_times = command_frame_times + response_lag_s
    dropped = rng.random(frames) < profile.frame_drop_probability
    luminance = np.full(frames, profile.low_luminance, dtype=float)
    previous = profile.low_luminance
    for index, command_t_s in enumerate(command_frame_times):
        if dropped[index]:
            luminance[index] = previous
            continue
        if frequency_hz is None:
            value = profile.low_luminance
        else:
            value = (
                profile.high_luminance
                if np.sin(2 * np.pi * frequency_hz * command_t_s) >= 0
                else profile.low_luminance
            )
        luminance[index] = value
        previous = value
    transitions = int(np.count_nonzero(np.diff(luminance) != 0))
    span = max(float(frame_times[-1] - frame_times[0]), nominal)
    observed = transitions / (2.0 * span) if frequency_hz is not None else 0.0
    return StimulusTrace(
        frequency_hz=frequency_hz,
        command_frame_times_s=command_frame_times,
        frame_times_s=frame_times,
        luminance=luminance,
        dropped_frames=dropped,
        observed_frequency_hz=float(observed),
        frame_drop_fraction=float(np.mean(dropped)),
        interval_jitter_ms=float(np.std(np.diff(frame_times) - nominal) * 1000.0),
    )


def sample_stimulus(
    trace: StimulusTrace,
    sample_times_s: np.ndarray,
    profile: DisplayProfile,
) -> np.ndarray:
    """Sample-and-hold the physically emitted luminance at EEG sample times.

    ``trace.frame_times_s`` is the modeled emission clock. Before the first
    emitted frame the display remains at the configured low luminance. Values are
    normalized to [-1, 1] around the configured low/high luminance. A no-target
    trace returns zeros so resting baseline does not become an artificial
    negative periodic drive.
    """
    times = np.asarray(sample_times_s, dtype=float)
    if times.ndim != 1:
        raise ValueError("sample_times_s must be 1-D")
    if trace.frequency_hz is None:
        return np.zeros(times.size, dtype=float)
    indices = np.searchsorted(trace.frame_times_s, times, side="right") - 1
    values = np.full(times.size, profile.low_luminance, dtype=float)
    valid = indices >= 0
    if np.any(valid):
        clipped = np.minimum(indices[valid], trace.luminance.size - 1)
        values[valid] = trace.luminance[clipped]
    midpoint = 0.5 * (profile.low_luminance + profile.high_luminance)
    half_range = max(0.5 * (profile.high_luminance - profile.low_luminance), 1e-12)
    return np.clip((values - midpoint) / half_range, -1.0, 1.0)


def apply_device(
    data_uv: np.ndarray,
    timestamps_s: np.ndarray,
    source_channel_names: tuple[str, ...],
    profile: DeviceProfile,
    seed: int,
) -> DeviceOutput:
    profile.validate()
    if data_uv.ndim != 2 or timestamps_s.ndim != 1 or data_uv.shape[1] != timestamps_s.size:
        raise ValueError("data must be channels x samples with one timestamp per sample")
    indices = []
    for name in profile.channel_names:
        if name not in source_channel_names:
            raise ValueError(f"device channel {name!r} is unavailable from source")
        indices.append(source_channel_names.index(name))
    rng = np.random.default_rng(seed)
    selected = np.asarray(data_uv[indices], dtype=float).copy()
    if profile.sensor_noise_uv > 0:
        selected += rng.normal(0.0, profile.sensor_noise_uv, size=selected.shape)
    if profile.line_noise_uv > 0:
        phases = rng.uniform(0.0, 2 * np.pi, size=(selected.shape[0], 1))
        selected += profile.line_noise_uv * np.sin(2 * np.pi * profile.line_frequency_hz * timestamps_s[None, :] + phases)
    half_range = profile.input_range_uv
    clipped = np.abs(selected) > half_range
    selected = np.clip(selected, -half_range, half_range)
    levels = float(2**profile.adc_bits - 1)
    lsb = 2.0 * half_range / levels
    selected = np.round((selected + half_range) / lsb) * lsb - half_range

    truth = np.asarray(timestamps_s, dtype=float).copy()
    clock_scale = 1.0 + profile.clock_drift_ppm * 1e-6
    device_timestamps = profile.clock_offset_ms / 1000.0 + truth * clock_scale
    if profile.timestamp_jitter_ms > 0:
        device_timestamps += rng.normal(0.0, profile.timestamp_jitter_ms / 1000.0, size=truth.size)
    return DeviceOutput(
        data_uv=selected.astype(np.float32),
        timestamps_s=device_timestamps,
        channel_names=profile.channel_names,
        lsb_uv=float(lsb),
        clipped_fraction=float(np.mean(clipped)),
        ground_truth_timestamps_s=truth,
    )


def _clock_metrics(source: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    if source.size < 2:
        return {
            "source_clock_offset_ms_estimated": 0.0,
            "source_clock_drift_ppm_estimated": 0.0,
            "source_timestamp_jitter_rms_ms": 0.0,
        }
    slope, intercept = np.polyfit(truth, source, 1)
    residual = source - (slope * truth + intercept)
    return {
        "source_clock_offset_ms_estimated": float(intercept * 1000.0),
        "source_clock_drift_ppm_estimated": float((slope - 1.0) * 1e6),
        "source_timestamp_jitter_rms_ms": float(np.sqrt(np.mean(residual**2)) * 1000.0),
    }


def packetize(
    data_uv: np.ndarray,
    timestamps_s: np.ndarray,
    chunk_samples: int,
    profile: TransportProfile,
    seed: int,
    ground_truth_timestamps_s: np.ndarray | None = None,
) -> tuple[list[TransportPacket], dict[str, float]]:
    """Packetize a stream while preserving causal/source/corrected clocks.

    ``timestamps_s`` represents the source/device clock. The optional ground-
    truth clock is the simulator's causal time. Downstream ``packet.timestamps_s``
    represents a decoder-facing clock after an ideal correction plus the
    explicitly configured residual synchronization errors. This mirrors the
    distinction between source timestamps and clock-offset correction in LSL
    without pretending to reproduce one specific synchronization algorithm.
    """
    profile.validate()
    if chunk_samples <= 0:
        raise ValueError("chunk_samples must be positive")
    source = np.asarray(timestamps_s, dtype=float)
    truth = source.copy() if ground_truth_timestamps_s is None else np.asarray(ground_truth_timestamps_s, dtype=float)
    if source.ndim != 1 or truth.shape != source.shape or data_uv.shape[1] != source.size:
        raise ValueError("timestamp domains must be 1-D, equal length, and align with data")
    rng = np.random.default_rng(seed)
    packets: list[TransportPacket] = []
    total = 0
    dropped = 0
    correction_scale = 1.0 + profile.clock_correction_drift_error_ppm * 1e-6
    for sequence, start in enumerate(range(0, source.size, chunk_samples)):
        stop = min(source.size, start + chunk_samples)
        total += 1
        generated = float(truth[stop - 1])
        in_silence = any(window_start <= generated < window_start + duration for window_start, duration in profile.silence_windows)
        if in_silence or rng.random() < profile.drop_probability:
            dropped += 1
            continue
        delivery_jitter_s = float(rng.uniform(0.0, profile.jitter_ms / 1000.0)) if profile.jitter_ms > 0 else 0.0
        correction_noise_s = (
            float(rng.normal(0.0, profile.clock_correction_noise_ms / 1000.0))
            if profile.clock_correction_noise_ms > 0
            else 0.0
        )
        corrected = (
            truth[start:stop] * correction_scale
            + profile.clock_correction_offset_error_ms / 1000.0
            + correction_noise_s
        )
        packets.append(TransportPacket(
            sequence=sequence,
            generated_s=generated,
            arrival_s=generated + delivery_jitter_s,
            timestamps_s=corrected.copy(),
            data_uv=data_uv[:, start:stop].copy(),
            source_timestamps_s=source[start:stop].copy(),
            ground_truth_timestamps_s=truth[start:stop].copy(),
        ))
    if profile.reorder_probability > 0:
        for index in range(len(packets) - 1):
            if rng.random() < profile.reorder_probability:
                first, second = packets[index], packets[index + 1]
                packets[index] = TransportPacket(
                    first.sequence,
                    first.generated_s,
                    second.arrival_s + 1e-6,
                    first.timestamps_s,
                    first.data_uv,
                    first.source_timestamps_s,
                    first.ground_truth_timestamps_s,
                )
    packets.sort(key=lambda packet: packet.arrival_s)
    delays = np.asarray([packet.arrival_s - packet.generated_s for packet in packets], dtype=float)
    max_gap = 0.0
    if len(packets) > 1:
        max_gap = float(np.max(np.diff([packet.arrival_s for packet in packets])))
    corrected_errors = np.concatenate(
        [packet.timestamps_s - packet.ground_truth_timestamps_s for packet in packets]
    ) if packets else np.asarray([], dtype=float)
    sequence_inversions = 0
    if len(packets) > 1:
        sequence_inversions = sum(
            int(packets[index + 1].sequence < packets[index].sequence)
            for index in range(len(packets) - 1)
        )
    metrics = {
        "packets_total": float(total),
        "packets_delivered": float(len(packets)),
        "packet_drop_fraction": float(dropped / max(total, 1)),
        "delivery_delay_p95_ms": float(np.percentile(delays, 95) * 1000.0) if delays.size else 0.0,
        "max_arrival_gap_s": max_gap,
        "arrival_sequence_inversion_fraction": float(sequence_inversions / max(len(packets) - 1, 1)),
        "corrected_timestamp_rmse_ms": (
            float(np.sqrt(np.mean(corrected_errors**2)) * 1000.0) if corrected_errors.size else 0.0
        ),
        "corrected_timestamp_p95_abs_ms": (
            float(np.percentile(np.abs(corrected_errors), 95) * 1000.0) if corrected_errors.size else 0.0
        ),
        "corrected_timestamp_max_abs_ms": (
            float(np.max(np.abs(corrected_errors)) * 1000.0) if corrected_errors.size else 0.0
        ),
        **_clock_metrics(source, truth),
    }
    return packets, metrics
