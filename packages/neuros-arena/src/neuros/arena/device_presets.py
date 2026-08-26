"""Canonical acquisition-device presets for Arena worlds.

These presets encode published hardware constants while keeping environmental
assumptions such as line noise, sensor noise, packet chunking, and clock error
explicit at the call site.
"""
from __future__ import annotations

from neuros.drivers.unicorn_hybrid_black_sim import UNICORN_SCALP_LABELS, UnicornHybridBlackSpec

from .specs import DeviceProfile


def unicorn_hybrid_black_eeg_profile(
    *,
    sensor_noise_uv: float = 0.0,
    line_frequency_hz: float = 60.0,
    line_noise_uv: float = 0.0,
    clock_offset_ms: float = 0.0,
    clock_drift_ppm: float = 0.0,
    timestamp_jitter_ms: float = 0.0,
    chunk_samples: int = 5,
) -> DeviceProfile:
    """Return an Arena EEG profile pinned to published Hybrid Black constants.

    The returned profile models only the eight EEG channels because Arena's
    neural-device layer operates on sensor-space EEG. Accelerometer, gyroscope,
    battery, counter and validation channels live in the full
    :class:`~neuros.drivers.unicorn_hybrid_black_sim.UnicornHybridBlackSimulator`
    device twin.

    ``sensor_noise_uv``, ``line_noise_uv``, clock terms and ``chunk_samples`` are
    deliberately caller-controlled environment/runtime assumptions.  They are
    not presented as manufacturer specifications.
    """

    spec = UnicornHybridBlackSpec()
    return DeviceProfile(
        name="unicorn-hybrid-black-eeg8",
        sampling_rate_hz=spec.sampling_rate_hz,
        channel_names=UNICORN_SCALP_LABELS,
        adc_bits=spec.resolution_bits,
        input_range_uv=spec.sensitivity_uv,
        sensor_noise_uv=sensor_noise_uv,
        line_frequency_hz=line_frequency_hz,
        line_noise_uv=line_noise_uv,
        clock_offset_ms=clock_offset_ms,
        clock_drift_ppm=clock_drift_ppm,
        timestamp_jitter_ms=timestamp_jitter_ms,
        chunk_samples=chunk_samples,
    )
