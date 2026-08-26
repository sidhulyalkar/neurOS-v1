from __future__ import annotations

from neuros.arena import unicorn_hybrid_black_eeg_profile
from neuros.drivers.unicorn_hybrid_black_sim import UNICORN_SCALP_LABELS, UnicornHybridBlackSpec


def test_arena_unicorn_profile_uses_published_hardware_constants():
    spec = UnicornHybridBlackSpec()
    profile = unicorn_hybrid_black_eeg_profile(chunk_samples=5)
    assert profile.name == "unicorn-hybrid-black-eeg8"
    assert profile.sampling_rate_hz == spec.sampling_rate_hz
    assert profile.channel_names == UNICORN_SCALP_LABELS
    assert profile.adc_bits == spec.resolution_bits
    assert profile.input_range_uv == spec.sensitivity_uv
    # Environmental assumptions remain explicit instead of being smuggled in as
    # manufacturer specifications.
    assert profile.sensor_noise_uv == 0.0
    assert profile.line_noise_uv == 0.0


def test_arena_unicorn_profile_allows_declared_environmental_noise_and_clocks():
    profile = unicorn_hybrid_black_eeg_profile(
        sensor_noise_uv=0.5,
        line_frequency_hz=50.0,
        line_noise_uv=1.5,
        clock_offset_ms=7.0,
        clock_drift_ppm=11.0,
        timestamp_jitter_ms=0.2,
        chunk_samples=10,
    )
    profile.validate()
    assert profile.sensor_noise_uv == 0.5
    assert profile.line_frequency_hz == 50.0
    assert profile.line_noise_uv == 1.5
    assert profile.clock_offset_ms == 7.0
    assert profile.clock_drift_ppm == 11.0
    assert profile.timestamp_jitter_ms == 0.2
    assert profile.chunk_samples == 10
