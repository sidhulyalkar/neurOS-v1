from __future__ import annotations

import numpy as np

from neuros.drivers.unicorn_hybrid_black_sim import (
    UnicornHybridBlackSimulationConfig,
    UnicornHybridBlackSimulator,
)
from neuros.drivers.unicorn_network_sim import (
    BANDPOWER_BANDS,
    BANDPOWER_FEATURE_COUNT,
    RAW_UDP_PAYLOAD_BYTES,
    UNICORN_RAW_UDP_NAMES,
    UnicornBandpowerReferenceStream,
    api17_scan_to_raw_udp_order,
    compute_unicorn_bandpower_payload,
    decode_unicorn_bandpower_ascii,
    decode_unicorn_udp_scan,
    encode_unicorn_bandpower_ascii,
    encode_unicorn_udp_scan,
    raw_udp_scan_to_api17_order,
)


def test_raw_udp_payload_is_exactly_one_17_float_scan_with_documented_wire_order():
    sim = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(
            schema="device17_api",
            seed=101,
            accelerometer_noise_g=0.0,
            gyroscope_noise_dps=0.0,
            counter_start=123,
            battery_start_percent=73.0,
        )
    )
    sim.set_motion((0.1, -0.2, 0.95), (1.0, 2.0, 3.0))
    block = sim.render(3)
    payload = encode_unicorn_udp_scan(block, 1)
    assert len(payload) == RAW_UDP_PAYLOAD_BYTES == 68
    decoded_wire = decode_unicorn_udp_scan(payload)
    assert decoded_wire.shape == (17,)
    assert UNICORN_RAW_UDP_NAMES[-3:] == ("BAT", "CNT", "VALID")
    # API17 stores CNT/BAT/VALID at rows 14/15/16. Raw UDP is BAT/CNT/VALID.
    assert np.isclose(decoded_wire[14], block.data[15, 1])
    assert np.isclose(decoded_wire[15], block.data[14, 1])
    assert np.isclose(decoded_wire[16], block.data[16, 1])
    assert np.allclose(raw_udp_scan_to_api17_order(decoded_wire), block.data[:, 1])


def test_api_udp_reorder_is_an_involution_and_does_not_touch_eeg_or_motion():
    values = np.arange(17, dtype=np.float32) + 0.25
    wire = api17_scan_to_raw_udp_order(values)
    assert np.allclose(wire[:14], values[:14])
    assert wire[14] == values[15]
    assert wire[15] == values[14]
    assert wire[16] == values[16]
    assert np.allclose(raw_udp_scan_to_api17_order(wire), values)


def test_bandpower_payload_has_documented_70_value_layout():
    fs = 250.0
    t = np.arange(250) / fs
    eeg = np.vstack([
        np.sin(2 * np.pi * (8.0 + index * 0.25) * t)
        for index in range(8)
    ])
    payload = compute_unicorn_bandpower_payload(eeg, sampling_rate_hz=fs)
    assert payload.shape == (BANDPOWER_FEATURE_COUNT,) == (70,)
    assert len(BANDPOWER_BANDS) == 7
    # First 56 values are seven band-major groups of eight channels.
    assert payload[:56].reshape(7, 8).shape == (7, 8)
    # Remaining values are seven channel averages + seven bipolar averages.
    assert payload[56:63].shape == (7,)
    assert payload[63:70].shape == (7,)


def test_disabled_bandpower_channels_are_nan_but_averages_remain_defined():
    rng = np.random.default_rng(5)
    eeg = rng.normal(size=(8, 250))
    enabled = [True, False, True, True, True, True, True, True]
    payload = compute_unicorn_bandpower_payload(eeg, enabled_channels=enabled)
    per_channel = payload[:56].reshape(7, 8)
    assert np.all(np.isnan(per_channel[:, 1]))
    assert np.all(np.isfinite(payload[56:63]))
    assert np.all(np.isfinite(payload[63:70]))


def test_bandpower_ascii_round_trip_preserves_nan_and_feature_count():
    values = np.arange(70, dtype=float)
    values[9] = np.nan
    encoded = encode_unicorn_bandpower_ascii(values)
    decoded = decode_unicorn_bandpower_ascii(encoded)
    assert decoded.shape == (70,)
    assert np.isnan(decoded[9])
    assert np.allclose(decoded[np.isfinite(decoded)], values[np.isfinite(values)])


def test_default_bandpower_stream_matches_documented_25hz_update_cadence():
    stream = UnicornBandpowerReferenceStream()
    assert stream.buffer_size == 250
    assert stream.buffer_overlap == 240
    assert stream.hop_samples == 10
    assert stream.update_rate_hz == 25.0
    rng = np.random.default_rng(11)
    # First full 250-sample window emits once, then every 10 new samples.
    first = stream.push(rng.normal(size=(8, 250)))
    assert len(first) == 1
    second = stream.push(rng.normal(size=(8, 30)))
    assert len(second) == 3
    assert [frame.sample_index for frame in second] == [259, 269, 279]
