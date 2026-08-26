"""Self-auditing Unicorn Hybrid Black compatibility suite.

The goal is to make a statement such as "tested against the Unicorn simulator"
inspectable. Each surface is tagged by evidence class:

``exact_contract``
    Public documentation specifies the shape/order/cadence strongly enough for
    deterministic compatibility testing.

``reference_implementation``
    Public documentation specifies the interface/layout but not all numerical
    internals. neurOS provides a transparent reference implementation without
    claiming numerical identity.

``synthetic_assumption``
    A useful stress-model policy that is intentionally not attributed to the
    manufacturer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .unicorn_api_sim import UnicornApiSimError, UnicornPythonApiSimulator
from .unicorn_hybrid_black_sim import (
    UNICORN_DEVICE17_NAMES,
    UNICORN_RECORDER19_NAMES,
    UNICORN_SCALP_LABELS,
    UnicornHybridBlackSimulationConfig,
    UnicornHybridBlackSimulator,
    UnicornHybridBlackSpec,
    validate_unicorn_block,
)
from .unicorn_network_sim import (
    BANDPOWER_FEATURE_COUNT,
    RAW_UDP_PAYLOAD_BYTES,
    UNICORN_RAW_UDP_NAMES,
    UnicornBandpowerReferenceStream,
    decode_unicorn_udp_scan,
    encode_unicorn_udp_scan,
)

EvidenceClass = Literal["exact_contract", "reference_implementation", "synthetic_assumption"]


@dataclass(frozen=True)
class UnicornCompatibilitySurface:
    name: str
    evidence_class: EvidenceClass
    passed: bool
    observations: dict[str, object]
    boundary: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "evidence_class": self.evidence_class,
            "passed": self.passed,
            "observations": dict(self.observations),
            "boundary": self.boundary,
        }


@dataclass(frozen=True)
class UnicornCompatibilityReport:
    surfaces: tuple[UnicornCompatibilitySurface, ...]
    device: str = "Unicorn Hybrid Black"
    synthetic: bool = True

    @property
    def passed(self) -> bool:
        return all(surface.passed for surface in self.surfaces)

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.unicorn_hybrid_black_sim.compatibility.v1",
            "device": self.device,
            "synthetic": self.synthetic,
            "passed": self.passed,
            "surfaces": [surface.to_dict() for surface in self.surfaces],
            "evidence_boundary": (
                "Synthetic device/interface compatibility only. This report cannot qualify Bluetooth radio behavior, "
                "physical electrode contact, actual Unicorn firmware, proprietary Bandpower numerics, or human EEG performance."
            ),
        }


def _surface(
    name: str,
    evidence_class: EvidenceClass,
    passed: bool,
    observations: dict[str, object],
    boundary: str,
) -> UnicornCompatibilitySurface:
    return UnicornCompatibilitySurface(name, evidence_class, bool(passed), observations, boundary)


def run_unicorn_compatibility_suite(*, seed: int = 7) -> UnicornCompatibilityReport:
    """Exercise the dependency-light published-interface contracts end to end."""

    spec = UnicornHybridBlackSpec()
    surfaces: list[UnicornCompatibilitySurface] = []

    eeg8 = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(schema="eeg8_anatomical", seed=seed)
    ).render(50)
    eeg8_report = validate_unicorn_block(eeg8)
    surfaces.append(_surface(
        "eeg8_anatomical",
        "exact_contract",
        eeg8_report.passed and eeg8.channel_names == UNICORN_SCALP_LABELS,
        {
            "channels": len(eeg8.channel_names),
            "sampling_rate_hz": spec.sampling_rate_hz,
            "resolution_bits": spec.resolution_bits,
            "sensitivity_uv": spec.sensitivity_uv,
            "eeg_lsb_uv": eeg8.lsb_uv,
        },
        "Published acquisition envelope plus the standard cap montage used by neurOS; does not assert subject physiology.",
    ))

    api_device = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(
            schema="device17_api",
            seed=seed + 1,
            accelerometer_noise_g=0.0,
            gyroscope_noise_dps=0.0,
            counter_start=44,
            battery_start_percent=81.0,
        )
    )
    api_device.set_motion((0.0, 0.0, 1.0), (0.0, 0.0, 0.0))
    api17 = api_device.render(20)
    api17_report = validate_unicorn_block(api17)
    surfaces.append(_surface(
        "direct_api17_scan",
        "exact_contract",
        api17_report.passed and api17.channel_names == UNICORN_DEVICE17_NAMES,
        {
            "channels": len(api17.channel_names),
            "tail": list(api17.channel_names[-3:]),
            "counter_first": int(api17.counter[0]),
        },
        "Mirrors documented acquired-channel order and lifecycle semantics, not the licensed g.tec binary API implementation.",
    ))

    udp_payload = encode_unicorn_udp_scan(api17, 0)
    udp_values = decode_unicorn_udp_scan(udp_payload)
    udp_order_ok = (
        len(udp_payload) == RAW_UDP_PAYLOAD_BYTES
        and tuple(UNICORN_RAW_UDP_NAMES[-3:]) == ("BAT", "CNT", "VALID")
        and np.isclose(udp_values[14], api17.data[15, 0])
        and np.isclose(udp_values[15], api17.data[14, 0])
        and np.isclose(udp_values[16], api17.data[16, 0])
    )
    surfaces.append(_surface(
        "raw_udp17_wire",
        "exact_contract",
        udp_order_ok,
        {
            "payload_bytes": len(udp_payload),
            "channels": len(udp_values),
            "auxiliary_tail": list(UNICORN_RAW_UDP_NAMES[-3:]),
            "nominal_rate_hz": spec.sampling_rate_hz,
        },
        "Wire shape/order/rate are compatibility targets; byte order defaults to little-endian because public docs do not explicitly specify it.",
    ))

    recorder = UnicornHybridBlackSimulator(
        config=UnicornHybridBlackSimulationConfig(schema="recorder19", seed=seed + 2)
    ).render(20)
    recorder_report = validate_unicorn_block(recorder)
    surfaces.append(_surface(
        "recorder19_fields",
        "exact_contract",
        recorder_report.passed and recorder.channel_names == UNICORN_RECORDER19_NAMES,
        {
            "fields": len(recorder.channel_names),
            "tail": list(recorder.channel_names[-5:]),
            "delta_time_ms": float(recorder.data[17, 0]),
        },
        "Matches documented Recorder/network field layout; file-format/GUI behavior is outside this dependency-light contract.",
    ))

    api = UnicornPythonApiSimulator(seed=seed + 3)
    api.start_acquisition(False)
    normal = api.get_data(5)
    api.inject_next_error("buffer_underflow")
    underflow_seen = False
    try:
        api.get_data(5)
    except UnicornApiSimError as exc:
        underflow_seen = exc.code == "buffer_underflow"
    recovered = api.get_data(5)
    api.stop_acquisition()
    surfaces.append(_surface(
        "python_api_lifecycle",
        "exact_contract",
        normal.shape == (5, 17) and recovered.shape == (5, 17) and underflow_seen,
        {
            "acquired_channels": normal.shape[1],
            "underflow_injection_observed": underflow_seen,
            "digital_output_bits": 8,
        },
        "Conceptual API/lifecycle/error semantics only; not a binary-compatible replacement for the licensed Unicorn API.",
    ))

    band = UnicornBandpowerReferenceStream()
    t = np.arange(280, dtype=float) / spec.sampling_rate_hz
    eeg = np.vstack([np.sin(2.0 * np.pi * (8.0 + 0.5 * index) * t) for index in range(8)])
    frames = band.push(eeg)
    band_ok = (
        band.buffer_size == 250
        and band.buffer_overlap == 240
        and band.hop_samples == 10
        and band.update_rate_hz == 25.0
        and len(frames) == 4
        and all(frame.values.shape == (BANDPOWER_FEATURE_COUNT,) for frame in frames)
    )
    surfaces.append(_surface(
        "bandpower70_reference",
        "reference_implementation",
        band_ok,
        {
            "feature_count": BANDPOWER_FEATURE_COUNT,
            "buffer_samples": band.buffer_size,
            "overlap_samples": band.buffer_overlap,
            "update_rate_hz": band.update_rate_hz,
            "frames_from_280_samples": len(frames),
        },
        "Layout/bands/cadence match public documentation; numerical spectral estimator is neurOS reference code, not proprietary parity.",
    ))

    delay = api17.available_timestamps_s - api17.sample_timestamps_s
    surfaces.append(_surface(
        "acquisition_availability_delay",
        "exact_contract",
        bool(np.allclose(delay, spec.device_delay_ms / 1000.0)),
        {
            "modeled_delay_ms": float(np.mean(delay) * 1000.0),
        },
        "Models the documented approximately 40 ms compensation boundary; real hardware jitter/radio scheduling still require measurement.",
    ))

    surfaces.append(_surface(
        "motion_and_battery_stress_policy",
        "synthetic_assumption",
        True,
        {
            "accelerometer_channels": spec.accelerometer_channels,
            "gyroscope_channels": spec.gyroscope_channels,
            "published_minimum_battery_hours": spec.minimum_published_battery_hours,
        },
        "Channel existence and minimum battery-duration provenance are published; sensor noise and discharge curves are explicit synthetic stress policies.",
    ))

    return UnicornCompatibilityReport(tuple(surfaces))
