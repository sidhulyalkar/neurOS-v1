"""Device-faithful Unicorn Hybrid Black simulation contracts.

This module models the *acquisition/interface behavior* of the Unicorn Hybrid
Black around an EEG source. It deliberately does not claim to be a human
physiology simulator. Neural voltages come from ``SyntheticEEGGenerator`` (or,
in higher-level Arena integrations, another world model); this layer adds the
hardware-facing channel schemas, quantization/clipping envelope, motion/auxiliary
telemetry, counter continuity, battery state, validation state, and acquisition
availability timing that BCI applications need to exercise before hardware is
available.

Specification constants are intentionally restricted to values published by
current g.tec/Unicorn documentation:

* 8 EEG channels at 250 Hz;
* 24-bit resolution;
* sensitivity ±750 mV;
* input impedance >1 GOhm (recorded here as a lower bound, not simulated as an
  impedance measurement);
* 3-axis accelerometer and 3-axis gyroscope;
* acquired auxiliary counter, battery and validation channels;
* about 40 ms device-delay compensation in g.Pype's HybridBlack source.

The Recorder/network 19-field view additionally exposes delta-time and status /
trigger fields. Those two fields are application/interface products rather than
extra amplifier electrodes.

Random stress processes use independent seeded streams and time-major draws so
replay does not depend on how a caller partitions an otherwise identical sample
sequence across ``render()`` calls.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .synthetic_eeg import SyntheticEEGConfig, SyntheticEEGGenerator

UnicornSchema = Literal["eeg8_anatomical", "device17_api", "recorder19"]

UNICORN_EEG_API_NAMES = tuple(f"EEG {index}" for index in range(1, 9))
UNICORN_SCALP_LABELS = ("Fz", "C3", "Cz", "C4", "Pz", "PO7", "Oz", "PO8")
UNICORN_ACCEL_NAMES = ("Accelerometer X", "Accelerometer Y", "Accelerometer Z")
UNICORN_GYRO_NAMES = ("Gyroscope X", "Gyroscope Y", "Gyroscope Z")
UNICORN_AUX_NAMES = ("Counter", "Battery Level", "Validation Indicator")
UNICORN_DEVICE17_NAMES = (
    UNICORN_EEG_API_NAMES
    + UNICORN_ACCEL_NAMES
    + UNICORN_GYRO_NAMES
    + UNICORN_AUX_NAMES
)
UNICORN_RECORDER19_NAMES = (
    UNICORN_EEG_API_NAMES
    + ("ACC X", "ACC Y", "ACC Z")
    + ("GYR X", "GYR Y", "GYR Z")
    + ("CNT", "BAT", "VALID", "DT", "STATUS")
)


@dataclass(frozen=True)
class UnicornHybridBlackSpec:
    """Published device constants used by the simulator.

    ``input_impedance_lower_bound_ohm`` is descriptive provenance only. The
    physical headset specification states >1 GOhm; the simulator does not claim
    to infer or reproduce electrode-skin impedance.
    """

    sampling_rate_hz: float = 250.0
    eeg_channels: int = 8
    resolution_bits: int = 24
    sensitivity_uv: float = 750_000.0
    input_impedance_lower_bound_ohm: float = 1_000_000_000.0
    accelerometer_channels: int = 3
    gyroscope_channels: int = 3
    acquired_channels: int = 17
    device_delay_ms: float = 40.0
    minimum_published_battery_hours: float = 3.0

    @property
    def eeg_lsb_uv(self) -> float:
        return 2.0 * self.sensitivity_uv / float(2**self.resolution_bits - 1)

    def validate(self) -> None:
        if self.sampling_rate_hz != 250.0:
            raise ValueError("Unicorn Hybrid Black simulation is fixed at 250 Hz")
        if self.eeg_channels != 8 or self.acquired_channels != 17:
            raise ValueError("Unicorn Hybrid Black channel-count constants are fixed")
        if self.resolution_bits != 24:
            raise ValueError("Unicorn Hybrid Black resolution is fixed at 24 bits")
        if self.sensitivity_uv <= 0 or self.device_delay_ms < 0:
            raise ValueError("invalid Unicorn hardware constants")


@dataclass(frozen=True)
class UnicornHybridBlackSimulationConfig:
    """Simulation policy around the published hardware contract.

    Parameters such as battery discharge shape and motion-sensor noise are
    explicit *stress-model assumptions*, not manufacturer specifications.
    """

    schema: UnicornSchema = "device17_api"
    seed: int = 7
    battery_start_percent: float = 100.0
    battery_runtime_hours: float = 3.0
    accelerometer_noise_g: float = 0.002
    gyroscope_noise_dps: float = 0.05
    acquisition_delay_jitter_ms: float = 0.0
    counter_start: int = 0
    validation_default: int = 1
    status_default: int = 0

    def validate(self) -> None:
        if self.schema not in {"eeg8_anatomical", "device17_api", "recorder19"}:
            raise ValueError(f"unsupported Unicorn schema: {self.schema!r}")
        if not 0.0 <= self.battery_start_percent <= 100.0:
            raise ValueError("battery_start_percent must be in [0, 100]")
        if self.battery_runtime_hours <= 0:
            raise ValueError("battery_runtime_hours must be positive")
        if self.accelerometer_noise_g < 0 or self.gyroscope_noise_dps < 0:
            raise ValueError("motion-sensor noise must be non-negative")
        if self.acquisition_delay_jitter_ms < 0:
            raise ValueError("acquisition_delay_jitter_ms must be non-negative")
        if self.validation_default not in {0, 1}:
            raise ValueError("validation_default must be 0 or 1")


@dataclass(frozen=True)
class UnicornHybridBlackBlock:
    """One simulated acquisition block.

    ``sample_timestamps_s`` represent causal sample times. ``available_timestamps_s``
    represent when those samples become available to host software after the
    configured device-delay model. Keeping these separate prevents a transport
    latency from silently becoming a neural timestamp.
    """

    data: np.ndarray
    channel_names: tuple[str, ...]
    channel_units: tuple[str, ...]
    sample_timestamps_s: np.ndarray
    available_timestamps_s: np.ndarray
    eeg_data_uv: np.ndarray
    counter: np.ndarray
    battery_percent: np.ndarray
    validation: np.ndarray
    status: np.ndarray
    clipped_fraction: float
    lsb_uv: float
    schema: UnicornSchema
    synthetic: bool = True
    emulated_device: str = "Unicorn Hybrid Black"


@dataclass(frozen=True)
class UnicornConformanceReport:
    """Dependency-light structural conformance report for a simulated block."""

    passed: bool
    checks: dict[str, bool]
    metrics: dict[str, float]
    evidence_boundary: str = (
        "Device/interface simulation conformance only; not physical-hardware or human-EEG validation."
    )

    def to_dict(self) -> dict:
        return {
            "schema": "neuros.unicorn_hybrid_black_sim.conformance.v1",
            "passed": self.passed,
            "checks": dict(self.checks),
            "metrics": dict(self.metrics),
            "evidence_boundary": self.evidence_boundary,
        }


class UnicornHybridBlackSimulator:
    """Synthetic EEG wrapped in a Unicorn Hybrid Black acquisition contract."""

    def __init__(
        self,
        eeg_generator: SyntheticEEGGenerator | None = None,
        *,
        config: UnicornHybridBlackSimulationConfig | None = None,
        spec: UnicornHybridBlackSpec | None = None,
    ) -> None:
        self.spec = spec or UnicornHybridBlackSpec()
        self.spec.validate()
        self.config = config or UnicornHybridBlackSimulationConfig()
        self.config.validate()
        if eeg_generator is None:
            eeg_generator = SyntheticEEGGenerator(
                SyntheticEEGConfig(
                    sampling_rate_hz=self.spec.sampling_rate_hz,
                    channel_names=UNICORN_SCALP_LABELS,
                    seed=self.config.seed,
                )
            )
        if tuple(eeg_generator.config.channel_names) != UNICORN_SCALP_LABELS:
            raise ValueError(
                "Unicorn simulator expects the standard Fz/C3/Cz/C4/Pz/PO7/Oz/PO8 source montage"
            )
        if float(eeg_generator.config.sampling_rate_hz) != self.spec.sampling_rate_hz:
            raise ValueError("Unicorn simulator source must run at 250 Hz")
        self.eeg = eeg_generator

        timing_seed, accel_seed, gyro_seed = np.random.SeedSequence(
            self.config.seed + 6011
        ).spawn(3)
        self._availability_rng = np.random.default_rng(timing_seed)
        self._accel_rng = np.random.default_rng(accel_seed)
        self._gyro_rng = np.random.default_rng(gyro_seed)

        self.accel_g = np.asarray([0.0, 0.0, 1.0], dtype=float)
        self.gyro_dps = np.zeros(3, dtype=float)
        self.validation_value = int(self.config.validation_default)
        self.status_value = int(self.config.status_default)
        self.counter_value = int(self.config.counter_start)

    @property
    def schema(self) -> UnicornSchema:
        return self.config.schema

    def set_motion(
        self,
        accel_xyz_g: tuple[float, float, float],
        gyro_xyz_dps: tuple[float, float, float],
    ) -> None:
        accel = np.asarray(accel_xyz_g, dtype=float)
        gyro = np.asarray(gyro_xyz_dps, dtype=float)
        if (
            accel.shape != (3,)
            or gyro.shape != (3,)
            or not np.all(np.isfinite(accel))
            or not np.all(np.isfinite(gyro))
        ):
            raise ValueError("motion vectors must contain three finite values")
        self.accel_g = accel
        self.gyro_dps = gyro

    def set_validation(self, value: int) -> None:
        if value not in {0, 1}:
            raise ValueError("validation indicator must be 0 or 1")
        self.validation_value = int(value)

    def set_status(self, value: int) -> None:
        self.status_value = int(value)

    def _quantize_eeg(self, eeg_uv: np.ndarray) -> tuple[np.ndarray, float]:
        half_range = self.spec.sensitivity_uv
        clipped_mask = np.abs(eeg_uv) > half_range
        clipped = np.clip(np.asarray(eeg_uv, dtype=float), -half_range, half_range)
        lsb = self.spec.eeg_lsb_uv
        quantized = np.round((clipped + half_range) / lsb) * lsb - half_range
        return quantized.astype(np.float32), float(np.mean(clipped_mask))

    def _battery(self, sample_times_s: np.ndarray) -> np.ndarray:
        runtime_s = self.config.battery_runtime_hours * 3600.0
        elapsed = np.maximum(0.0, np.asarray(sample_times_s, dtype=float))
        fraction = np.clip(elapsed / runtime_s, 0.0, 1.0)
        return np.maximum(
            0.0,
            self.config.battery_start_percent * (1.0 - fraction),
        )

    def _availability(self, sample_times_s: np.ndarray) -> np.ndarray:
        base = (
            np.asarray(sample_times_s, dtype=float)
            + self.spec.device_delay_ms / 1000.0
        )
        if self.config.acquisition_delay_jitter_ms <= 0:
            return base
        jitter = self._availability_rng.normal(
            0.0,
            self.config.acquisition_delay_jitter_ms / 1000.0,
            size=base.size,
        )
        return base + jitter

    def _motion(self, samples: int) -> tuple[np.ndarray, np.ndarray]:
        accel = np.repeat(self.accel_g[:, None], samples, axis=1)
        gyro = np.repeat(self.gyro_dps[:, None], samples, axis=1)
        if self.config.accelerometer_noise_g:
            # Time-major draws make one N-sample render equivalent to any
            # partition of the same N samples.
            accel += self._accel_rng.normal(
                0.0,
                self.config.accelerometer_noise_g,
                size=(samples, 3),
            ).T
        if self.config.gyroscope_noise_dps:
            gyro += self._gyro_rng.normal(
                0.0,
                self.config.gyroscope_noise_dps,
                size=(samples, 3),
            ).T
        return accel.astype(np.float32), gyro.astype(np.float32)

    def render(self, samples: int) -> UnicornHybridBlackBlock:
        if samples <= 0:
            raise ValueError("samples must be positive")
        source = self.eeg.render(samples)
        eeg_uv, clipped_fraction = self._quantize_eeg(source.data_uv)
        times = np.asarray(source.timestamps_s, dtype=float)
        available = self._availability(times)
        accel, gyro = self._motion(samples)
        counter = self.counter_value + np.arange(samples, dtype=np.int64)
        self.counter_value += samples
        battery = self._battery(times).astype(np.float32)
        validation = np.full(samples, self.validation_value, dtype=np.float32)
        status = np.full(samples, self.status_value, dtype=np.float32)

        if self.schema == "eeg8_anatomical":
            data = eeg_uv
            names = UNICORN_SCALP_LABELS
            units = ("microvolts",) * 8
        elif self.schema == "device17_api":
            data = np.vstack(
                [
                    eeg_uv,
                    accel,
                    gyro,
                    counter.astype(np.float32)[None, :],
                    battery[None, :],
                    validation[None, :],
                ]
            )
            names = UNICORN_DEVICE17_NAMES
            units = (
                ("microvolts",) * 8
                + ("g",) * 3
                + ("deg/s",) * 3
                + ("count", "percent", "boolean")
            )
        else:
            delta_ms = np.full(
                samples,
                1000.0 / self.spec.sampling_rate_hz,
                dtype=np.float32,
            )
            data = np.vstack(
                [
                    eeg_uv,
                    accel,
                    gyro,
                    counter.astype(np.float32)[None, :],
                    battery[None, :],
                    validation[None, :],
                    delta_ms[None, :],
                    status[None, :],
                ]
            )
            names = UNICORN_RECORDER19_NAMES
            units = (
                ("microvolts",) * 8
                + ("g",) * 3
                + ("deg/s",) * 3
                + ("count", "percent", "boolean", "ms", "code")
            )

        return UnicornHybridBlackBlock(
            data=np.asarray(data, dtype=np.float32),
            channel_names=tuple(names),
            channel_units=tuple(units),
            sample_timestamps_s=times,
            available_timestamps_s=available,
            eeg_data_uv=eeg_uv,
            counter=counter,
            battery_percent=battery,
            validation=validation,
            status=status,
            clipped_fraction=clipped_fraction,
            lsb_uv=self.spec.eeg_lsb_uv,
            schema=self.schema,
        )


def validate_unicorn_block(
    block: UnicornHybridBlackBlock,
    *,
    spec: UnicornHybridBlackSpec | None = None,
) -> UnicornConformanceReport:
    """Validate observable invariants of one simulated acquisition block."""

    device = spec or UnicornHybridBlackSpec()
    samples = int(block.data.shape[1]) if block.data.ndim == 2 else 0
    expected_channels = {
        "eeg8_anatomical": 8,
        "device17_api": 17,
        "recorder19": 19,
    }[block.schema]
    if samples > 1:
        sample_dt = np.diff(block.sample_timestamps_s)
        available_delay_ms = (
            block.available_timestamps_s - block.sample_timestamps_s
        ) * 1000.0
        mean_rate = 1.0 / float(np.mean(sample_dt))
        counter_continuity = bool(np.all(np.diff(block.counter) == 1))
        mean_delay = float(np.mean(available_delay_ms))
    else:
        mean_rate = device.sampling_rate_hz
        counter_continuity = True
        mean_delay = (
            float(
                np.mean(
                    (block.available_timestamps_s - block.sample_timestamps_s)
                    * 1000.0
                )
            )
            if samples
            else 0.0
        )
    checks = {
        "two_dimensional_samples": block.data.ndim == 2 and samples > 0,
        "expected_channel_count": (
            block.data.ndim == 2 and block.data.shape[0] == expected_channels
        ),
        "channel_metadata_count": (
            len(block.channel_names) == expected_channels
            and len(block.channel_units) == expected_channels
        ),
        "250hz_sample_clock": abs(mean_rate - device.sampling_rate_hz) < 1e-6,
        "counter_continuity": counter_continuity,
        "battery_range": bool(
            np.all(
                (block.battery_percent >= 0.0)
                & (block.battery_percent <= 100.0)
            )
        ),
        "validation_binary": bool(np.all(np.isin(block.validation, [0.0, 1.0]))),
        "eeg_within_sensitivity": bool(
            np.all(
                np.abs(block.eeg_data_uv)
                <= device.sensitivity_uv + block.lsb_uv
            )
        ),
        "published_quantization_lsb": (
            abs(block.lsb_uv - device.eeg_lsb_uv) < 1e-12
        ),
        "availability_not_before_sample": bool(
            np.all(block.available_timestamps_s >= block.sample_timestamps_s)
        ),
    }
    if block.schema == "device17_api":
        checks["api_channel_order"] = block.channel_names == UNICORN_DEVICE17_NAMES
    elif block.schema == "recorder19":
        checks["recorder_channel_order"] = (
            block.channel_names == UNICORN_RECORDER19_NAMES
        )
        checks["recorder_delta_time"] = bool(
            np.allclose(block.data[17], 4.0, atol=1e-6)
        )
    else:
        checks["anatomical_eeg_order"] = block.channel_names == UNICORN_SCALP_LABELS
    metrics = {
        "observed_sampling_rate_hz": float(mean_rate),
        "mean_availability_delay_ms": mean_delay,
        "eeg_lsb_uv": float(block.lsb_uv),
        "clipped_fraction": float(block.clipped_fraction),
        "battery_min_percent": (
            float(np.min(block.battery_percent)) if samples else 0.0
        ),
        "battery_max_percent": (
            float(np.max(block.battery_percent)) if samples else 0.0
        ),
    }
    return UnicornConformanceReport(
        passed=all(checks.values()),
        checks=checks,
        metrics=metrics,
    )
