"""Stateful high-level simulator for the Unicorn Hybrid Black API contract.

The real Unicorn Python API exposes configuration, channel lookup, acquisition
start/stop, test-signal mode, GetData, digital outputs and explicit device error
conditions.  This module mirrors those *conceptual* contracts with Pythonic data
objects so applications can exercise lifecycle and fail-closed behavior without
physical hardware.

It is not a binary-compatible replacement for g.tec's licensed Python/C/.NET
libraries.  In particular, the manufacturer does not publicly specify enough
information to reproduce the factory rectangular test signal numerically, so
its frequency/amplitude are explicit simulator policy parameters.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .unicorn_hybrid_black_sim import (
    UNICORN_DEVICE17_NAMES,
    UnicornHybridBlackSimulationConfig,
    UnicornHybridBlackSimulator,
    UnicornHybridBlackSpec,
)

UnicornApiErrorCode = Literal[
    "buffer_overflow",
    "buffer_underflow",
    "connection_problem",
    "operation_not_allowed",
    "invalid_configuration",
]


class UnicornApiSimError(RuntimeError):
    def __init__(self, code: UnicornApiErrorCode, message: str | None = None) -> None:
        self.code = code
        super().__init__(message or code.replace("_", " "))


@dataclass(frozen=True)
class AmplifierChannelSim:
    name: str
    unit: str
    range_min: float
    range_max: float
    enabled: bool = True


@dataclass(frozen=True)
class AmplifierConfigurationSim:
    channels: tuple[AmplifierChannelSim, ...]

    def with_channel(self, name: str, *, enabled: bool) -> "AmplifierConfigurationSim":
        if name not in {channel.name for channel in self.channels}:
            raise KeyError(name)
        return AmplifierConfigurationSim(
            tuple(replace(channel, enabled=enabled) if channel.name == name else channel for channel in self.channels)
        )


@dataclass(frozen=True)
class DeviceInformationSim:
    number_of_eeg_channels: int
    serial: str
    firmware_version: str = "synthetic"
    device_version: str = "Unicorn Hybrid Black (simulated)"
    pcb_version: str = "synthetic"
    enclosure_version: str = "synthetic"


class UnicornPythonApiSimulator:
    """Pythonic lifecycle/configuration twin of the Unicorn API."""

    def __init__(
        self,
        *,
        serial: str = "SIM-UNICORN-0001",
        seed: int = 7,
        test_signal_frequency_hz: float = 5.0,
        test_signal_amplitude_uv: float = 100.0,
    ) -> None:
        if not serial.startswith("SIM-"):
            raise ValueError("synthetic serials must start with 'SIM-' to avoid physical-device ambiguity")
        if test_signal_frequency_hz <= 0 or test_signal_amplitude_uv <= 0:
            raise ValueError("test-signal policy parameters must be positive")
        self.spec = UnicornHybridBlackSpec()
        self.serial = serial
        self.test_signal_frequency_hz = float(test_signal_frequency_hz)
        self.test_signal_amplitude_uv = float(test_signal_amplitude_uv)
        self.device = UnicornHybridBlackSimulator(
            config=UnicornHybridBlackSimulationConfig(
                schema="device17_api",
                seed=seed,
                accelerometer_noise_g=0.0,
                gyroscope_noise_dps=0.0,
            )
        )
        self.configuration = self._default_configuration()
        self.acquiring = False
        self.test_signal_enabled = False
        self.digital_outputs = 0
        self._next_error: UnicornApiErrorCode | None = None

    def _default_configuration(self) -> AmplifierConfigurationSim:
        units = (
            ("microvolts",) * 8
            + ("g",) * 3
            + ("deg/s",) * 3
            + ("count", "percent", "boolean")
        )
        ranges: list[tuple[float, float]] = []
        for index, name in enumerate(UNICORN_DEVICE17_NAMES):
            if index < 8:
                ranges.append((-self.spec.sensitivity_uv, self.spec.sensitivity_uv))
            elif 8 <= index < 14:
                # Motion ranges are intentionally broad simulator bounds.  The
                # device twin does not claim these are amplifier-channel specs.
                ranges.append((-np.inf, np.inf))
            elif name == "Battery Level":
                ranges.append((0.0, 100.0))
            elif name == "Validation Indicator":
                ranges.append((0.0, 1.0))
            else:
                ranges.append((0.0, np.inf))
        return AmplifierConfigurationSim(
            tuple(
                AmplifierChannelSim(name, unit, low, high, True)
                for name, unit, (low, high) in zip(UNICORN_DEVICE17_NAMES, units, ranges, strict=True)
            )
        )

    @staticmethod
    def get_available_devices() -> list[str]:
        return ["SIM-UNICORN-0001"]

    def get_device_information(self) -> DeviceInformationSim:
        return DeviceInformationSim(self.spec.eeg_channels, self.serial)

    def get_configuration(self) -> AmplifierConfigurationSim:
        return self.configuration

    def set_configuration(self, configuration: AmplifierConfigurationSim) -> None:
        if self.acquiring:
            raise UnicornApiSimError("operation_not_allowed", "configuration cannot change during acquisition")
        names = tuple(channel.name for channel in configuration.channels)
        if names != UNICORN_DEVICE17_NAMES or len(set(names)) != 17:
            raise UnicornApiSimError("invalid_configuration", "configuration must preserve the 17 known channels")
        if not any(channel.enabled for channel in configuration.channels[:8]):
            raise UnicornApiSimError("invalid_configuration", "at least one EEG channel must remain enabled")
        self.configuration = configuration

    def get_number_of_acquired_channels(self) -> int:
        return sum(int(channel.enabled) for channel in self.configuration.channels)

    def get_channel_index(self, name: str) -> int:
        enabled = [channel.name for channel in self.configuration.channels if channel.enabled]
        if name not in enabled:
            raise KeyError(name)
        return enabled.index(name)

    def set_digital_outputs(self, value: int) -> None:
        if not 0 <= int(value) <= 255:
            raise ValueError("digital output state must fit in 8 bits")
        self.digital_outputs = int(value)

    def get_digital_outputs(self) -> int:
        return self.digital_outputs

    def start_acquisition(self, test_signal_enabled: bool = False) -> None:
        if self.acquiring:
            raise UnicornApiSimError("operation_not_allowed", "acquisition already started")
        self.acquiring = True
        self.test_signal_enabled = bool(test_signal_enabled)

    def stop_acquisition(self) -> None:
        if not self.acquiring:
            raise UnicornApiSimError("operation_not_allowed", "acquisition is not running")
        self.acquiring = False
        self.test_signal_enabled = False

    def inject_next_error(self, code: UnicornApiErrorCode) -> None:
        if code not in {
            "buffer_overflow",
            "buffer_underflow",
            "connection_problem",
            "operation_not_allowed",
            "invalid_configuration",
        }:
            raise ValueError(f"unsupported API error: {code}")
        self._next_error = code

    def _consume_error(self) -> None:
        if self._next_error is None:
            return
        code = self._next_error
        self._next_error = None
        raise UnicornApiSimError(code)

    def _test_signal(self, times_s: np.ndarray) -> np.ndarray:
        # Transparent simulator policy only.  The current public API/Recorder
        # documentation states that the hardware can emit a rectangular test
        # signal but does not specify enough parameters for numerical cloning.
        phase = np.sin(2.0 * np.pi * self.test_signal_frequency_hz * times_s)
        square = np.where(phase >= 0.0, self.test_signal_amplitude_uv, -self.test_signal_amplitude_uv)
        return np.repeat(square[None, :], 8, axis=0).astype(np.float32)

    def get_data(self, number_of_scans: int) -> np.ndarray:
        """Return scans × enabled-channels float32 data.

        This Pythonic return value represents the same logical scans that the
        physical API writes into a caller-provided float buffer.
        """
        if number_of_scans <= 0:
            raise ValueError("number_of_scans must be positive")
        if not self.acquiring:
            raise UnicornApiSimError("operation_not_allowed", "start acquisition before reading data")
        self._consume_error()
        block = self.device.render(number_of_scans)
        data = block.data.copy()
        if self.test_signal_enabled:
            data[:8] = self._test_signal(block.sample_timestamps_s)
        enabled_indices = [index for index, channel in enumerate(self.configuration.channels) if channel.enabled]
        return np.asarray(data[enabled_indices].T, dtype=np.float32)

    def get_data_bytes(self, number_of_scans: int, *, byte_order: str = "<") -> bytes:
        data = self.get_data(number_of_scans)
        dtype = np.dtype(byte_order + "f4")
        return np.asarray(data, dtype=dtype).tobytes(order="C")
