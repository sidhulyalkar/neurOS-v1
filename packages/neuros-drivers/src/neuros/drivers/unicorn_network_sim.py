"""Protocol-compatible Unicorn Hybrid Black network simulation helpers.

The raw UDP encoder follows the documented Unicorn UDP wire contract: one scan
contains 17 float32 values (68 bytes) and is emitted at 250 Hz.

A subtle but important compatibility detail is preserved explicitly: the
standalone raw-UDP payload orders its auxiliary tail as ``BAT, CNT, VALID``,
whereas the direct API/Recorder acquisition order is ``Counter, Battery,
Validation``. The simulator therefore reorders an API17 device block at the
wire boundary instead of pretending those interfaces share one schema.

The Bandpower helper follows the documented *payload layout, frequency bands,
window/hop cadence, disabled-channel NaN behavior, and 70-value ASCII shape* of
Unicorn Bandpower. The exact proprietary numerical estimator is not documented,
so neurOS deliberately uses a transparent reference FFT estimator and does not
claim bit-for-bit numerical equivalence to the Unicorn Bandpower application.
"""
from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Iterable

import numpy as np

from .unicorn_hybrid_black_sim import UNICORN_EEG_API_NAMES, UnicornHybridBlackBlock

BANDPOWER_BANDS: tuple[tuple[str, float, float], ...] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 12.0),
    ("beta_low", 12.0, 16.0),
    ("beta_mid", 16.0, 20.0),
    ("beta_high", 20.0, 30.0),
    ("gamma", 30.0, 50.0),
)
BANDPOWER_FEATURE_COUNT = 70
RAW_UDP_CHANNEL_COUNT = 17
RAW_UDP_PAYLOAD_BYTES = RAW_UDP_CHANNEL_COUNT * 4
UNICORN_RAW_UDP_NAMES = (
    UNICORN_EEG_API_NAMES
    + ("ACC X", "ACC Y", "ACC Z")
    + ("GYR X", "GYR Y", "GYR Z")
    + ("BAT", "CNT", "VALID")
)
# API17 rows are EEG[0:8], accel[8:11], gyro[11:14], CNT[14], BAT[15],
# VALID[16]. The standalone raw UDP wire contract swaps only BAT/CNT.
RAW_UDP_FROM_API_INDICES = tuple(range(14)) + (15, 14, 16)
API_FROM_RAW_UDP_INDICES = RAW_UDP_FROM_API_INDICES  # the swap is its own inverse


def api17_scan_to_raw_udp_order(values: np.ndarray) -> np.ndarray:
    """Reorder one API17 scan into the standalone raw-UDP field order."""

    scan = np.asarray(values, dtype=np.float32)
    if scan.shape != (RAW_UDP_CHANNEL_COUNT,):
        raise ValueError("API17 scan must contain exactly 17 values")
    return scan[np.asarray(RAW_UDP_FROM_API_INDICES, dtype=int)]


def raw_udp_scan_to_api17_order(values: np.ndarray) -> np.ndarray:
    """Reorder one raw-UDP scan back into direct API17 acquisition order."""

    scan = np.asarray(values, dtype=np.float32)
    if scan.shape != (RAW_UDP_CHANNEL_COUNT,):
        raise ValueError("raw UDP scan must contain exactly 17 values")
    return scan[np.asarray(API_FROM_RAW_UDP_INDICES, dtype=int)]


def encode_unicorn_udp_scan(
    block: UnicornHybridBlackBlock,
    sample_index: int,
    *,
    byte_order: str = "<",
) -> bytes:
    """Encode one API17 block sample as the documented 17-float UDP payload.

    The official documentation specifies 17 floats / 68 bytes and the field
    order, but does not explicitly document byte order. The default little-
    endian representation matches the Windows/x86 environment of Unicorn Suite;
    callers can override it when testing defensive receivers.
    """

    if block.schema != "device17_api":
        raise ValueError("raw Unicorn UDP requires a device17_api block")
    if block.data.shape[0] != RAW_UDP_CHANNEL_COUNT:
        raise ValueError("raw Unicorn UDP requires exactly 17 channels")
    if not 0 <= sample_index < block.data.shape[1]:
        raise IndexError("sample_index out of range")
    if byte_order not in {"<", ">", "=", "!"}:
        raise ValueError("unsupported struct byte order")
    values = api17_scan_to_raw_udp_order(block.data[:, sample_index])
    payload = struct.pack(byte_order + "17f", *[float(value) for value in values])
    if len(payload) != RAW_UDP_PAYLOAD_BYTES:
        raise AssertionError("unexpected Unicorn UDP payload size")
    return payload


def decode_unicorn_udp_scan(payload: bytes, *, byte_order: str = "<") -> np.ndarray:
    """Decode a 68-byte raw UDP payload in *wire order*.

    Use :func:`raw_udp_scan_to_api17_order` if downstream code expects the
    direct API/Recorder auxiliary order.
    """

    if len(payload) != RAW_UDP_PAYLOAD_BYTES:
        raise ValueError(f"expected {RAW_UDP_PAYLOAD_BYTES} bytes, received {len(payload)}")
    return np.asarray(struct.unpack(byte_order + "17f", payload), dtype=np.float32)


def _band_power(window_uv: np.ndarray, sampling_rate_hz: float) -> np.ndarray:
    """Return transparent reference band power as channels × seven bands."""

    data = np.asarray(window_uv, dtype=float)
    if data.ndim != 2 or data.shape[1] < 16:
        raise ValueError("bandpower window must be channels x samples")
    centered = data - np.mean(data, axis=1, keepdims=True)
    taper = np.hanning(data.shape[1])
    spectrum = np.fft.rfft(centered * taper[None, :], axis=1)
    # Window-energy normalization produces an ordinary one-sided power-density
    # estimate adequate for deterministic compatibility testing.
    scale = sampling_rate_hz * max(float(np.sum(taper**2)), 1e-12)
    psd = np.abs(spectrum) ** 2 / scale
    if psd.shape[1] > 2:
        psd[:, 1:-1] *= 2.0
    frequencies = np.fft.rfftfreq(data.shape[1], d=1.0 / sampling_rate_hz)
    output = np.empty((data.shape[0], len(BANDPOWER_BANDS)), dtype=float)
    for band_i, (_, low, high) in enumerate(BANDPOWER_BANDS):
        mask = (frequencies >= low) & (frequencies < high)
        if not np.any(mask):
            output[:, band_i] = np.nan
        else:
            output[:, band_i] = np.trapz(psd[:, mask], frequencies[mask], axis=1)
    return output


def compute_unicorn_bandpower_payload(
    eeg_window_uv: np.ndarray,
    *,
    sampling_rate_hz: float = 250.0,
    enabled_channels: Iterable[bool] | None = None,
) -> np.ndarray:
    """Compute the documented 70-value Bandpower layout.

    Layout:
      0..55  = seven bands × eight channels, band-major
      56..62 = seven bands averaged over enabled channels
      63..69 = seven bands averaged over all enabled bipolar derivations

    Disabled channel values are NaN, matching Unicorn Bandpower's documented UDP
    behavior. Averages ignore disabled channels. If fewer than two channels are
    enabled, bipolar features are NaN.
    """

    eeg = np.asarray(eeg_window_uv, dtype=float)
    if eeg.ndim != 2 or eeg.shape[0] != 8:
        raise ValueError("Unicorn Bandpower requires 8 EEG channels")
    if sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive")
    enabled = np.ones(8, dtype=bool) if enabled_channels is None else np.asarray(tuple(enabled_channels), dtype=bool)
    if enabled.shape != (8,):
        raise ValueError("enabled_channels must contain exactly 8 booleans")

    per_channel = _band_power(eeg, sampling_rate_hz)  # 8 × 7
    visible = per_channel.copy()
    visible[~enabled, :] = np.nan
    band_major = visible.T.reshape(-1)

    channel_average = np.full(7, np.nan, dtype=float)
    if np.any(enabled):
        channel_average = np.mean(per_channel[enabled], axis=0)

    indices = np.flatnonzero(enabled)
    bipolar_average = np.full(7, np.nan, dtype=float)
    if indices.size >= 2:
        bipolar = []
        for first_i, first in enumerate(indices[:-1]):
            for second in indices[first_i + 1 :]:
                bipolar.append(eeg[first] - eeg[second])
        bipolar_power = _band_power(np.asarray(bipolar, dtype=float), sampling_rate_hz)
        bipolar_average = np.mean(bipolar_power, axis=0)

    payload = np.concatenate([band_major, channel_average, bipolar_average]).astype(float)
    if payload.shape != (BANDPOWER_FEATURE_COUNT,):
        raise AssertionError("Bandpower payload layout must contain 70 values")
    return payload


def encode_unicorn_bandpower_ascii(values: np.ndarray) -> bytes:
    """Encode a 70-value Bandpower payload as comma-separated ASCII."""

    array = np.asarray(values, dtype=float)
    if array.shape != (BANDPOWER_FEATURE_COUNT,):
        raise ValueError("Bandpower payload must contain exactly 70 values")
    text = ",".join("NaN" if np.isnan(value) else format(float(value), ".9g") for value in array)
    return text.encode("ascii")


def decode_unicorn_bandpower_ascii(payload: bytes | str) -> np.ndarray:
    text = payload.decode("ascii") if isinstance(payload, bytes) else str(payload)
    parts = text.strip().split(",")
    if len(parts) != BANDPOWER_FEATURE_COUNT:
        raise ValueError("Bandpower ASCII payload must contain exactly 70 comma-separated values")
    values = [np.nan if part.strip().lower() == "nan" else float(part) for part in parts]
    return np.asarray(values, dtype=float)


@dataclass(frozen=True)
class BandpowerFrame:
    sample_index: int
    timestamp_s: float
    values: np.ndarray


class UnicornBandpowerReferenceStream:
    """Stateful documented-cadence Bandpower reference estimator.

    Defaults match the currently documented Unicorn Bandpower settings:
    250-sample buffer, 240-sample overlap, therefore 10-sample hop / 25 Hz
    feature update rate at 250 Hz EEG.
    """

    def __init__(
        self,
        *,
        sampling_rate_hz: float = 250.0,
        buffer_size: int = 250,
        buffer_overlap: int = 240,
        enabled_channels: Iterable[bool] | None = None,
    ) -> None:
        if sampling_rate_hz <= 0:
            raise ValueError("sampling_rate_hz must be positive")
        if buffer_size <= 0 or not 0 <= buffer_overlap < buffer_size:
            raise ValueError("buffer settings require 0 <= overlap < buffer_size")
        self.sampling_rate_hz = float(sampling_rate_hz)
        self.buffer_size = int(buffer_size)
        self.buffer_overlap = int(buffer_overlap)
        self.hop_samples = self.buffer_size - self.buffer_overlap
        self.update_rate_hz = self.sampling_rate_hz / self.hop_samples
        self.enabled_channels = np.ones(8, dtype=bool) if enabled_channels is None else np.asarray(tuple(enabled_channels), dtype=bool)
        if self.enabled_channels.shape != (8,):
            raise ValueError("enabled_channels must contain exactly 8 booleans")
        self._data = np.empty((8, 0), dtype=float)
        self._sample_count = 0
        self._next_emit = self.buffer_size

    def push(self, eeg_uv: np.ndarray) -> tuple[BandpowerFrame, ...]:
        block = np.asarray(eeg_uv, dtype=float)
        if block.ndim != 2 or block.shape[0] != 8:
            raise ValueError("eeg_uv must be 8 x samples")
        if block.shape[1] == 0:
            return ()
        self._data = np.concatenate([self._data, block], axis=1)
        self._sample_count += block.shape[1]
        frames: list[BandpowerFrame] = []
        while self._sample_count >= self._next_emit:
            global_stop = self._next_emit
            global_start = global_stop - self.buffer_size
            buffer_origin = self._sample_count - self._data.shape[1]
            local_start = global_start - buffer_origin
            local_stop = global_stop - buffer_origin
            if local_start < 0 or local_stop > self._data.shape[1]:
                break
            window = self._data[:, local_start:local_stop]
            values = compute_unicorn_bandpower_payload(
                window,
                sampling_rate_hz=self.sampling_rate_hz,
                enabled_channels=self.enabled_channels,
            )
            frames.append(
                BandpowerFrame(
                    sample_index=global_stop - 1,
                    timestamp_s=(global_stop - 1) / self.sampling_rate_hz,
                    values=values,
                )
            )
            self._next_emit += self.hop_samples
        # Retain only enough history for the next overlapping window plus a
        # small hop margin. This keeps long game simulations bounded.
        keep = self.buffer_size + self.hop_samples
        if self._data.shape[1] > keep:
            self._data = self._data[:, -keep:]
        return tuple(frames)
