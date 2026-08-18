"""Clock alignment utilities for multimodal neural streams."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from neuros.contracts import ClockDomain, QualityFlag, SignalFrame
from neuros.errors import ClockSyncError


@dataclass(frozen=True, slots=True)
class ClockEstimate:
    """Affine mapping from device time to host monotonic time."""

    scale: float
    offset_ns: float
    drift_ppm: float
    uncertainty_ns: float
    sample_count: int

    def map_time(self, device_time_ns: int) -> int:
        return int(round(self.scale * device_time_ns + self.offset_ns))


class ClockSynchronizer:
    """Estimate device-to-host clock mapping from timestamp pairs.

    The model is ``host_time = scale * device_time + offset``. A bounded
    least-squares window allows slow drift to be tracked without unbounded state.
    This is a timing primitive, not a replacement for hardware synchronization.
    """

    def __init__(
        self,
        *,
        window_size: int = 64,
        min_samples: int = 4,
        uncertainty_threshold_ns: float = 2_000_000.0,
    ) -> None:
        if window_size < 2:
            raise ValueError("window_size must be >= 2")
        if min_samples < 2 or min_samples > window_size:
            raise ValueError("min_samples must be between 2 and window_size")
        if uncertainty_threshold_ns <= 0:
            raise ValueError("uncertainty_threshold_ns must be positive")
        self.window_size = window_size
        self.min_samples = min_samples
        self.uncertainty_threshold_ns = uncertainty_threshold_ns
        self._pairs: deque[tuple[int, int]] = deque(maxlen=window_size)
        self._estimate: ClockEstimate | None = None

    @property
    def estimate(self) -> ClockEstimate | None:
        return self._estimate

    def update(self, *, device_time_ns: int, host_time_ns: int) -> ClockEstimate | None:
        if self._pairs:
            last_device, last_host = self._pairs[-1]
            if device_time_ns <= last_device:
                raise ClockSyncError("device timestamps must increase monotonically")
            if host_time_ns <= last_host:
                raise ClockSyncError("host timestamps must increase monotonically")
        self._pairs.append((device_time_ns, host_time_ns))
        if len(self._pairs) < self.min_samples:
            return None

        device = np.asarray([item[0] for item in self._pairs], dtype=np.float64)
        host = np.asarray([item[1] for item in self._pairs], dtype=np.float64)
        origin_device = device[0]
        origin_host = host[0]
        x = device - origin_device
        y = host - origin_host
        design = np.column_stack([x, np.ones_like(x)])
        scale, local_offset = np.linalg.lstsq(design, y, rcond=None)[0]
        offset = origin_host + local_offset - scale * origin_device
        residuals = host - (scale * device + offset)
        uncertainty = float(np.sqrt(np.mean(residuals**2)))
        self._estimate = ClockEstimate(
            scale=float(scale),
            offset_ns=float(offset),
            drift_ppm=float((scale - 1.0) * 1_000_000.0),
            uncertainty_ns=uncertainty,
            sample_count=len(self._pairs),
        )
        return self._estimate

    def synchronize(self, frame: SignalFrame) -> SignalFrame:
        """Return a frame with synchronized time and quality annotation."""
        if frame.device_time_ns is None:
            raise ClockSyncError("frame has no device_time_ns")
        if self._estimate is None:
            raise ClockSyncError("clock estimate is not ready")
        quality = frame.quality
        if self._estimate.uncertainty_ns > self.uncertainty_threshold_ns:
            quality |= QualityFlag.CLOCK_UNCERTAIN
        return SignalFrame(
            stream_id=frame.stream_id,
            sequence_id=frame.sequence_id,
            data=frame.data,
            sample_rate_hz=frame.sample_rate_hz,
            host_receive_time_ns=frame.host_receive_time_ns,
            device_time_ns=frame.device_time_ns,
            synchronized_time_ns=self._estimate.map_time(frame.device_time_ns),
            clock_domain=ClockDomain.SYNCHRONIZED,
            quality=quality,
            metadata={
                **dict(frame.metadata),
                "clock_drift_ppm": self._estimate.drift_ppm,
                "clock_uncertainty_ns": self._estimate.uncertainty_ns,
            },
        )
