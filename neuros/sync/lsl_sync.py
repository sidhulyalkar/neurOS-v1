"""
LSL stream synchronisation and drift estimation.

This module contains utilities to access LabStreamingLayer (LSL) streams
and estimate clock drift and jitter relative to the local host clock.
LSL provides "time_correction" which returns the offset between the
remote clock (sensor) and the local machine; this module wraps that
functionality and accumulates statistics over a configurable duration.
"""
from __future__ import annotations

import logging
import math
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple

try:
    from pylsl import resolve_stream, StreamInlet
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pylsl is required for LSL synchronisation.  Install with `pip install pylsl`."
    ) from exc

logger = logging.getLogger(__name__)


@dataclass
class DriftStats:
    """Summary statistics for clock drift and jitter."""

    mean_offset: float
    std_offset: float
    max_offset: float
    min_offset: float

    @classmethod
    def from_offsets(cls, offsets: List[float]) -> "DriftStats":
        return cls(
            mean_offset=float(statistics.mean(offsets)),
            std_offset=float(statistics.stdev(offsets)) if len(offsets) > 1 else 0.0,
            max_offset=float(max(offsets)),
            min_offset=float(min(offsets)),
        )


def measure_time_correction(inlet: StreamInlet, duration: float = 10.0) -> List[float]:
    """Collect a series of time correction measurements.

    Parameters
    ----------
    inlet:
        An LSL ``StreamInlet`` instance connected to the desired stream.
    duration:
        Time in seconds over which to collect measurements.

    Returns
    -------
    List[float]
        Offsets in seconds from the remote clock to the local clock.
    """
    offsets: List[float] = []
    start = time.perf_counter()
    while time.perf_counter() - start < duration:
        offset = inlet.time_correction()
        offsets.append(offset)
        # Sleep a bit to avoid hammering the device; LSL docs suggest ~0.1 s
        time.sleep(0.1)
    return offsets


def run_sync_qa(stream_name: str, duration: float = 10.0) -> DriftStats:
    """Resolve an LSL stream and print drift/jitter statistics.

    This function looks up the specified LSL stream by name, connects an
    inlet and collects a series of time_correction measurements over
    ``duration`` seconds.  At the end it logs summary statistics
    (mean, standard deviation, min and max) and returns them.

    Parameters
    ----------
    stream_name:
        The name of the LSL stream to resolve.  Use the same name as
        defined by the device or shim publishing to LSL.
    duration:
        Time in seconds over which to collect drift samples.

    Returns
    -------
    DriftStats
        Summary statistics for the clock offset.
    """
    # Resolve the stream (wait up to 5 seconds)
    logger.info("Resolving LSL stream '%s'", stream_name)
    streams = resolve_stream("name", stream_name, timeout=5)
    if not streams:
        raise RuntimeError(f"No LSL stream named '{stream_name}' found")
    inlet = StreamInlet(streams[0])

    logger.info("Connected to LSL stream; collecting drift data for %.1f seconds", duration)
    offsets = measure_time_correction(inlet, duration=duration)

    stats = DriftStats.from_offsets(offsets)
    logger.info(
        "Drift statistics for stream '%s': mean=%.6f s, std=%.6f s, min=%.6f s, max=%.6f s",
        stream_name,
        stats.mean_offset,
        stats.std_offset,
        stats.min_offset,
        stats.max_offset,
    )
    return stats


__all__ = [
    "DriftStats",
    "measure_time_correction",
    "run_sync_qa",
]