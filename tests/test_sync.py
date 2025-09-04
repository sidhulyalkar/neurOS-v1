"""
Unit tests for the synchronisation module.

These tests exercise the ``DriftStats.from_offsets`` helper to ensure
statistics are computed correctly.  We avoid connecting to an LSL
stream by not invoking ``run_sync_qa`` here, since that would
require a live device.
"""
from __future__ import annotations

import neuros.sync.lsl_sync as lsl


def test_drift_stats_from_offsets():
    offsets = [0.1, 0.2, 0.15, 0.05]
    stats = lsl.DriftStats.from_offsets(offsets)
    assert abs(stats.mean_offset - 0.125) < 1e-6
    assert stats.max_offset == 0.2
    assert stats.min_offset == 0.05
    # Standard deviation of [0.1, 0.2, 0.15, 0.05] is ~0.0559
    assert abs(stats.std_offset - 0.05590169943749474) < 1e-6