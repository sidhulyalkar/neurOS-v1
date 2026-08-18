import numpy as np
import pytest

from neuros.contracts import ClockDomain, SignalFrame
from neuros.errors import ClockSyncError
from neuros.synchronization import ClockSynchronizer


def test_clock_synchronizer_recovers_affine_mapping():
    sync = ClockSynchronizer(window_size=8, min_samples=4)
    estimate = None
    for i in range(8):
        device = 1_000_000_000 + i * 10_000_000
        host = int(1.0001 * device + 5_000_000)
        estimate = sync.update(device_time_ns=device, host_time_ns=host)
    assert estimate is not None
    assert estimate.drift_ppm == pytest.approx(100.0, abs=0.5)
    assert estimate.uncertainty_ns < 10.0


def test_clock_synchronizer_marks_frame_synchronized():
    sync = ClockSynchronizer(min_samples=2, window_size=4)
    sync.update(device_time_ns=100, host_time_ns=1100)
    sync.update(device_time_ns=200, host_time_ns=1200)
    frame = SignalFrame(
        stream_id="eeg",
        sequence_id=0,
        data=np.zeros((2, 4)),
        sample_rate_hz=250.0,
        host_receive_time_ns=1200,
        device_time_ns=200,
    )
    aligned = sync.synchronize(frame)
    assert aligned.clock_domain is ClockDomain.SYNCHRONIZED
    assert aligned.synchronized_time_ns == pytest.approx(1200, abs=1)


def test_clock_synchronizer_rejects_non_monotonic_device_time():
    sync = ClockSynchronizer(min_samples=2)
    sync.update(device_time_ns=100, host_time_ns=1000)
    with pytest.raises(ClockSyncError):
        sync.update(device_time_ns=100, host_time_ns=1100)
