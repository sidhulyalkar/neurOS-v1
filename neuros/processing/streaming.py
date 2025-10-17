"""
Real-time streaming and buffering optimizations for low-latency BCI.

This module provides optimized data structures and algorithms for
real-time signal processing with minimal latency and memory overhead.
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Callable, List
import numpy as np
from threading import Lock
import time


class RingBuffer:
    """
    Efficient ring buffer for streaming data with O(1) operations.

    Optimized for real-time BCI applications where data arrives
    continuously and old samples need to be discarded.

    Parameters
    ----------
    capacity : int
        Maximum number of samples to store
    n_channels : int
        Number of data channels
    dtype : numpy dtype
        Data type for storage (default: float32 for performance)
    """

    def __init__(self, capacity: int, n_channels: int, dtype=np.float32):
        self.capacity = capacity
        self.n_channels = n_channels
        self.dtype = dtype

        # Preallocate buffer for zero-copy operations
        self._buffer = np.zeros((capacity, n_channels), dtype=dtype)
        self._write_pos = 0
        self._count = 0
        self._lock = Lock()

    def append(self, data: np.ndarray) -> None:
        """
        Append new data to the buffer.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_channels) or (n_channels,)
        """
        with self._lock:
            if data.ndim == 1:
                data = data[np.newaxis, :]

            n_samples = data.shape[0]

            for i in range(n_samples):
                self._buffer[self._write_pos] = data[i]
                self._write_pos = (self._write_pos + 1) % self.capacity
                self._count = min(self._count + 1, self.capacity)

    def get_last(self, n_samples: int) -> np.ndarray:
        """
        Get the last n samples (most recent).

        Parameters
        ----------
        n_samples : int
            Number of samples to retrieve

        Returns
        -------
        data : np.ndarray
            Shape (n_samples, n_channels)
        """
        with self._lock:
            if n_samples > self._count:
                n_samples = self._count

            if n_samples == 0:
                return np.array([])

            # Calculate indices
            start_idx = (self._write_pos - n_samples) % self.capacity

            if start_idx + n_samples <= self.capacity:
                # Contiguous block
                return self._buffer[start_idx:start_idx + n_samples].copy()
            else:
                # Wrap-around case
                first_part = self._buffer[start_idx:]
                second_part = self._buffer[:self._write_pos]
                return np.vstack([first_part, second_part])

    def get_all(self) -> np.ndarray:
        """Get all data in the buffer."""
        return self.get_last(self._count)

    def clear(self) -> None:
        """Clear the buffer."""
        with self._lock:
            self._write_pos = 0
            self._count = 0

    @property
    def size(self) -> int:
        """Current number of samples in buffer."""
        return self._count

    @property
    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self._count >= self.capacity


class StreamingFilter:
    """
    Streaming IIR filter with minimal latency.

    Uses state-space form for efficient online filtering
    without storing entire signal history.

    Parameters
    ----------
    b : array-like
        Numerator coefficients
    a : array-like
        Denominator coefficients
    n_channels : int
        Number of channels to filter
    """

    def __init__(self, b: np.ndarray, a: np.ndarray, n_channels: int):
        self.b = np.asarray(b, dtype=np.float32)
        self.a = np.asarray(a, dtype=np.float32)
        self.n_channels = n_channels

        # Initialize filter state
        self.zi = np.zeros((max(len(b), len(a)) - 1, n_channels), dtype=np.float32)

    def process(self, data: np.ndarray) -> np.ndarray:
        """
        Process new data samples through the filter.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_samples, n_channels) or (n_channels,)

        Returns
        -------
        filtered : np.ndarray
            Filtered data, same shape as input
        """
        from scipy.signal import lfilter

        if data.ndim == 1:
            data = data[np.newaxis, :]

        filtered = np.zeros_like(data)

        for ch in range(self.n_channels):
            filtered[:, ch], self.zi[:, ch] = lfilter(
                self.b, self.a, data[:, ch], zi=self.zi[:, ch]
            )

        return filtered.squeeze()

    def reset(self) -> None:
        """Reset filter state."""
        self.zi[:] = 0


class StreamingFeatureExtractor:
    """
    Efficient streaming feature extraction with windowing.

    Computes features incrementally as new data arrives,
    minimizing computational overhead.

    Parameters
    ----------
    window_size : int
        Size of the analysis window in samples
    hop_size : int
        Number of samples to advance between windows
    feature_func : callable
        Function that extracts features from a window
    """

    def __init__(self, window_size: int, hop_size: int,
                 feature_func: Callable[[np.ndarray], np.ndarray]):
        self.window_size = window_size
        self.hop_size = hop_size
        self.feature_func = feature_func
        self._buffer = None
        self._samples_since_last = 0

    def process(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Process new data and extract features when window is ready.

        Parameters
        ----------
        data : np.ndarray
            New data samples

        Returns
        -------
        features : np.ndarray or None
            Features if window is complete, None otherwise
        """
        if self._buffer is None:
            # Initialize buffer
            n_channels = data.shape[-1] if data.ndim > 1 else 1
            self._buffer = RingBuffer(self.window_size, n_channels)

        # Add new data to buffer
        self._buffer.append(data)
        self._samples_since_last += data.shape[0] if data.ndim > 1 else 1

        # Check if we should extract features
        if self._samples_since_last >= self.hop_size and self._buffer.is_full:
            window_data = self._buffer.get_all()
            features = self.feature_func(window_data)
            self._samples_since_last = 0
            return features

        return None


class AdaptiveBuffer:
    """
    Adaptive buffering with automatic size adjustment.

    Dynamically adjusts buffer size based on data arrival rate
    and processing latency to maintain optimal performance.

    Parameters
    ----------
    initial_size : int
        Initial buffer size
    min_size : int
        Minimum buffer size
    max_size : int
        Maximum buffer size
    """

    def __init__(self, initial_size: int = 250, min_size: int = 100,
                 max_size: int = 1000):
        self.min_size = min_size
        self.max_size = max_size
        self.current_size = initial_size

        self._buffer = deque(maxlen=max_size)
        self._arrival_times = deque(maxlen=100)
        self._processing_times = deque(maxlen=100)

    def append(self, data: np.ndarray) -> None:
        """Add data to buffer and record arrival time."""
        self._buffer.append(data)
        self._arrival_times.append(time.perf_counter())

    def get_batch(self) -> Optional[List[np.ndarray]]:
        """Get a batch of samples for processing."""
        if len(self._buffer) < self.current_size:
            return None

        batch = [self._buffer.popleft() for _ in range(self.current_size)]
        return batch

    def record_processing_time(self, duration: float) -> None:
        """Record processing time and adjust buffer size if needed."""
        self._processing_times.append(duration)

        # Adjust buffer size based on recent statistics
        if len(self._processing_times) >= 10:
            avg_processing = np.mean(list(self._processing_times))
            avg_arrival_rate = self._estimate_arrival_rate()

            if avg_arrival_rate > 0:
                # Calculate optimal buffer size
                optimal_size = int(avg_processing * avg_arrival_rate * 1.5)
                optimal_size = np.clip(optimal_size, self.min_size, self.max_size)

                # Gradually adjust current size
                if optimal_size > self.current_size:
                    self.current_size = min(self.current_size + 10, optimal_size)
                elif optimal_size < self.current_size:
                    self.current_size = max(self.current_size - 10, optimal_size)

    def _estimate_arrival_rate(self) -> float:
        """Estimate data arrival rate (samples/second)."""
        if len(self._arrival_times) < 2:
            return 0.0

        times = list(self._arrival_times)
        duration = times[-1] - times[0]
        if duration > 0:
            return len(times) / duration
        return 0.0


class LowLatencyPipeline:
    """
    Optimized pipeline for low-latency real-time processing.

    Combines ring buffering, streaming filters, and incremental
    feature extraction for minimal end-to-end latency.

    Parameters
    ----------
    buffer_size : int
        Size of the ring buffer
    n_channels : int
        Number of EEG channels
    filters : list
        List of streaming filters to apply
    feature_extractor : StreamingFeatureExtractor
        Feature extractor for online processing
    """

    def __init__(self, buffer_size: int, n_channels: int,
                 filters: Optional[List[StreamingFilter]] = None,
                 feature_extractor: Optional[StreamingFeatureExtractor] = None):
        self.buffer = RingBuffer(buffer_size, n_channels)
        self.filters = filters or []
        self.feature_extractor = feature_extractor

        # Performance monitoring
        self._latencies = deque(maxlen=100)
        self._throughputs = deque(maxlen=100)

    def process_sample(self, data: np.ndarray) -> Optional[np.ndarray]:
        """
        Process a single sample with minimal latency.

        Parameters
        ----------
        data : np.ndarray
            Shape (n_channels,) or (1, n_channels)

        Returns
        -------
        features : np.ndarray or None
            Extracted features if available
        """
        start_time = time.perf_counter()

        # Add to buffer
        self.buffer.append(data)

        # Apply streaming filters
        filtered_data = data
        for filt in self.filters:
            filtered_data = filt.process(filtered_data)

        # Extract features if extractor is configured
        features = None
        if self.feature_extractor is not None:
            features = self.feature_extractor.process(filtered_data)

        # Record latency
        latency = (time.perf_counter() - start_time) * 1000  # ms
        self._latencies.append(latency)

        return features

    def get_performance_stats(self) -> dict:
        """Get pipeline performance statistics."""
        if not self._latencies:
            return {}

        return {
            'mean_latency_ms': float(np.mean(self._latencies)),
            'p95_latency_ms': float(np.percentile(self._latencies, 95)),
            'p99_latency_ms': float(np.percentile(self._latencies, 99)),
            'max_latency_ms': float(np.max(self._latencies)),
        }


class CircularBufferOptimized:
    """
    Highly optimized circular buffer using memoryviews for zero-copy operations.

    Designed for maximum throughput in real-time BCI applications.

    Parameters
    ----------
    capacity : int
        Buffer capacity
    n_channels : int
        Number of channels
    dtype : numpy dtype
        Data type
    """

    def __init__(self, capacity: int, n_channels: int, dtype=np.float32):
        self.capacity = capacity
        self.n_channels = n_channels

        # Use contiguous C-order array for cache efficiency
        self._data = np.empty((capacity, n_channels), dtype=dtype, order='C')
        self._head = 0
        self._tail = 0
        self._size = 0
        self._lock = Lock()

    def push(self, samples: np.ndarray) -> None:
        """Push samples with zero-copy when possible."""
        with self._lock:
            n_samples = samples.shape[0] if samples.ndim > 1 else 1

            if n_samples > self.capacity:
                # Take only the last 'capacity' samples
                samples = samples[-self.capacity:]
                n_samples = self.capacity

            # Calculate how many samples we can write before wrapping
            space_to_end = self.capacity - self._head

            if n_samples <= space_to_end:
                # No wrap needed - single copy
                self._data[self._head:self._head + n_samples] = samples
                self._head = (self._head + n_samples) % self.capacity
            else:
                # Need to wrap - two copies
                first_chunk = space_to_end
                self._data[self._head:] = samples[:first_chunk]
                self._data[:n_samples - first_chunk] = samples[first_chunk:]
                self._head = n_samples - first_chunk

            # Update size
            self._size = min(self._size + n_samples, self.capacity)
            if self._size == self.capacity:
                self._tail = self._head

    def get_view(self, n_samples: int) -> np.ndarray:
        """Get a view of the last n samples (no copy)."""
        with self._lock:
            if n_samples > self._size:
                n_samples = self._size

            start = (self._head - n_samples) % self.capacity

            if start + n_samples <= self.capacity:
                return self._data[start:start + n_samples]
            else:
                # Need to handle wrap - must copy in this case
                return np.vstack([
                    self._data[start:],
                    self._data[:self._head]
                ])
