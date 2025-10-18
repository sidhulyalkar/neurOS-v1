"""
Real-time inference pipeline for NeuroFM-X.

Provides optimized inference for low-latency neural decoding:
- Dynamic batching for variable latency
- Model caching and warm-up
- Latency profiling and monitoring
- Multi-threaded/async inference
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from pathlib import Path
from collections import deque
import time
import threading
import queue
from dataclasses import dataclass
import json

import numpy as np
import torch
import torch.nn as nn


@dataclass
class InferenceRequest:
    """Single inference request.

    Attributes
    ----------
    request_id : str
        Unique identifier for this request.
    data : torch.Tensor
        Input data.
    timestamp : float
        Request creation timestamp.
    callback : callable, optional
        Callback to invoke with results.
    """
    request_id: str
    data: torch.Tensor
    timestamp: float
    callback: Optional[Callable] = None


@dataclass
class InferenceResult:
    """Inference result.

    Attributes
    ----------
    request_id : str
        Request identifier.
    predictions : torch.Tensor or dict
        Model predictions.
    latency_ms : float
        End-to-end latency in milliseconds.
    batch_size : int
        Batch size used for inference.
    """
    request_id: str
    predictions: Union[torch.Tensor, Dict[str, torch.Tensor]]
    latency_ms: float
    batch_size: int


class DynamicBatcher:
    """Dynamic batching for variable-latency inference.

    Collects requests into batches based on time or size constraints.

    Parameters
    ----------
    max_batch_size : int, optional
        Maximum batch size.
        Default: 32.
    max_wait_ms : float, optional
        Maximum wait time in milliseconds before processing batch.
        Default: 10.0.
    padding_value : float, optional
        Value to use for padding variable-length sequences.
        Default: 0.0.

    Examples
    --------
    >>> batcher = DynamicBatcher(max_batch_size=16, max_wait_ms=5.0)
    >>> batcher.add_request(request)
    >>> batch = batcher.get_batch()  # Returns when batch is ready
    """

    def __init__(
        self,
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
        padding_value: float = 0.0,
    ):
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms
        self.padding_value = padding_value

        self.queue = deque()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)

    def add_request(self, request: InferenceRequest):
        """Add inference request to queue.

        Parameters
        ----------
        request : InferenceRequest
            Request to add.
        """
        with self.lock:
            self.queue.append(request)
            self.condition.notify()

    def get_batch(
        self,
        timeout: Optional[float] = None,
    ) -> Optional[Tuple[List[InferenceRequest], torch.Tensor]]:
        """Get next batch of requests.

        Blocks until batch is ready or timeout.

        Parameters
        ----------
        timeout : float, optional
            Timeout in seconds. None means wait indefinitely.

        Returns
        -------
        tuple or None
            (requests, batched_data) or None if timeout.
        """
        with self.condition:
            start_time = time.time()

            while True:
                # Check if we have enough requests
                if len(self.queue) >= self.max_batch_size:
                    return self._create_batch(self.max_batch_size)

                # Check if oldest request has waited too long
                if len(self.queue) > 0:
                    oldest_age = (time.time() - self.queue[0].timestamp) * 1000
                    if oldest_age >= self.max_wait_ms:
                        return self._create_batch(min(len(self.queue), self.max_batch_size))

                # Wait for new requests
                remaining_timeout = None
                if timeout is not None:
                    elapsed = time.time() - start_time
                    remaining_timeout = max(0, timeout - elapsed)
                    if remaining_timeout <= 0:
                        # Timeout: return whatever we have
                        if len(self.queue) > 0:
                            return self._create_batch(min(len(self.queue), self.max_batch_size))
                        return None

                # Wait for condition
                wait_time = self.max_wait_ms / 1000.0
                if remaining_timeout is not None:
                    wait_time = min(wait_time, remaining_timeout)

                self.condition.wait(timeout=wait_time)

    def _create_batch(self, batch_size: int) -> Tuple[List[InferenceRequest], torch.Tensor]:
        """Create batch from queue.

        Parameters
        ----------
        batch_size : int
            Number of requests to batch.

        Returns
        -------
        tuple
            (requests, batched_data).
        """
        requests = []
        data_list = []

        for _ in range(batch_size):
            if len(self.queue) == 0:
                break
            req = self.queue.popleft()
            requests.append(req)
            data_list.append(req.data)

        # Pad and stack data
        if len(data_list) > 0:
            # Find max dimensions
            max_dims = [max(d.size(i) for d in data_list) for i in range(data_list[0].ndim)]

            # Pad each tensor
            padded_data = []
            for data in data_list:
                pad_sizes = []
                for i in range(data.ndim - 1, -1, -1):
                    pad_before = 0
                    pad_after = max_dims[i] - data.size(i)
                    pad_sizes.extend([pad_before, pad_after])

                if sum(pad_sizes) > 0:
                    padded = torch.nn.functional.pad(
                        data,
                        pad_sizes,
                        value=self.padding_value,
                    )
                else:
                    padded = data

                padded_data.append(padded)

            # Stack into batch
            batched_data = torch.stack(padded_data, dim=0)
        else:
            batched_data = torch.empty(0)

        return requests, batched_data


class ModelCache:
    """Cache for model instances and precomputed tensors.

    Manages model lifecycle and warm-up.

    Parameters
    ----------
    model : nn.Module
        Model to cache.
    device : str, optional
        Device to run model on.
        Default: 'cpu'.
    warmup_steps : int, optional
        Number of warm-up forward passes.
        Default: 10.

    Examples
    --------
    >>> cache = ModelCache(model, device='cuda', warmup_steps=10)
    >>> cache.warmup(example_input)
    >>> output = cache.forward(input_data)
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        warmup_steps: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.warmup_steps = warmup_steps

        # Set to eval mode
        self.model.eval()

        # Disable gradient computation
        for param in self.model.parameters():
            param.requires_grad = False

    def warmup(self, example_input: torch.Tensor):
        """Warm up model with example inputs.

        Runs several forward passes to initialize CUDA kernels.

        Parameters
        ----------
        example_input : torch.Tensor
            Example input for warm-up.
        """
        example_input = example_input.to(self.device)

        with torch.no_grad():
            for _ in range(self.warmup_steps):
                _ = self.model(example_input)

        # Synchronize CUDA
        if self.device.startswith('cuda'):
            torch.cuda.synchronize()

    @torch.no_grad()
    def forward(self, input_data: torch.Tensor) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run forward pass.

        Parameters
        ----------
        input_data : torch.Tensor
            Input data.

        Returns
        -------
        torch.Tensor or dict
            Model predictions.
        """
        input_data = input_data.to(self.device)
        output = self.model(input_data)
        return output


class LatencyProfiler:
    """Profile and monitor inference latency.

    Tracks latency statistics over time.

    Parameters
    ----------
    window_size : int, optional
        Number of recent samples to track.
        Default: 1000.

    Examples
    --------
    >>> profiler = LatencyProfiler()
    >>> with profiler.profile('inference'):
    ...     output = model(input)
    >>> stats = profiler.get_stats()
    """

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.operation_latencies = {}
        self.lock = threading.Lock()

    def record(self, latency_ms: float, operation: str = 'inference'):
        """Record latency measurement.

        Parameters
        ----------
        latency_ms : float
            Latency in milliseconds.
        operation : str, optional
            Operation name.
        """
        with self.lock:
            self.latencies.append(latency_ms)

            if operation not in self.operation_latencies:
                self.operation_latencies[operation] = deque(maxlen=self.window_size)

            self.operation_latencies[operation].append(latency_ms)

    def profile(self, operation: str = 'inference'):
        """Context manager for profiling.

        Parameters
        ----------
        operation : str, optional
            Operation name.

        Returns
        -------
        context manager
            Records latency when exiting context.
        """
        return _ProfileContext(self, operation)

    def get_stats(self, operation: Optional[str] = None) -> Dict[str, float]:
        """Get latency statistics.

        Parameters
        ----------
        operation : str, optional
            Specific operation to get stats for. None for overall.

        Returns
        -------
        dict
            Latency statistics (mean, median, p95, p99, min, max).
        """
        with self.lock:
            if operation is not None:
                latencies = list(self.operation_latencies.get(operation, []))
            else:
                latencies = list(self.latencies)

            if len(latencies) == 0:
                return {
                    'mean': 0.0,
                    'median': 0.0,
                    'p95': 0.0,
                    'p99': 0.0,
                    'min': 0.0,
                    'max': 0.0,
                    'count': 0,
                }

            latencies_np = np.array(latencies)

            return {
                'mean': float(np.mean(latencies_np)),
                'median': float(np.median(latencies_np)),
                'p95': float(np.percentile(latencies_np, 95)),
                'p99': float(np.percentile(latencies_np, 99)),
                'min': float(np.min(latencies_np)),
                'max': float(np.max(latencies_np)),
                'count': len(latencies),
            }

    def reset(self):
        """Reset all statistics."""
        with self.lock:
            self.latencies.clear()
            self.operation_latencies.clear()


class _ProfileContext:
    """Context manager for latency profiling."""

    def __init__(self, profiler: LatencyProfiler, operation: str):
        self.profiler = profiler
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = time.perf_counter()
        return self

    def __exit__(self, *args):
        latency_ms = (time.perf_counter() - self.start_time) * 1000
        self.profiler.record(latency_ms, self.operation)


class RealtimeInferencePipeline:
    """Real-time inference pipeline with batching and monitoring.

    Complete pipeline for low-latency neural decoding.

    Parameters
    ----------
    model : nn.Module
        Model for inference.
    device : str, optional
        Device to run on.
        Default: 'cpu'.
    max_batch_size : int, optional
        Maximum batch size.
        Default: 32.
    max_wait_ms : float, optional
        Maximum batching wait time (ms).
        Default: 10.0.
    warmup_steps : int, optional
        Number of warm-up steps.
        Default: 10.

    Examples
    --------
    >>> pipeline = RealtimeInferencePipeline(model, device='cuda')
    >>> pipeline.start()
    >>> result = pipeline.predict(data, request_id='req-001')
    >>> pipeline.stop()
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        max_batch_size: int = 32,
        max_wait_ms: float = 10.0,
        warmup_steps: int = 10,
    ):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.max_wait_ms = max_wait_ms

        # Initialize components
        self.batcher = DynamicBatcher(max_batch_size, max_wait_ms)
        self.cache = ModelCache(model, device, warmup_steps)
        self.profiler = LatencyProfiler()

        # Threading
        self.worker_thread = None
        self.running = False
        self.result_queue = queue.Queue()

    def start(self, example_input: Optional[torch.Tensor] = None):
        """Start inference pipeline.

        Parameters
        ----------
        example_input : torch.Tensor, optional
            Example input for warm-up.
        """
        if self.running:
            return

        # Warm up model
        if example_input is not None:
            print("Warming up model...")
            self.cache.warmup(example_input)

        # Start worker thread
        self.running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()

        print(f"Inference pipeline started (device={self.device}, batch_size={self.max_batch_size})")

    def stop(self):
        """Stop inference pipeline."""
        if not self.running:
            return

        self.running = False
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=5.0)

        print("Inference pipeline stopped")

    def predict(
        self,
        data: torch.Tensor,
        request_id: Optional[str] = None,
        timeout: float = 1.0,
    ) -> Optional[InferenceResult]:
        """Submit prediction request and wait for result.

        Parameters
        ----------
        data : torch.Tensor
            Input data.
        request_id : str, optional
            Request ID. Auto-generated if None.
        timeout : float, optional
            Timeout in seconds.

        Returns
        -------
        InferenceResult or None
            Prediction result or None if timeout.
        """
        if request_id is None:
            request_id = f"req-{time.time()}"

        # Create request
        request = InferenceRequest(
            request_id=request_id,
            data=data,
            timestamp=time.time(),
        )

        # Submit to batcher
        self.batcher.add_request(request)

        # Wait for result
        try:
            result = self.result_queue.get(timeout=timeout)
            if result.request_id == request_id:
                return result
        except queue.Empty:
            return None

    def _worker_loop(self):
        """Worker thread loop for processing batches."""
        while self.running:
            try:
                # Get next batch
                batch_data = self.batcher.get_batch(timeout=0.1)

                if batch_data is None:
                    continue

                requests, batched_input = batch_data

                # Run inference
                with self.profiler.profile('batch_inference'):
                    predictions = self.cache.forward(batched_input)

                # Split predictions and create results
                batch_size = len(requests)
                for i, request in enumerate(requests):
                    # Extract prediction for this request
                    if isinstance(predictions, dict):
                        pred = {k: v[i] for k, v in predictions.items()}
                    else:
                        pred = predictions[i]

                    # Calculate latency
                    latency_ms = (time.time() - request.timestamp) * 1000
                    self.profiler.record(latency_ms, 'end_to_end')

                    # Create result
                    result = InferenceResult(
                        request_id=request.request_id,
                        predictions=pred,
                        latency_ms=latency_ms,
                        batch_size=batch_size,
                    )

                    # Send to result queue
                    self.result_queue.put(result)

                    # Invoke callback if provided
                    if request.callback is not None:
                        request.callback(result)

            except Exception as e:
                print(f"Error in worker loop: {e}")
                continue

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics.

        Returns
        -------
        dict
            Latency and throughput statistics.
        """
        return {
            'latency': self.profiler.get_stats(),
            'batch_latency': self.profiler.get_stats('batch_inference'),
            'end_to_end': self.profiler.get_stats('end_to_end'),
            'queue_size': len(self.batcher.queue),
        }

    def save_stats(self, save_path: str):
        """Save statistics to JSON file.

        Parameters
        ----------
        save_path : str
            Path to save JSON file.
        """
        stats = self.get_stats()
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=2)

        print(f"Statistics saved to {save_path}")
