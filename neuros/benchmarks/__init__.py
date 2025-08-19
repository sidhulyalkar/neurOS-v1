"""
Benchmarking utilities for neurOS.

This package contains functions to measure the performance of the pipeline.
Benchmarks report throughput, latency and accuracy metrics to facilitate
comparisons with other BCI frameworks.
"""

from .benchmark_pipeline import run_benchmark  # noqa: F401