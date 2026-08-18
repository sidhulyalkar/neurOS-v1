"""Public facade for the neurOS command-line interface.

Compatibility symbols remain exported because downstream tests and scripts have
historically patched them from ``neuros.cli``.
"""

import asyncio as asyncio

from neuros.benchmarks.benchmark_pipeline import run_benchmark as run_benchmark
from neuros.pipeline import Pipeline as Pipeline

from .app import _parse_args, main

__all__ = ["Pipeline", "_parse_args", "asyncio", "main", "run_benchmark"]
