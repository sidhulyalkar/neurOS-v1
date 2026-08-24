"""User-facing neurOS benchmark composition helpers.

Benchmarks that instantiate concrete drivers and decoders belong in the
``neuros`` meta-distribution rather than ``neuros-core``.
"""

from .benchmark_pipeline import run_benchmark

__all__ = ["run_benchmark"]
