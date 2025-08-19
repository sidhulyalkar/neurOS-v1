import pytest
import asyncio

from neuros.benchmarks.benchmark_pipeline import run_benchmark


@pytest.mark.asyncio
async def test_run_benchmark():
    metrics = await run_benchmark(duration=2.0)
    assert "accuracy" in metrics
    assert metrics["accuracy"] >= 0.0 and metrics["accuracy"] <= 1.0
    assert metrics["throughput"] > 0
    assert metrics["mean_latency"] >= 0