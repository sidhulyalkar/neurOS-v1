import pytest
import asyncio

from neuros.pipeline import Pipeline
from neuros.drivers.mock_driver import MockDriver
from neuros.models.simple_classifier import SimpleClassifier


@pytest.mark.asyncio
async def test_pipeline_run_returns_metrics():
    pipeline = Pipeline(driver=MockDriver(), model=SimpleClassifier())
    # train on random data
    import numpy as np

    X_train = np.random.randn(50, 5 * pipeline.driver.channels)
    y_train = np.random.randint(0, 2, size=50)
    pipeline.train(X_train, y_train)
    metrics = await pipeline.run(duration=1.0)
    assert isinstance(metrics, dict)
    assert metrics["samples"] > 0
    assert metrics["throughput"] > 0
    assert metrics["mean_latency"] >= 0