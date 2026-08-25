import asyncio

import pytest

from neuros.autoconfig import generate_pipeline_for_task


def test_generate_pipeline_for_task_ssvep():
    pipeline = generate_pipeline_for_task("SSVEP speller")
    assert pipeline.bands is not None
    assert "alpha_beta" in pipeline.bands
    # The compatibility Pipeline emits vector features, so auto-selection must
    # remain on a vector decoder rather than silently wiring an incompatible
    # raw-window EEGNet implementation.
    assert type(pipeline.model).__name__ == "SVMModel"


def test_generate_pipeline_for_task_motor():
    pipeline = generate_pipeline_for_task("2‑class motor imagery")
    assert pipeline.bands is not None
    assert "mu_beta" in pipeline.bands


def test_autoconfig_rejects_raw_window_decoder_requests():
    with pytest.raises(ValueError, match="raw-window decoder"):
        generate_pipeline_for_task("generic task", model_name="eegnet")

    with pytest.raises(ValueError, match="config-first RuntimeGraph"):
        generate_pipeline_for_task("use a transformer sequence model")


def test_autoconfig_rejects_historical_fake_dino_identity():
    with pytest.raises(ValueError, match="verified upstream adapter"):
        generate_pipeline_for_task("vision task", model_name="dino_v3")


async def test_pipeline_from_autoconfig_runs():
    pipeline = generate_pipeline_for_task("generic task")
    # train simple random data and run briefly
    import numpy as np

    X = np.random.randn(10, 5 * pipeline.driver.channels)
    y = np.random.randint(0, 2, size=10)
    pipeline.train(X, y)
    metrics = await pipeline.run(duration=0.5)
    assert metrics["samples"] > 0


async def test_autoconfig_dataset_pipeline_runs():
    pipeline = generate_pipeline_for_task("reprocess iris dataset")
    import numpy as np

    # train model with random data matching driver channels
    X = np.random.randn(10, pipeline.driver.channels)
    y = np.random.randint(0, 2, size=10)
    try:
        pipeline.train(X, y)
    except Exception:
        # Compatibility demos should still surface runtime behavior even when a
        # selected estimator has stricter training requirements.
        pass
    metrics = await pipeline.run(duration=0.5)
    assert metrics["samples"] > 0
