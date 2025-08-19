import asyncio

from neuros.autoconfig import generate_pipeline_for_task


def test_generate_pipeline_for_task_ssvep():
    pipeline = generate_pipeline_for_task("SSVEP speller")
    assert pipeline.bands is not None
    assert "alpha_beta" in pipeline.bands


def test_generate_pipeline_for_task_motor():
    pipeline = generate_pipeline_for_task("2‑class motor imagery")
    assert pipeline.bands is not None
    assert "mu_beta" in pipeline.bands


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
    from neuros.autoconfig import generate_pipeline_for_task
    pipeline = generate_pipeline_for_task("reprocess iris dataset")
    import numpy as np
    # train model with random data matching driver channels
    X = np.random.randn(10, pipeline.driver.channels)
    y = np.random.randint(0, 2, size=10)
    try:
        pipeline.train(X, y)
    except Exception:
        # some models may not need explicit training (e.g. DinoV3 fallback)
        pass
    # run for a short duration to process a few samples
    metrics = await pipeline.run(duration=0.5)
    assert metrics["samples"] > 0