import asyncio

from neuros.drivers.brainflow_driver import BrainFlowDriver


async def test_brainflow_fallback():
    driver = BrainFlowDriver()
    # should either use brainflow if installed or fallback to mock
    assert driver.sampling_rate > 0
    assert driver.channels > 0
    # start and stop driver
    await driver.start()
    # obtain a few samples
    async for ts, data in driver:
        assert isinstance(data, list)
        assert len(data) == driver.channels
        break
    await driver.stop()