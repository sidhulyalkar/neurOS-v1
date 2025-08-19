import asyncio

import numpy as np

from neuros.drivers.dataset_driver import DatasetDriver


async def test_dataset_driver_stream():
    # instantiate the driver with a small sampling rate to speed up the test
    driver = DatasetDriver(dataset_name="iris", sampling_rate=10.0)
    await driver.start()
    count = 0
    async for ts, data in driver:
        assert isinstance(data, np.ndarray)
        # ensure data dimensionality matches channels
        assert data.ndim == 1
        assert len(data) == driver.channels
        count += 1
        if count >= 5:
            break
    await driver.stop()