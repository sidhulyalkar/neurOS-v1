# neurOS Drivers

Hardware, simulated, and dataset sources for neurOS.

The driver layer adapts acquisition systems to the neurOS streaming contracts. Hardware support is deliberately separated from hardware **qualification**: a driver can satisfy its software contract without implying that every device, firmware, transport, operating system, or montage has been validated.

## Installation

```bash
# Minimal sources and simulated/data workflows
pip install neuros-drivers

# EEG acquisition through BrainFlow + LSL dependencies
pip install "neuros-drivers[eeg]"

# Other optional integrations
pip install "neuros-drivers[video]"
pip install "neuros-drivers[nwb]"
pip install "neuros-drivers[all]"
```

## Mock data

Synthetic data is always explicit. neurOS does not silently replace a requested hardware source with a mock source.

```python
import asyncio

from neuros.drivers import MockDriver


async def main() -> None:
    driver = MockDriver(sampling_rate=250, channels=8)
    await driver.start()
    try:
        async for frame in driver.frames():
            print(frame.sequence_id, frame.data.shape, frame.timestamp_ns)
            break
    finally:
        await driver.stop()


asyncio.run(main())
```

## BrainFlow EEG

`BrainFlowDriver` uses BrainFlow's board metadata to select EEG rows, drains the BrainFlow ringbuffer so buffered samples are not repeatedly emitted, and validates the actual prepared-session sampling rate.

```python
import asyncio

from neuros.drivers import BrainFlowDriver


async def main() -> None:
    driver = BrainFlowDriver(
        board_id=0,
        # serial_port="/dev/ttyUSB0",  # when required by the selected board
    )
    await driver.start()
    try:
        print(driver.descriptor)
        async for frame in driver.frames():
            print(frame.data)
            break
    finally:
        await driver.stop()


asyncio.run(main())
```

If BrainFlow is not installed, construction fails with an actionable error directing the user to install `neuros-drivers[eeg]`. Use `MockDriver` explicitly when simulation is intended.

The optional `sampling_rate=` argument is an **assertion about the expected prepared-session rate**, not a software resampler. If the hardware reports a different rate, startup fails rather than publishing incorrect timing metadata.

## BrainFlow safety semantics

BrainFlow matrices are board-specific. The driver therefore:

- selects channels through `BoardShim.get_eeg_channels(...)` rather than assuming EEG occupies rows `0..N-1`;
- uses BrainFlow's timestamp row when one is available;
- drains data with `get_board_data()` so one buffered sample is not replayed on every polling cycle;
- rejects misspelled/unknown `BrainFlowInputParams` fields;
- rejects impossible requested channel counts;
- validates the actual prepared-session sampling rate when BrainFlow exposes it;
- propagates acquisition and cleanup failures instead of turning them into plausible synthetic data.

These are software safety properties, not evidence that a physical device is qualified. Device qualification still requires measured packet loss, synchronization uncertainty, drift, reconnect behavior, sustained recording reliability, and end-to-end latency for the exact hardware/firmware/software combination.

## Supported acquisition families

Current package surfaces include:

- BrainFlow-compatible EEG/biosignal devices;
- simulated/mock sources;
- dataset playback;
- ECG, ECoG, EMG, EOG, fNIRS, GSR, respiration, and other research-oriented source modules;
- optional video/audio dependencies;
- NWB-related integration dependencies.

Lab Streaming Layer support is part of the broader neurOS interoperability stack. New hardware integrations should prefer the neurOS source/plugin contract rather than adding device-specific behavior to `neuros-core`.

## Contributor rule

A hardware PR should distinguish three different claims:

1. **software contract:** the driver behaves correctly against deterministic fakes/fixtures;
2. **integration:** the real SDK/device can stream through neurOS;
3. **hardware qualification:** a named hardware/firmware/transport/software combination passes a recorded qualification protocol.

Only claim the strongest tier actually tested.

## Project documentation

See the repository's current documentation and maturity map:

- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `ROADMAP.md`

## License

MIT License.
