# neurOS Drivers

Hardware, simulated, network-stream, and dataset sources for neurOS.

The driver layer adapts acquisition systems to the neurOS streaming contracts. Hardware support is deliberately separated from hardware **qualification**: a driver can satisfy its software contract without implying that every device, firmware, transport, operating system, montage, or network topology has been validated.

## Installation

```bash
# Minimal sources and simulated/data workflows
pip install neuros-drivers

# EEG acquisition through BrainFlow + LSL dependencies
pip install "neuros-drivers[eeg]"

# LSL-only network acquisition
pip install "neuros-drivers[lsl]"

# Other optional integrations
pip install "neuros-drivers[video]"
pip install "neuros-drivers[nwb]"
pip install "neuros-drivers[all]"
```

## Mock data

Synthetic data is always explicit. neurOS does not silently replace a requested hardware or network source with a mock source.

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

## Lab Streaming Layer

`LSLDriver` is the first-class neurOS source for continuous regular-rate Lab Streaming Layer streams. It is designed for synchronized multimodal lab acquisition, not for silently choosing whichever outlet happens to be visible first.

```python
import asyncio

from neuros.drivers import LSLDriver


async def main() -> None:
    driver = LSLDriver(
        source_id="my-eeg-headset",
        stream_type="EEG",
        sampling_rate=250,  # expected rate assertion
        channels=8,         # expected geometry assertion
    )
    await driver.start()
    try:
        print(driver.descriptor)
        async for frame in driver.frames():
            print(frame.sequence_id, frame.data, frame.timestamp_seconds)
            break
    finally:
        await driver.stop()


asyncio.run(main())
```

At least one exact selector is required: `source_id`, `name`, or `stream_type`. `source_id` is preferred whenever the producer exposes a stable identity. If the supplied selectors still match multiple streams, neurOS fails startup and asks for a more specific selector instead of binding nondeterministically.

### LSL timing semantics

The v1 source keeps liblsl post-processing flags disabled and records its timing transformation explicitly:

```text
raw LSL sample timestamp
        +
LSL inlet time_correction estimate
        =
SignalFrame.synchronized_time_ns
```

Each canonical frame is published with `ClockDomain.SYNCHRONIZED` and retains both `lsl_raw_timestamp_seconds` and `lsl_time_correction_seconds` in metadata. The correction estimate can be refreshed periodically without hiding the exact correction used for any emitted frame.

This is intentional. neurOS does not silently enable LSL dejittering, monotonization, or other timestamp post-processing at the acquisition boundary. Those operations can materially change timing evidence and should be explicit transforms in a qualified pipeline.

The initial first-class source supports **continuous streams with a positive nominal sampling rate**. Irregular event/marker streams should use a dedicated event contract rather than pretending that a zero-rate marker channel is sampled neural data.

When `recover=True`, liblsl recovery is enabled only when the discovered stream exposes a non-empty `source_id`. Expected `sampling_rate=` and `channels=` values are startup assertions, not resampling or channel-selection instructions.

## LSL software qualification boundary

The deterministic LSL contract tests verify, without requiring a physical network stream:

- fail-closed optional dependency behavior;
- deterministic and unambiguous stream selection;
- continuous-rate and channel-geometry validation;
- explicit raw-timestamp plus time-correction semantics;
- synchronized `SignalFrame` publication;
- disabled hidden timestamp post-processing;
- recovery gating on `source_id`;
- malformed-chunk failure propagation and inlet cleanup.

They do **not** establish a hardware/network qualification result. A named LSL deployment still needs measured clock offset/uncertainty, packet loss, reconnect behavior, topology and firewall behavior, long-run buffer integrity, and end-to-end latency on the actual hosts and network.

## Supported acquisition families

Current package surfaces include:

- BrainFlow-compatible EEG/biosignal devices;
- first-class continuous Lab Streaming Layer sources;
- simulated/mock sources;
- dataset playback;
- ECG, ECoG, EMG, EOG, fNIRS, GSR, respiration, and other research-oriented source modules;
- optional video/audio dependencies;
- NWB-related integration dependencies.

New hardware integrations should prefer the neurOS source/plugin contract rather than adding device-specific behavior to `neuros-core`.

## Contributor rule

A hardware or live-stream PR should distinguish three different claims:

1. **software contract:** the driver behaves correctly against deterministic fakes/fixtures;
2. **integration:** the real SDK/device/network stream can run through neurOS;
3. **hardware qualification:** a named hardware/firmware/transport/software/network combination passes a recorded qualification protocol.

Only claim the strongest tier actually tested.

## Project documentation

See the repository's current documentation and maturity map:

- `docs/PROJECT_STATUS.md`
- `docs/ARCHITECTURE.md`
- `docs/API_REFERENCE.md`
- `ROADMAP.md`

## License

MIT License.
