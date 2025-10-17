# neurOS Drivers

Hardware drivers for neurOS - EEG, video, audio, and multi-modal data acquisition.

## Features

- **EEG/BCI Devices**: BrainFlow integration for 20+ devices
- **Lab Streaming Layer**: LSL for synchronized multi-modal capture
- **Video Capture**: OpenCV-based video/camera drivers
- **Audio Recording**: Microphone and audio driver support
- **NWB I/O**: Neurodata Without Borders file format support
- **Mock Drivers**: Simulation drivers for testing

## Installation

```bash
# Minimal installation
pip install neuros-drivers

# With specific hardware support
pip install neuros-drivers[eeg]        # BrainFlow + LSL
pip install neuros-drivers[video]      # Camera support
pip install neuros-drivers[all]        # Everything
```

## Quick Start

```python
from neuros.drivers import MockDriver

# Create a mock EEG driver
driver = MockDriver(
    n_channels=64,
    sampling_rate=250,
    modality='eeg'
)

# Start streaming
driver.start_stream()
data = driver.get_data(duration=1.0)
driver.stop_stream()
```

## Supported Devices

- **EEG**: OpenBCI, BrainAmp, g.Nautilus, Muse, and more via BrainFlow
- **Video**: Any OpenCV-compatible camera
- **Audio**: System microphones via PyAudio
- **Multi-modal**: LSL-compatible devices

## Documentation

Full documentation: https://neuros.readthedocs.io

## License

MIT License
