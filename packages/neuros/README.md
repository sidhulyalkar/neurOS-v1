# neurOS

A modular operating system for brain-computer interfaces.

## Overview

neurOS is a comprehensive platform for building BCI applications, from hardware integration to deep learning models. This meta-package provides convenient installation options for common use cases.

## Installation

### Standard Installation (Recommended)

Includes core functionality, drivers, and models:

```bash
pip install neuros
```

### Minimal Installation

Only core functionality (lightweight, ~50MB):

```bash
pip install neuros-core
```

### Full Installation

Everything including foundation models, UI, and cloud:

```bash
pip install neuros[all]
```

### Custom Installations

```bash
# BCI applications
pip install neuros[bci]        # Core + EEG drivers + PyTorch models + Dashboard

# Research with foundation models
pip install neuros[research]   # Models + POYO/NDT/CEBRA

# Production deployment
pip install neuros[deployment] # UI + Cloud infrastructure
```

### Individual Packages

```bash
pip install neuros-core        # Core pipeline and agents
pip install neuros-drivers     # Hardware drivers (EEG, video, audio)
pip install neuros-models      # Deep learning models
pip install neuros-foundation  # Foundation models (POYO, NDT, CEBRA)
pip install neuros-ui          # Dashboard and API
pip install neuros-cloud       # Cloud infrastructure
```

## Quick Start

```python
from neuros.drivers import MockDriver
from neuros.models import EEGNet
from neuros.core.pipeline import Pipeline

# Create a simple BCI pipeline
driver = MockDriver(n_channels=64, sampling_rate=250)
model = EEGNet(n_classes=4, n_channels=64, sampling_rate=250)
pipeline = Pipeline(driver=driver, model=model)

# Train
pipeline.train(X_train, y_train)

# Predict
predictions = pipeline.predict(X_test)
```

## Package Structure

neurOS is organized into focused packages:

- **neuros-core**: Pipeline, orchestrator, agents, signal processing
- **neuros-drivers**: BrainFlow, LSL, video, audio, NWB I/O
- **neuros-models**: EEGNet, Transformers, LSTM, classical ML
- **neuros-foundation**: POYO, NDT, CEBRA, Neuroformer
- **neuros-ui**: Streamlit dashboard, FastAPI server, visualizations
- **neuros-cloud**: Kafka, SageMaker, WebDataset, Zarr

## Features

✅ **Multi-Modal**: Synchronize EEG, video, audio, and more
✅ **Real-Time**: Low-latency processing for live BCI
✅ **Foundation Models**: Pre-trained models for transfer learning
✅ **Cloud-Ready**: Kafka, SageMaker, distributed processing
✅ **Extensible**: Plugin system for custom components

## Documentation

- **Tutorials**: https://neuros.readthedocs.io/tutorials
- **API Reference**: https://neuros.readthedocs.io/api
- **User Guides**: https://neuros.readthedocs.io/guides

## Command Line Interface

```bash
# List available devices
neuros devices

# Start a pipeline
neuros run --config pipeline_config.yaml

# Export data to NWB
neuros export --input raw_data/ --output session.nwb
```

## Examples

See the [notebooks/](https://github.com/<your-user>/neuros2/tree/main/notebooks) directory for tutorials:

- Tutorial 1: Basic Pipelines
- Tutorial 2: Real-Time Processing
- Tutorial 3: Multi-Modal Processing
- Tutorial 4: Custom Models
- Tutorial 5: Benchmarking
- Tutorial 6: NWB Integration

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](https://github.com/<your-user>/neuros2/blob/main/CONTRIBUTING.md) for guidelines.

## License

MIT License

## Citation

If you use neurOS in your research, please cite:

```bibtex
@software{neuros2025,
  title={neurOS: A Modular Operating System for Brain-Computer Interfaces},
  author={neurOS Development Team},
  year={2025},
  url={https://github.com/<your-user>/neuros2}
}
```

## Support

- **Issues**: https://github.com/<your-user>/neuros2/issues
- **Discussions**: https://github.com/<your-user>/neuros2/discussions
- **Documentation**: https://neuros.readthedocs.io
