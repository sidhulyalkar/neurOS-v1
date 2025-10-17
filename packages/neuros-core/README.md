# neurOS Core

Core functionality for neurOS - a modular operating system for brain-computer interfaces.

## Features

- **Pipeline Architecture**: Composable pipelines for data processing
- **Multi-Modal Orchestration**: Synchronize and process heterogeneous data streams
- **Agent Framework**: DeviceAgent, ProcessingAgent, ModelAgent, FusionAgent
- **Configuration Management**: Unified config system with YAML support
- **Signal Processing**: Base processing utilities and feature extraction

## Installation

```bash
pip install neuros-core
```

## Quick Start

```python
from neuros.core.pipeline import Pipeline
from neuros.core.agents import DeviceAgent, ModelAgent

# Create a simple pipeline
pipeline = Pipeline(
    driver=my_driver,
    model=my_model
)

# Process data
predictions = pipeline.predict(data)
```

## Dependencies

Minimal dependencies for fast, lightweight installations:
- numpy>=1.24.0
- scipy>=1.11.0
- pyyaml>=6.0
- python-dotenv>=1.0.0
- pydantic>=1.10.10

## Documentation

Full documentation: https://neuros.readthedocs.io

## License

MIT License
