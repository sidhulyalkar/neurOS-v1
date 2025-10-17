# neurOS Cloud

Cloud infrastructure for neurOS - Kafka, AWS SageMaker, WebDataset, Zarr export.

## Features

- **Kafka Integration**: Real-time data streaming at scale
- **AWS SageMaker**: Cloud-based model training
- **WebDataset Export**: Efficient dataset format for PyTorch
- **Zarr Export**: Cloud-native chunked array storage
- **Distributed Processing**: Multi-node data pipelines
- **Monitoring**: Prometheus, MLflow, W&B integration

## Installation

```bash
# Minimal installation
pip install neuros-cloud

# With Kafka support
pip install neuros-cloud[kafka]

# With AWS integration
pip install neuros-cloud[aws]

# With export formats
pip install neuros-cloud[export]

# Everything
pip install neuros-cloud[all]
```

## Quick Start

### Kafka Streaming

```python
from neuros.cloud import KafkaWriter

# Stream data to Kafka
writer = KafkaWriter(
    bootstrap_servers='localhost:9092',
    topic='neuros-eeg'
)

writer.write(data, metadata={'session_id': '001'})
```

### WebDataset Export

```python
from neuros.export import WebDatasetExporter

# Export to WebDataset format
exporter = WebDatasetExporter(output_dir='./data')
exporter.export(samples, labels)
```

### SageMaker Training

```python
from neuros.training import SageMakerLauncher

# Launch training job on SageMaker
launcher = SageMakerLauncher(
    role='arn:aws:iam::...',
    instance_type='ml.p3.2xlarge'
)

launcher.train(training_script='train.py', data_path='s3://...')
```

## Use Cases

- Large-scale data collection
- Distributed model training
- Multi-site experiments
- Cloud-based inference
- Dataset publishing

## Documentation

Full documentation: https://neuros.readthedocs.io

## License

MIT License
