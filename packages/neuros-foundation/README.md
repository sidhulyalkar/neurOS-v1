# neurOS Foundation Models

Pre-trained foundation models for neurOS - POYO, NDT, CEBRA, Neuroformer.

## Features

- **POYO/POYO+**: Pre-trained on large-scale neural datasets
- **NDT2/NDT3**: Neural Data Transformers
- **CEBRA**: Consistent EmBeddings of high-dimensional Recordings
- **Neuroformer**: Transformer-based foundation model
- **Allen Datasets**: Easy access to Allen Institute data
- **Transfer Learning**: Fine-tune pre-trained models on your data

## Installation

```bash
# Base installation
pip install neuros-foundation

# With specific models
pip install neuros-foundation[poyo]
pip install neuros-foundation[ndt]
pip install neuros-foundation[cebra]
pip install neuros-foundation[all]
```

## Quick Start

```python
from neuros.foundation_models import POYOModel

# Load pre-trained POYO model
model = POYOModel.from_pretrained('poyo-base')

# Zero-shot inference
embeddings = model.encode(neural_data)

# Fine-tune on your data
model.fine_tune(X_train, y_train)
predictions = model.predict(X_test)
```

## Available Models

- **POYO**: Pre-trained on Neuropixels data
- **POYO+**: Enhanced version with multi-area support
- **NDT2**: Neural Data Transformer v2
- **NDT3**: Latest NDT with improved performance
- **CEBRA**: Self-supervised representation learning
- **Neuroformer**: Large-scale transformer model

## Use Cases

- Zero-shot neural decoding
- Transfer learning from large datasets
- Representation learning
- Cross-dataset generalization

## Documentation

Full documentation: https://neuros.readthedocs.io

## License

MIT License (Note: Individual models may have additional license requirements)
