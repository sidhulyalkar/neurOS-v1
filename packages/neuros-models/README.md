# neurOS Models

Deep learning models for neurOS - EEGNet, Transformers, LSTM, and more.

## Features

- **EEGNet**: State-of-the-art CNN for EEG classification
- **Transformer Models**: Attention-based sequence models
- **LSTM Models**: Recurrent networks for temporal data
- **Classical Models**: SVM, Random Forest, k-NN classifiers
- **Training Utilities**: Streamlined training loops and callbacks
- **Evaluation Metrics**: Comprehensive metrics for BCI tasks

## Installation

```bash
# Minimal installation
pip install neuros-models

# With PyTorch support
pip install neuros-models[pytorch]

# With scikit-learn
pip install neuros-models[sklearn]

# Everything
pip install neuros-models[all]
```

## Quick Start

```python
from neuros.models import EEGNet
from neuros.core.pipeline import Pipeline

# Create an EEGNet model
model = EEGNet(
    n_classes=4,
    n_channels=64,
    sampling_rate=250
)

# Use in a pipeline
pipeline = Pipeline(driver=my_driver, model=model)
pipeline.train(X_train, y_train)
predictions = pipeline.predict(X_test)
```

## Available Models

### Deep Learning (PyTorch)
- **EEGNet**: Compact CNN for EEG
- **TransformerModel**: Multi-head attention
- **LSTMModel**: Bidirectional LSTM
- **AttentionFusion**: Multi-modal fusion

### Classical ML (scikit-learn)
- **SimpleClassifier**: Linear, SVM, RF, k-NN
- **GBDTModel**: Gradient-boosted trees

## Documentation

Full documentation: https://neuros.readthedocs.io

## License

MIT License
