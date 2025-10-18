# NeuroFM-X: Foundation Model for Neural Population Dynamics

**NeuroFM-X** is a state-of-the-art foundation model for neural population dynamics that combines selective state-space models (Mamba/SSM), multi-modal fusion (Perceiver-IO), population transformers (PopT), and latent diffusion for generative modeling.

## Key Features

- **Linear Complexity**: Mamba/SSM backbone processes sequences in O(L) time (5x faster than Transformers)
- **Multi-Modal Fusion**: Perceiver-IO handles EEG, spikes, LFP, calcium imaging, and behavior
- **Population-Level Learning**: PopT aggregates information across neural populations
- **Generative Modeling**: Latent diffusion for forecasting and imputation (1-2s horizons)
- **Transfer Learning**: POYO/Unit-ID adapters with frozen core for few-shot adaptation
- **Contrastive Learning**: CEBRA-style behavior-aligned latent spaces

## Architecture Overview

```
Input Data (Spikes/LFP/EEG/Behavior)
  ↓
Neural Tokenizers (spikes-as-tokens, binned tensors, LFP encoders)
  ↓
Mamba/SSM Backbone (d_model=768, 16 blocks, multi-rate streams)
  ↓
Perceiver-IO Fusion Hub (512-dim latents, 128 slots)
  ↓
PopT Population Aggregator (3 layers, width 512)
  ↓
Latent Diffusion Prior (1-2s forecast horizon)
  ↓
Multi-Task Heads (decoding, encoding, contrastive)
  ↓
Adapters (Unit-ID, session/region stitchers, LoRA)
```

## Installation

```bash
# Install from source
cd packages/neuros-neurofm
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Install with dataset tools
pip install -e ".[datasets]"

# Install everything
pip install -e ".[all]"
```

## Quick Start

### Run the Demo

```bash
cd packages/neuros-neurofm
python examples/quickstart_demo.py
```

This will:
- Generate synthetic neural data
- Train a NeuroFM-X model
- Evaluate behavioral decoding (R² ~ 0.58)
- Test transfer learning with adapters
- Save/load model checkpoints

### Basic Usage

```python
from neuros_neurofm.datasets import SyntheticNeuralDataset, create_dataloaders
from neuros_neurofm.tokenizers import BinnedTokenizer
from neuros_neurofm.fusion import PerceiverIO
from neuros_neurofm.models import PopT, MultiTaskHeads

# Create dataset
dataset = SyntheticNeuralDataset(n_samples=1000, n_units=96)
train_loader, val_loader = create_dataloaders(dataset)

# Build model pipeline
tokenizer = BinnedTokenizer(n_units=96, d_model=256)
fusion = PerceiverIO(n_latents=32, latent_dim=128, input_dim=256)
popt = PopT(d_model=128, n_output_seeds=1)
heads = MultiTaskHeads(input_dim=128, decoder_output_dim=2)

# Tokenize and predict
tokens, mask = tokenizer(neural_data)  # (batch, seq, units) → (batch, seq, 256)
latents = fusion(tokens, mask)  # (batch, seq, 256) → (batch, 32, 128)
aggregated = popt(latents)  # (batch, 32, 128) → (batch, 128)
behavior = heads(aggregated, task="decoder")  # (batch, 128) → (batch, 2)
```

### Complete Model

```python
from neuros_neurofm.models.neurofmx_complete import NeuroFMXComplete

# Load model
model = NeuroFMX(
    d_model=768,
    n_blocks=16,
    n_latents=128,
    latent_dim=512,
)

# Setup data
datamodule = NWBDataModule(
    data_dir="/path/to/nwb/files",
    batch_size=32,
    num_workers=4,
)

# Pretrain
trainer = NeuroFMXPretrainer(
    model=model,
    datamodule=datamodule,
    max_epochs=100,
    accelerator="gpu",
)
trainer.fit()
```

### Fine-tuning with Adapters

```python
from neuros_neurofm.adapters import UnitIDAdapter

# Add Unit-ID adapter
adapter = UnitIDAdapter(
    input_dim=768,
    n_units=96,
    freeze_backbone=True,
)
model.add_adapter(adapter)

# Fine-tune on target dataset
trainer.fit(model, target_datamodule)
```

### Inference

```python
# Make predictions
predictions = model.predict(neural_data, behavior_data)

# Generate future states (1-2s ahead)
generated = model.generate(context, num_steps=500)  # 500ms @ 1kHz

# Extract features for downstream tasks
features = model.encode(neural_data)
```

## Benchmarks

NeuroFM-X is evaluated on:

- **IBL Repeated Site**: Motor cortex recordings with behavioral tasks
- **Allen Brain Observatory**: Visual cortex calcium imaging and Neuropixels
- **DANDI Archive**: Public iEEG datasets for speech/handwriting
- **FALCON Benchmark**: Few-shot robustness evaluation

### Performance Targets

| Task | Metric | Target | Status |
|------|--------|--------|--------|
| Behavioral Decoding | R² | >0.60 | 🔄 |
| Neural Forecasting | BPS | >2.5 | 🔄 |
| Few-Shot Transfer | Accuracy | >0.70 | 🔄 |
| Inference Latency | ms/sample | <10 | 🔄 |
| Throughput | samples/s | >1000 | 🔄 |

## Integration with neurOS

NeuroFM-X integrates seamlessly with the neurOS pipeline:

```python
from neuros.pipeline import Pipeline
from neuros_neurofm.models import NeuroFMX

# Create neurOS pipeline with NeuroFM-X model
pipeline = Pipeline(
    driver=your_driver,
    model=NeuroFMX.from_pretrained("neurofmx-base"),
    fs=1000.0,
)

# Run real-time inference
metrics = await pipeline.run(duration=60.0)
```

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Training Guide](docs/training.md)
- [API Reference](docs/api_reference.md)
- [Tutorials](tutorials/)
- [Integration Plan](../../docs/NEUROFM_X_PLAN.md)

## Citation

If you use NeuroFM-X in your research, please cite:

```bibtex
@software{neurofmx2024,
  title={NeuroFM-X: Foundation Model for Neural Population Dynamics},
  author={neurOS Team},
  year={2024},
  url={https://github.com/shulyalk/neuros-v1}
}
```

## License

MIT License - see [LICENSE](../../LICENSE) for details.
