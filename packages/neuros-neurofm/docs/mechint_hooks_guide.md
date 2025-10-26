# Mechanistic Interpretability Hooks - User Guide

## Overview

The NeuroFMX Mechanistic Interpretability Hooks system provides seamless integration between model training and interpretability analysis. It automatically captures activations during training, manages storage (local/S3), and runs comprehensive analyses at evaluation time.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Components](#core-components)
3. [PyTorch Lightning Integration](#pytorch-lightning-integration)
4. [Manual Integration](#manual-integration)
5. [Evaluation and Analysis](#evaluation-and-analysis)
6. [FastAPI Integration](#fastapi-integration)
7. [Storage Backends](#storage-backends)
8. [Configuration Reference](#configuration-reference)
9. [Advanced Usage](#advanced-usage)
10. [Troubleshooting](#troubleshooting)

---

## Quick Start

### 1. Basic PyTorch Lightning Usage

```python
import pytorch_lightning as pl
from neuros_neurofm.interpretability import MechIntCallback, MechIntConfig
from neuros_neurofm.training import NeuroFMXLightningModule

# Configure mechanistic interpretability
config = MechIntConfig(
    sample_layers=['mamba_backbone.blocks.3', 'popt'],
    save_hidden_every_n_steps=200,
    analyses_to_run=['sae', 'neuron', 'feature']
)

# Add callback to trainer
trainer = pl.Trainer(
    max_epochs=100,
    callbacks=[MechIntCallback(config=config)]
)

# Train normally - activations are captured automatically!
trainer.fit(model, train_dataloader)
```

### 2. Run Post-Training Analysis

```python
from neuros_neurofm.interpretability import EvalMechIntRunner

# Create evaluation runner
runner = EvalMechIntRunner(
    model=model,
    config=config,
    device='cuda'
)

# Run comprehensive analysis
results = runner.run_mechint_eval(
    checkpoint_path='./checkpoints/best.pt',
    hidden_shards_path='./mechint_cache'
)

# Export results
runner.export_results('./mechint_results')
```

---

## Core Components

### MechIntConfig

Configuration dataclass for all mechanistic interpretability settings.

```python
from neuros_neurofm.interpretability import MechIntConfig

config = MechIntConfig(
    # Which layers to track
    sample_layers=['layer1', 'layer2', 'layer3'],

    # How often to save (every N training steps)
    save_hidden_every_n_steps=200,

    # Which analyses to run
    analyses_to_run=['sae', 'neuron', 'circuit', 'feature', 'causal'],

    # Storage settings
    storage_backend='local',  # 'local', 's3', or 'both'
    storage_path='./mechint_cache',

    # S3 settings (if using S3)
    s3_bucket='my-bucket',
    s3_prefix='experiments/exp001',

    # Performance settings
    max_activations_per_shard=10000,

    # Optional features
    enable_feature_steering=False,
    verbose=True
)
```

### ActivationSampler

Low-level component for capturing and saving activations.

```python
from neuros_neurofm.interpretability import ActivationSampler

sampler = ActivationSampler(
    layers=['layer1', 'layer2'],
    save_dir='./activations',
    max_samples_per_shard=5000
)

# Register hooks
sampler.register_hooks(model)

# Forward passes will automatically capture activations
output = model(input_data)

# Save captured activations
save_path = sampler.save_activations(global_step=100)

# Get statistics
stats = sampler.get_statistics()
print(f"Captured {stats['current_cache_samples']} samples")

# Clear cache
sampler.clear_cache()
```

### MechIntHooks

Orchestrator that manages the entire hook system.

```python
from neuros_neurofm.interpretability import MechIntHooks

hooks = MechIntHooks(config)
hooks.register_hooks(model, trainer)

# In training loop
hooks.on_training_step(
    trainer=trainer,
    pl_module=model,
    outputs=outputs,
    batch=batch,
    batch_idx=batch_idx,
    global_step=global_step
)

# At epoch end
hooks.on_epoch_end(trainer, model)

# At training end
hooks.on_train_end(trainer, model)
```

---

## PyTorch Lightning Integration

### Basic Callback

The `MechIntCallback` integrates seamlessly with PyTorch Lightning:

```python
from neuros_neurofm.interpretability import MechIntCallback

callback = MechIntCallback(
    config={
        'sample_layers': ['layer1', 'layer2'],
        'save_hidden_every_n_steps': 100
    }
)

trainer = pl.Trainer(
    callbacks=[callback],
    max_epochs=50
)

trainer.fit(model, train_dataloader)
```

### Multiple Callbacks

Combine with other callbacks:

```python
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

trainer = pl.Trainer(
    callbacks=[
        MechIntCallback(config=mechint_config),
        ModelCheckpoint(monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=10)
    ]
)
```

### Custom Lightning Module

```python
class MyNeuroFMX(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = MultiModalNeuroFMX(...)

    def training_step(self, batch, batch_idx):
        # Your training logic
        outputs = self.model(batch)
        loss = self.compute_loss(outputs, batch)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

# MechIntCallback works automatically with any LightningModule!
trainer = pl.Trainer(callbacks=[MechIntCallback(config)])
trainer.fit(MyNeuroFMX())
```

---

## Manual Integration

For custom training loops without PyTorch Lightning:

```python
from neuros_neurofm.interpretability import MechIntHooks, MechIntConfig

# Setup
config = MechIntConfig(
    sample_layers=['layer1', 'layer2'],
    save_hidden_every_n_steps=100
)

hooks = MechIntHooks(config)
hooks.register_hooks(model)

# Training loop
optimizer = torch.optim.Adam(model.parameters())

for epoch in range(num_epochs):
    for batch_idx, batch in enumerate(train_loader):
        global_step = epoch * len(train_loader) + batch_idx

        # Forward pass (activations captured automatically)
        outputs = model(batch)
        loss = criterion(outputs, targets)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Save activations periodically
        if global_step % config.save_hidden_every_n_steps == 0:
            # Trigger save manually
            if hooks.sampler:
                save_path = hooks.sampler.save_activations(global_step)
                hooks.saved_shards.append(save_path)
                hooks.sampler.clear_cache()

    # Epoch end
    print(f"Epoch {epoch} complete")

# Training end - generate manifest
hooks.on_train_end(trainer=None, pl_module=model)
```

---

## Evaluation and Analysis

### Running Full Analysis Suite

```python
from neuros_neurofm.interpretability import EvalMechIntRunner

runner = EvalMechIntRunner(
    model=model,
    config=MechIntConfig(
        sample_layers=['layer1', 'layer2', 'layer3'],
        analyses_to_run=['sae', 'neuron', 'feature', 'circuit']
    ),
    device='cuda'
)

# Run all analyses
results = runner.run_mechint_eval(
    checkpoint_path='./checkpoints/model_epoch_50.pt',
    hidden_shards_path='./mechint_cache',
    eval_data=eval_dataloader  # Optional, for causal analysis
)

# Results structure
print(results.keys())
# >>> ['config', 'analyses']

print(results['analyses'].keys())
# >>> ['sae', 'neuron', 'feature', 'circuit']
```

### SAE Analysis

```python
# SAE results contain trained autoencoders
sae_results = results['analyses']['sae']

for layer_name, layer_results in sae_results.items():
    print(f"\n{layer_name}:")
    print(f"  Final L0 sparsity: {layer_results['final_l0']:.3f}")
    print(f"  Final loss: {layer_results['final_loss']:.6f}")

    # Access training statistics
    training_stats = layer_results['training_stats']
    print(f"  Training epochs: {len(training_stats[f'{layer_name}/loss'])}")
```

### Neuron Analysis

```python
# Neuron analysis provides activation statistics
neuron_results = results['analyses']['neuron']

for layer_name, stats in neuron_results.items():
    mean_act = np.array(stats['mean_activation'])
    sparsity = np.array(stats['sparsity'])

    print(f"\n{layer_name}:")
    print(f"  Avg activation: {mean_act.mean():.3f}")
    print(f"  Avg sparsity: {sparsity.mean():.3f}")
    print(f"  Dead neurons: {(sparsity < 0.01).sum()}/{len(sparsity)}")
```

### Feature Analysis

```python
# Feature analysis includes PCA and clustering
feature_results = results['analyses']['feature']

for layer_name, analysis in feature_results.items():
    explained_var = np.array(analysis['explained_variance'])

    print(f"\n{layer_name}:")
    print(f"  Top 10 PCs explain: {explained_var[:10].sum():.2%}")
    print(f"  Total components: {analysis['n_components']}")
```

### Exporting Results

```python
# Export to directory
runner.export_results('./results')

# Creates:
# - results/mechint_results.json (full results)
# - results/mechint_report.md (human-readable report)
```

---

## FastAPI Integration

Add real-time interpretation endpoints to your API server:

```python
from fastapi import FastAPI
from neuros_neurofm.interpretability import FastAPIIntegrationMixin
from neuros_neurofm.api.server import create_app

# Create base app
app = create_app(model_path='./model.pt')

# Add interpretation endpoints
mixin = FastAPIIntegrationMixin(
    model=model,
    config=MechIntConfig(
        sample_layers=['layer1', 'layer2'],
        storage_path='./mechint_cache'
    ),
    device='cuda'
)

mixin.add_routes(app)

# Now you have:
# - POST /interpret - Run analysis on cached activations
# - POST /interpret/upload - Upload and analyze activations
# - GET /interpret/layers - List available layers
```

### API Usage Examples

**List available layers:**
```bash
curl http://localhost:8000/interpret/layers
```

**Run neuron analysis:**
```bash
curl -X POST http://localhost:8000/interpret \
  -H "Content-Type: application/json" \
  -d '{
    "analysis_type": "neuron",
    "layer_name": "layer1",
    "config": {}
  }'
```

**Upload and analyze activations:**
```bash
curl -X POST http://localhost:8000/interpret/upload \
  -F "file=@activations.pt" \
  -F "analysis_type=feature"
```

---

## Storage Backends

### Local Storage

Default storage backend - saves to local filesystem:

```python
config = MechIntConfig(
    storage_backend='local',
    storage_path='./mechint_cache'
)
```

Directory structure:
```
mechint_cache/
├── activations_shard_000000_step_0.pt
├── activations_shard_000001_step_200.pt
├── activations_shard_000002_step_400.pt
└── manifest.json
```

### S3 Storage

Automatically upload to S3:

```python
config = MechIntConfig(
    storage_backend='s3',
    storage_path='./local_cache',  # Temporary local cache
    s3_bucket='my-neurofmx-bucket',
    s3_prefix='experiments/exp001/activations'
)
```

Requirements:
- Install: `pip install boto3`
- AWS credentials configured (via environment variables or ~/.aws/credentials)

### Hybrid Storage (Both)

Save locally AND upload to S3:

```python
config = MechIntConfig(
    storage_backend='both',
    storage_path='./mechint_cache',
    s3_bucket='my-neurofmx-bucket',
    s3_prefix='experiments/exp001'
)
```

Useful for:
- Redundant storage
- Local access during training
- Cloud backup for long-term storage

---

## Configuration Reference

### Complete Configuration Options

```python
config = MechIntConfig(
    # === Layer Selection ===
    sample_layers=['mamba_backbone.blocks.3', 'popt'],
    # Default: ['mamba_backbone.blocks.3', 'popt']

    # === Sampling Frequency ===
    save_hidden_every_n_steps=200,
    # Default: 200
    # Save activations every N training steps

    # === Analyses ===
    analyses_to_run=['sae', 'neuron', 'circuit', 'feature', 'causal'],
    # Default: ['sae', 'neuron', 'feature']
    # Available: 'sae', 'neuron', 'circuit', 'causal', 'feature'

    # === Storage Backend ===
    storage_backend='local',
    # Default: 'local'
    # Options: 'local', 's3', 'both'

    storage_path='./mechint_cache',
    # Default: './mechint_cache'
    # Local storage directory

    # === S3 Settings ===
    s3_bucket=None,
    # Default: None
    # S3 bucket name

    s3_prefix='neurofmx/activations',
    # Default: 'neurofmx/activations'
    # S3 key prefix

    # === Performance ===
    max_activations_per_shard=10000,
    # Default: 10000
    # Maximum samples per shard file

    # === Features ===
    enable_feature_steering=False,
    # Default: False
    # Enable feature steering experiments

    # === Logging ===
    verbose=True
    # Default: True
    # Enable verbose logging
)
```

---

## Advanced Usage

### Auto-Detecting Layers

Let the system automatically find interesting layers:

```python
config = MechIntConfig(
    sample_layers=None,  # Will auto-detect
    # Finds: Mamba blocks, Perceiver layers, PopT aggregator
)
```

### Custom Activation Processing

Subclass `ActivationSampler` for custom processing:

```python
from neuros_neurofm.interpretability import ActivationSampler

class CustomSampler(ActivationSampler):
    def save_activations(self, global_step, metadata=None):
        # Custom preprocessing
        for layer_name, acts in self.activations.items():
            # Normalize activations
            self.activations[layer_name] = [
                (a - a.mean()) / (a.std() + 1e-8)
                for a in acts
            ]

        # Call parent
        return super().save_activations(global_step, metadata)
```

### Multi-GPU Training

Hooks work seamlessly with DDP/FSDP:

```python
trainer = pl.Trainer(
    strategy='ddp',  # or 'fsdp'
    devices=8,
    callbacks=[MechIntCallback(config)]
)

# Activations are only saved on rank 0 to avoid duplicates
```

### Checkpointing and Resumption

Resume training with hooks:

```python
# Initial training
trainer = pl.Trainer(callbacks=[MechIntCallback(config)])
trainer.fit(model)

# Resume training
trainer = pl.Trainer(
    callbacks=[MechIntCallback(config)],
    resume_from_checkpoint='./checkpoints/last.ckpt'
)
trainer.fit(model)

# Activation shards continue from where they left off
```

---

## Troubleshooting

### Issue: Hooks not capturing activations

**Symptoms:** Empty activation cache, no shard files created

**Solutions:**
1. Verify layer names are correct:
```python
# Print all layer names
for name, _ in model.named_modules():
    print(name)
```

2. Check if model forward is called:
```python
# Add debug logging
config = MechIntConfig(verbose=True)
```

3. Ensure hooks are registered before forward pass:
```python
hooks.register_hooks(model)
# Then run training
```

### Issue: Out of memory

**Symptoms:** CUDA OOM errors, slow training

**Solutions:**
1. Reduce shard size:
```python
config = MechIntConfig(max_activations_per_shard=1000)
```

2. Save less frequently:
```python
config = MechIntConfig(save_hidden_every_n_steps=1000)
```

3. Track fewer layers:
```python
config = MechIntConfig(sample_layers=['layer_final'])
```

### Issue: S3 upload fails

**Symptoms:** Warnings about S3 upload failures

**Solutions:**
1. Check AWS credentials:
```bash
aws s3 ls s3://my-bucket/
```

2. Verify bucket exists and you have permissions

3. Fall back to local storage:
```python
config = MechIntConfig(storage_backend='local')
```

### Issue: Analysis fails

**Symptoms:** Empty results, errors during analysis

**Solutions:**
1. Check if activation shards exist:
```python
import glob
shards = glob.glob('./mechint_cache/activations_shard_*.pt')
print(f"Found {len(shards)} shards")
```

2. Verify shard format:
```python
data = torch.load(shards[0])
print(data.keys())  # Should have 'activations', 'metadata'
```

3. Run individual analyses:
```python
runner = EvalMechIntRunner(model, config)
activations = runner._load_activations('./mechint_cache')
results = runner._run_neuron_analysis(activations)
```

### Issue: Slow training

**Symptoms:** Training is significantly slower with hooks

**Solutions:**
1. Use CPU for activation storage:
```python
sampler = ActivationSampler(device='cpu')
```

2. Save less frequently:
```python
config = MechIntConfig(save_hidden_every_n_steps=500)
```

3. Disable verbose logging:
```python
config = MechIntConfig(verbose=False)
```

---

## Best Practices

1. **Start Small:** Begin with 1-2 layers and low frequency
2. **Monitor Storage:** Check disk space regularly
3. **Use S3 for Long Runs:** Prevent local disk filling up
4. **Save Frequently During Dev:** Use smaller intervals for debugging
5. **Run Analysis Separately:** Don't analyze during training
6. **Keep Manifests:** The `manifest.json` tracks all shards
7. **Version Experiments:** Use descriptive `s3_prefix` values

---

## Additional Resources

- **Examples:** `examples/mechint_hooks_example.py`
- **Tests:** `tests/test_mechint_hooks.py`
- **API Reference:** See docstrings in `hooks.py`
- **Related Modules:**
  - SAE Training: `interpretability/sae_training.py`
  - Feature Analysis: `interpretability/feature_analysis.py`
  - Circuit Discovery: `interpretability/circuit_discovery.py`

---

## Summary

The Mechanistic Interpretability Hooks system provides:

- **Automatic Integration:** Works seamlessly with PyTorch Lightning
- **Flexible Storage:** Local, S3, or both
- **Comprehensive Analysis:** SAE, neuron, circuit, feature, causal
- **Minimal Overhead:** Efficient hook-based capture
- **Production Ready:** FastAPI endpoints for real-time interpretation

Start with the [Quick Start](#quick-start) guide and explore the examples!
