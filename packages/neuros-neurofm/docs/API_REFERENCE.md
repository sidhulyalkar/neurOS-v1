# NeuroFMX API Reference

Complete API documentation for NeuroFMX foundation model.

## Table of Contents

1. [Core Model](#core-model)
2. [Training Infrastructure](#training-infrastructure)
3. [Data Pipeline](#data-pipeline)
4. [Training Objectives](#training-objectives)
5. [Mechanistic Interpretability](#mechanistic-interpretability)
6. [Evaluation](#evaluation)
7. [Deployment](#deployment)

---

## Core Model

### `NeuroFMX`

Main foundation model class.

```python
from neuros_neurofm.model import NeuroFMX

model = NeuroFMX(
    d_model=768,              # Model dimension
    n_layers=12,              # Number of layers
    n_heads=12,               # Number of attention heads
    architecture='mamba',     # 'mamba' or 'transformer'
    modality_configs={        # Per-modality configuration
        'eeg': {'channels': 64, 'sample_rate': 1000},
        'spikes': {'units': 96, 'sample_rate': 30000},
    },
    fusion_type='perceiver',  # 'perceiver' or 'attention'
    fusion_latents=256,       # Number of fusion latent codes
    enable_lora=False,        # Enable LoRA adapters
    lora_rank=8,             # LoRA rank
)
```

**Parameters:**
- `d_model` (int): Model dimension (default: 768)
- `n_layers` (int): Number of transformer/mamba layers (default: 12)
- `n_heads` (int): Number of attention heads for transformer (default: 12)
- `architecture` (str): Backbone architecture - 'mamba' or 'transformer'
- `modality_configs` (dict): Configuration for each modality
- `fusion_type` (str): Multi-modal fusion - 'perceiver' or 'attention'
- `fusion_latents` (int): Number of latent codes for perceiver fusion
- `enable_lora` (bool): Enable LoRA adapters for fine-tuning
- `lora_rank` (int): Rank for LoRA decomposition

**Methods:**

#### `forward(inputs, return_embeddings=False, return_attention=False)`

Forward pass through the model.

**Args:**
- `inputs` (dict): Dictionary of modality tensors `{modality: (B, T, C)}`
- `return_embeddings` (bool): Return intermediate embeddings
- `return_attention` (bool): Return attention weights

**Returns:**
- If `return_embeddings=False` and `return_attention=False`: `(B, T, D)` output tensor
- Otherwise: dict with keys `{'output', 'embeddings', 'attention'}`

**Example:**
```python
inputs = {
    'eeg': torch.randn(32, 1000, 64),  # (batch, time, channels)
    'spikes': torch.randn(32, 1000, 96),
}

# Simple forward
output = model(inputs)  # (32, 1000, 768)

# Get embeddings and attention
result = model(inputs, return_embeddings=True, return_attention=True)
output = result['output']
embeddings = result['embeddings']  # Dict per layer
attention = result['attention']    # Attention weights
```

---

## Training Infrastructure

### `FSDPTrainer`

Fully Sharded Data Parallel (FSDP) trainer for distributed training.

```python
from neuros_neurofm.training import FSDPTrainer, FSDPConfig

config = FSDPConfig(
    sharding_strategy='FULL_SHARD',  # ZeRO-3 equivalent
    cpu_offload=False,
    mixed_precision='bf16',
    activation_checkpointing=True,
)

trainer = FSDPTrainer(config)
fsdp_model = trainer.wrap_model(model, device_id=0)
```

**FSDPConfig Parameters:**
- `sharding_strategy` (str): 'FULL_SHARD', 'SHARD_GRAD_OP', or 'NO_SHARD'
- `cpu_offload` (bool): Offload parameters to CPU
- `mixed_precision` (str): 'bf16', 'fp16', or 'fp32'
- `activation_checkpointing` (bool): Enable activation checkpointing

### `CheckpointManager`

Manages model checkpoints with top-K retention.

```python
from neuros_neurofm.training import CheckpointManager

manager = CheckpointManager(
    checkpoint_dir='checkpoints/',
    save_top_k=3,
    mode='min',  # 'min' or 'max'
    monitor='val_loss',
)

# Save checkpoint
path = manager.save(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    global_step=1000,
    epoch=5,
    metrics={'val_loss': 0.5, 'val_acc': 0.85},
    data_cursor={'shard': 10, 'sample': 1500},  # For resumption
    is_best=True,
)

# Load checkpoint
checkpoint = manager.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
```

### `CurriculumScheduler`

3-stage curriculum learning scheduler.

```python
from neuros_neurofm.training import CurriculumScheduler

scheduler = CurriculumScheduler(
    stages=[
        {
            'name': 'unimodal',
            'duration_steps': 10000,
            'modality_pairs': None,  # Single modalities only
            'learning_rate_multiplier': 1.0,
        },
        {
            'name': 'pairwise',
            'duration_steps': 20000,
            'modality_pairs': [('eeg', 'spikes'), ('eeg', 'video')],
            'learning_rate_multiplier': 0.5,
        },
        {
            'name': 'multimodal',
            'duration_steps': 50000,
            'modality_pairs': 'all',
            'learning_rate_multiplier': 0.3,
        }
    ],
    transition_steps=1000,
    warmup_steps=1000,
)

# Get current stage
stage = scheduler.get_stage(global_step=15000)
print(stage['name'])  # 'pairwise'

# Apply stage-specific configuration
batch = scheduler.apply_curriculum(batch, global_step)
lr_mult = scheduler.get_lr_multiplier(global_step)
```

---

## Data Pipeline

### `create_webdataset_loader`

Create WebDataset data loader for efficient streaming.

```python
from neuros_neurofm.datasets import create_webdataset_loader

loader = create_webdataset_loader(
    shard_urls='data/train/train-{000000..000099}.tar',
    batch_size=32,
    num_workers=8,
    shuffle_buffer=10000,
    modality_specs={
        'eeg': {'channels': 64, 'sample_rate': 1000},
        'spikes': {'units': 96, 'sample_rate': 30000},
    },
)

for batch in loader:
    # batch is a dict: {'eeg': (32, T, 64), 'spikes': (32, T, 96)}
    output = model(batch)
```

**Parameters:**
- `shard_urls` (str): URL pattern for shards (supports brace expansion)
- `batch_size` (int): Batch size
- `num_workers` (int): Number of worker processes
- `shuffle_buffer` (int): Shuffle buffer size (0 for no shuffling)
- `modality_specs` (dict): Specification for each modality

### `TemporalAligner`

Aligns multi-rate temporal data.

```python
from neuros_neurofm.tokenizers import TemporalAligner

aligner = TemporalAligner(
    target_rate=1000,  # Target sampling rate (Hz)
    interpolation='linear',  # 'nearest', 'linear', 'cubic', 'causal'
    jitter_correction=True,
)

# Align modalities
aligned = aligner.align_modalities(
    modalities={
        'eeg': {'data': eeg_data, 'rate': 1000, 't0': 0.0},
        'spikes': {'data': spike_data, 'rate': 30000, 't0': 0.001},
        'video': {'data': video_data, 'rate': 30, 't0': 0.0},
    }
)

# Result: {'eeg': (T, C), 'spikes': (T, C), 'video': (T, C)}
# All aligned to target_rate=1000 Hz
```

---

## Training Objectives

### `MaskedModelingLoss`

BERT-style masked modeling loss.

```python
from neuros_neurofm.losses import MaskedModelingLoss

loss_fn = MaskedModelingLoss(
    mask_ratio=0.15,
    mask_strategy='random',  # 'random', 'block', 'adaptive'
    block_size=10,           # For 'block' strategy
)

loss = loss_fn(
    tokens=input_tokens,      # (B, T, D)
    reconstructed=output,     # (B, T, D)
    mask=None,               # Optional pre-computed mask
)
```

**Masking Strategies:**
- `random`: Random token masking
- `block`: Contiguous block masking
- `adaptive`: Attention-based adaptive masking

### `MultiHorizonForecastingLoss`

Multi-horizon prediction loss.

```python
from neuros_neurofm.losses import MultiHorizonForecastingLoss

loss_fn = MultiHorizonForecastingLoss(
    horizons_ms=[100, 250, 500, 1000],  # Prediction horizons
    distance_weighting='exponential',    # 'uniform', 'linear', 'exponential'
)

loss = loss_fn(
    current_state=current,    # (B, T, D)
    predictions=pred,         # (B, T, num_horizons, D)
    targets=targets,          # (B, T, num_horizons, D)
)
```

### `DiffusionLoss`

DDPM-style diffusion denoising loss.

```python
from neuros_neurofm.losses import DiffusionLoss

loss_fn = DiffusionLoss(
    num_timesteps=1000,
    noise_schedule='cosine',  # 'linear', 'cosine', 'polynomial'
)

loss = loss_fn(
    clean_data=clean,        # (B, T, D)
    denoised=output,         # (B, T, D)
)
```

### `CombinedLoss`

Combine multiple objectives with adaptive weighting.

```python
from neuros_neurofm.losses import CombinedLoss

combined = CombinedLoss(
    losses={
        'masked_modeling': MaskedModelingLoss(),
        'forecasting': MultiHorizonForecastingLoss(),
        'contrastive': CrossModalContrastiveLoss(),
    },
    weights={
        'masked_modeling': 1.0,
        'forecasting': 0.5,
        'contrastive': 0.3,
    },
    adaptive_weights=True,  # Enable adaptive balancing
)

loss_dict = combined(
    model_output=output,
    targets=targets,
    metadata=batch_metadata,
)

total_loss = loss_dict['total_loss']
individual_losses = loss_dict['losses']  # Dict per objective
```

---

## Mechanistic Interpretability

### Sparse Autoencoders

#### `HierarchicalSAE`

Multi-level sparse autoencoder for concept discovery.

```python
from neuros_neurofm.interpretability import HierarchicalSAE

sae = HierarchicalSAE(
    layer_sizes=[768, 4096, 16384],
    sparsity_coefficients=[0.01, 0.005, 0.001],
)

# Train on activations
sae.train(activations, num_epochs=10, learning_rate=1e-4)

# Encode to sparse codes
sparse_codes = sae.encode(activations)  # (N, 16384)

# Decode back
reconstructed = sae.decode(sparse_codes)  # (N, 768)
```

#### `ConceptDictionary`

Build semantic concept dictionary from SAE features.

```python
from neuros_neurofm.interpretability import ConceptDictionary

concept_dict = ConceptDictionary(sae)

# Build dictionary with probe labels
concept_dict.build_dictionary(
    activations=activations,
    probe_labels={
        'region': region_labels,
        'task': task_labels,
    }
)

# Query concepts
top_concepts = concept_dict.get_top_concepts(
    feature_idx=1024,
    top_k=10,
)

for concept in top_concepts:
    print(f"{concept.name}: {concept.score:.3f}")
```

### Brain Alignment

#### `CCAAlignment`

Canonical Correlation Analysis for brain-model alignment.

```python
from neuros_neurofm.interpretability import CCAAlignment

cca = CCAAlignment(
    n_components=50,
    regularization=0.01,
)

# Compute alignment score
score, dimensions, ci = cca.fit_and_score(
    model_activations=model_acts,  # (N, D_model)
    brain_activations=brain_acts,  # (N, D_brain)
    bootstrap=200,  # Bootstrap iterations for CI
)

print(f"CCA score: {score:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
print(f"Optimal dimensions: {dimensions}")
```

#### `RSAAlignment`

Representational Similarity Analysis.

```python
from neuros_neurofm.interpretability import RSAAlignment

rsa = RSAAlignment(method='spearman')

score, ci = rsa.fit_and_score(
    model_activations=model_acts,
    brain_activations=brain_acts,
    bootstrap=200,
)

print(f"RSA score: {score:.3f} [{ci[0]:.3f}, {ci[1]:.3f}]")
```

### Dynamical Systems

#### `KoopmanOperator`

Koopman operator for linear representation of nonlinear dynamics.

```python
from neuros_neurofm.interpretability import KoopmanOperator

koopman = KoopmanOperator(
    state_dim=768,
    window_size=128,
)

# Fit on trajectories
koopman.fit(trajectories)  # (N_traj, T, D)

# Analyze stability
eigenvalues = koopman.eigenvalues
stable_modes = (np.abs(eigenvalues) < 1).sum()
unstable_modes = (np.abs(eigenvalues) >= 1).sum()

# Predict future states
future = koopman.predict(current_state, steps=100)
```

#### `LyapunovAnalyzer`

Compute Lyapunov exponents for chaos detection.

```python
from neuros_neurofm.interpretability import DynamicsAnalyzer

analyzer = DynamicsAnalyzer(dim=768, dt=0.001)

lyapunov_exponents = analyzer.compute_lyapunov_exponents(
    trajectories=trajectories,  # (N, T, D)
    steps=1000,
)

max_lyapunov = lyapunov_exponents.max()
if max_lyapunov > 0:
    print("Chaotic dynamics detected!")
```

### Counterfactuals

#### `LatentSurgery`

Perform targeted edits in latent space.

```python
from neuros_neurofm.interpretability import LatentSurgery

surgery = LatentSurgery(model)

# Define edit function
def amplify(latent):
    return latent * 1.5

# Apply intervention
intervened_output = surgery.edit_latent(
    input_data=batch,
    layer_name='backbone.layers.6',
    edit_fn=amplify,
)

# Measure effect
effect = (intervened_output - baseline_output).abs().mean()
```

#### `DoCalculusEngine`

Causal interventions with do-calculus.

```python
from neuros_neurofm.interpretability import DoCalculusEngine

engine = DoCalculusEngine(model)

# Estimate causal effect: P(Y | do(X=x))
causal_effect = engine.intervene(
    intervention={'layer_6': intervention_value},
    outcome_layer='layer_12',
    data=validation_data,
)
```

### Topology

#### `TopologyAnalyzer`

Persistent homology and Betti numbers.

```python
from neuros_neurofm.interpretability import TopologyAnalyzer

analyzer = TopologyAnalyzer(max_dimension=2)

# Compute persistent homology
persistence = analyzer.compute_persistence(activations)

# Extract Betti numbers
betti = analyzer.compute_betti_numbers(persistence)
print(f"β₀ (connected components): {betti[0]}")
print(f"β₁ (loops): {betti[1]}")
print(f"β₂ (voids): {betti[2]}")
```

### Reporting

#### `MechIntReporter`

Generate comprehensive HTML reports.

```python
from neuros_neurofm.interpretability import MechIntReporter

reporter = MechIntReporter(
    output_dir='reports/',
    format='html',  # or 'markdown'
)

# Add analysis sections
reporter.add_section(
    name='SAE Analysis',
    results=sae_results,
)

reporter.add_section(
    name='Brain Alignment',
    results=alignment_results,
)

# Generate report
report_path = reporter.generate_report(
    title="NeuroFMX Mechanistic Interpretability",
    description="Complete analysis of learned representations",
)

print(f"Report generated: {report_path}")
```

---

## Evaluation

### `TaskRegistry`

Register and manage evaluation tasks.

```python
from neuros_neurofm.evaluation import TaskRegistry

registry = TaskRegistry()

# Register task
registry.register_task(
    name='motor_decoding',
    task_type='regression',
    input_modalities=['spikes', 'lfp'],
    output_dim=2,  # (x, y) cursor position
    metric='r2',
    description='Decode cursor position from motor cortex',
)

# Get task
task = registry.get_task('motor_decoding')
```

### `ZeroShotEvaluator`

Zero-shot evaluation with frozen features.

```python
from neuros_neurofm.evaluation import ZeroShotEvaluator

evaluator = ZeroShotEvaluator(
    model=model,
    freeze_backbone=True,
)

# Extract features
train_features, train_labels = evaluator.extract_features(
    data_loader=train_loader,
    task_config=task_config,
)

# Train linear probe
probe = evaluator.train_probe(
    features=train_features,
    labels=train_labels,
    task_type='regression',
    output_dim=2,
)

# Evaluate
metrics = evaluator.evaluate(
    probe=probe,
    features=test_features,
    labels=test_labels,
    task_type='regression',
    metric='r2',
)

print(f"R² score: {metrics['r2']:.4f}")
```

### `FewShotEvaluator`

Few-shot learning with LoRA.

```python
from neuros_neurofm.evaluation import FewShotEvaluator

evaluator = FewShotEvaluator(
    model=model,
    lora_rank=8,
    lora_alpha=16,
)

# K-shot learning
for k in [1, 5, 10, 25, 50]:
    # Create K-shot loader
    k_shot_loader = evaluator.create_k_shot_loader(
        data_loader=train_loader,
        k=k,
        task_type='regression',
    )

    # Fine-tune with LoRA
    evaluator.fine_tune(
        train_loader=k_shot_loader,
        task_config=task_config,
        num_epochs=10,
        learning_rate=1e-4,
    )

    # Evaluate
    metrics = evaluator.evaluate(
        test_loader=test_loader,
        task_config=task_config,
    )

    print(f"{k}-shot R²: {metrics['r2']:.4f}")
```

---

## Deployment

### Model Export

#### TorchScript

```python
import torch

# Trace model
model.eval()
example_inputs = {'eeg': torch.randn(1, 100, 64)}

with torch.no_grad():
    traced_model = torch.jit.trace(model, example_inputs)

# Save
traced_model.save('neurofmx.pt')

# Load and use
loaded = torch.jit.load('neurofmx.pt')
output = loaded(example_inputs)
```

#### ONNX

```python
import torch

# Export to ONNX
torch.onnx.export(
    model,
    example_inputs,
    'neurofmx.onnx',
    opset_version=14,
    input_names=['eeg', 'spikes'],
    output_names=['output'],
    dynamic_axes={
        'eeg': {0: 'batch', 1: 'time'},
        'output': {0: 'batch', 1: 'time'},
    }
)

# Use with ONNX Runtime
import onnxruntime as ort

session = ort.InferenceSession('neurofmx.onnx')
ort_inputs = {k: v.numpy() for k, v in example_inputs.items()}
output = session.run(None, ort_inputs)
```

### Optimization

```python
# Dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)

# Pruning
from torch.nn.utils import prune

for name, module in model.named_modules():
    if isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.3)
```

### FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class InferenceRequest(BaseModel):
    modalities: dict
    return_embeddings: bool = False

@app.post("/predict")
async def predict(request: InferenceRequest):
    inputs = {k: torch.tensor(v) for k, v in request.modalities.items()}

    with torch.no_grad():
        output = model(inputs, return_embeddings=request.return_embeddings)

    return {
        'predictions': output.tolist(),
        'timestamp': datetime.now().isoformat()
    }
```

---

## Full Example: Training to Deployment

```python
# 1. Create model
from neuros_neurofm.model import NeuroFMX

model = NeuroFMX(d_model=768, n_layers=12, architecture='mamba')

# 2. Setup distributed training
from neuros_neurofm.training import FSDPTrainer, FSDPConfig

config = FSDPConfig(sharding_strategy='FULL_SHARD', mixed_precision='bf16')
trainer = FSDPTrainer(config)
fsdp_model = trainer.wrap_model(model)

# 3. Create data loader
from neuros_neurofm.datasets import create_webdataset_loader

loader = create_webdataset_loader(
    shard_urls='data/train-{000000..000099}.tar',
    batch_size=32,
    modality_specs={'eeg': {'channels': 64, 'sample_rate': 1000}}
)

# 4. Setup losses
from neuros_neurofm.losses import CombinedLoss, MaskedModelingLoss

loss_fn = CombinedLoss(
    losses={'masked': MaskedModelingLoss()},
    weights={'masked': 1.0}
)

# 5. Train
optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

for batch in loader:
    output = fsdp_model(batch)
    loss = loss_fn(output, batch)
    loss.backward()
    optimizer.step()

# 6. Evaluate
from neuros_neurofm.evaluation import ZeroShotEvaluator

evaluator = ZeroShotEvaluator(model)
metrics = evaluator.evaluate(test_loader, task_config)

# 7. Interpret
from neuros_neurofm.interpretability import HierarchicalSAE, MechIntReporter

sae = HierarchicalSAE(layer_sizes=[768, 4096])
sae.train(activations)

reporter = MechIntReporter()
reporter.generate_report(all_results)

# 8. Deploy
traced_model = torch.jit.trace(model, example_inputs)
traced_model.save('neurofmx.pt')
```

---

## Configuration Files

All components support YAML configuration:

```yaml
# config.yaml
model:
  d_model: 768
  n_layers: 12
  architecture: mamba

training:
  batch_size: 32
  learning_rate: 1e-4
  max_steps: 100000

distributed:
  sharding_strategy: FULL_SHARD
  mixed_precision: bf16

mechint:
  sae:
    enabled: true
    layer_sizes: [768, 4096, 16384]
  alignment:
    enabled: true
    method: CCA
```

Load configuration:

```python
import yaml

with open('config.yaml') as f:
    config = yaml.safe_load(f)

model = NeuroFMX(**config['model'])
```

---

## See Also

- [Examples](../examples/README.md) - Complete usage examples
- [Training Guide](TRAINING_GUIDE.md) - Detailed training instructions
- [Interpretability Guide](mechint_hooks_guide.md) - Mech-int workflows
- [Deployment Guide](DEPLOYMENT_GUIDE.md) - Production deployment

---

**Last Updated:** January 2025
**Version:** 1.0.0
