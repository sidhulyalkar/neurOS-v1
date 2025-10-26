# Ray Tune Hyperparameter Search for NeuroFMX

This guide explains how to use Ray Tune for large-scale hyperparameter optimization of NeuroFMX models.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Search Algorithms](#search-algorithms)
4. [Search Spaces](#search-spaces)
5. [Distributed Training](#distributed-training)
6. [Experiment Tracking](#experiment-tracking)
7. [Advanced Usage](#advanced-usage)
8. [Best Practices](#best-practices)

## Installation

Install Ray Tune with all dependencies:

```bash
pip install 'ray[tune]' 'ray[train]' optuna
```

Optional: Install experiment tracking tools:

```bash
pip install mlflow wandb
```

## Quick Start

### Basic Example

```python
from neuros_neurofm.optimization import (
    NeuroFMXRayTuner,
    NeuroFMXSearchSpace,
    create_neurofmx_train_fn,
)

# 1. Create training function
train_fn = create_neurofmx_train_fn(
    data_config={'data_path': './data'},
    base_config={'checkpoint_dir': './checkpoints'},
    num_epochs=20,
)

# 2. Create tuner
tuner = NeuroFMXRayTuner(
    train_fn=train_fn,
    search_algorithm='asha',
    num_samples=100,
    resources_per_trial={'cpu': 4, 'gpu': 1},
    metric='val_loss',
    mode='min',
)

# 3. Run optimization
results = tuner.run()

# 4. Export best configuration
tuner.export_best_config('best_config.yaml')
```

### Command-Line Usage

```bash
# ASHA scheduler with 50 trials
python examples/ray_tune_example.py \
    --search_algorithm asha \
    --num_samples 50 \
    --max_concurrent 4 \
    --gpus_per_trial 1

# Population-Based Training
python examples/ray_tune_example.py \
    --search_algorithm pbt \
    --num_samples 20 \
    --epochs_per_trial 50

# Bayesian Optimization with small search space
python examples/ray_tune_example.py \
    --search_algorithm bayesian \
    --search_space small \
    --num_samples 30
```

## Search Algorithms

### 1. ASHA (Asynchronous Successive Halving Algorithm)

**Best for:** Quick exploration with early stopping.

**Features:**
- Aggressively stops underperforming trials
- Resource-efficient
- Good for large search spaces

**Configuration:**
```python
tuner = NeuroFMXRayTuner(
    search_algorithm='asha',
    grace_period=10,        # Min epochs before stopping
    reduction_factor=3,      # Aggressive = 2, Conservative = 4
    max_epochs=100,
)
```

**When to use:**
- Limited compute budget
- Large search space (100+ trials)
- Want to try many configurations quickly

### 2. Population-Based Training (PBT)

**Best for:** Online hyperparameter adaptation.

**Features:**
- Dynamically adjusts hyperparameters during training
- Exploits good performers, explores via perturbations
- Great for learning rate schedules

**Configuration:**
```python
tuner = NeuroFMXRayTuner(
    search_algorithm='pbt',
    num_samples=20,          # Population size
    max_epochs=100,
)
```

**When to use:**
- Training long models (50+ epochs)
- Hyperparameters affect training dynamics
- Want to find schedules, not just fixed values

### 3. Bayesian Optimization (OptunaSearch)

**Best for:** Sample-efficient optimization.

**Features:**
- Models the objective function
- Focuses on promising regions
- Good for expensive trials

**Configuration:**
```python
tuner = NeuroFMXRayTuner(
    search_algorithm='bayesian',
    num_samples=30,          # Fewer samples needed
    max_concurrent_trials=4, # Sequential is more efficient
)
```

**When to use:**
- Expensive trials (long training times)
- Small-medium search spaces (< 50 trials)
- Want to maximize performance per trial

### 4. Random Search

**Best for:** Baseline comparisons.

**Configuration:**
```python
tuner = NeuroFMXRayTuner(
    search_algorithm='random',
    num_samples=100,
)
```

## Search Spaces

### Default Search Space

Comprehensive search over all NeuroFMX hyperparameters:

```python
search_space = NeuroFMXSearchSpace.get_default_search_space()
```

**Includes:**
- Model architecture: d_model, n_layers, latent_dim, etc.
- Training: lr, batch_size, weight_decay
- Multi-rate: rates, fusion methods
- Losses: mask_ratio, contrastive weights

**Use when:** Unsure which hyperparameters matter most.

### Small Search Space

Focused on most impactful hyperparameters:

```python
search_space = NeuroFMXSearchSpace.get_small_search_space()
```

**Includes:**
- d_model: [256, 512, 768]
- n_layers: [8, 12, 16]
- lr: log-uniform [1e-4, 5e-4]
- batch_size: [32, 64]

**Use when:** Quick experiments, limited compute.

### Architecture Search Space

Neural architecture search (NAS):

```python
search_space = NeuroFMXSearchSpace.get_architecture_search_space()
```

**Includes:**
- Wide range of architectural choices
- Depth, width, latent dimensions
- Multi-rate configurations

**Use when:** Designing model architecture.

### Custom Search Space

Define your own:

```python
from ray import tune

custom_space = {
    'model': {
        'd_model': tune.choice([512, 1024, 2048]),
        'n_layers': tune.randint(8, 24),
        'dropout': tune.uniform(0.1, 0.3),
    },
    'training': {
        'lr': tune.loguniform(1e-5, 1e-3),
        'batch_size': tune.choice([32, 64, 128]),
    },
}

tuner = NeuroFMXRayTuner(
    search_space=custom_space,
    ...
)
```

## Distributed Training

### Multi-GPU on Single Node

```python
tuner = NeuroFMXRayTuner(
    resources_per_trial={'cpu': 8, 'gpu': 1},
    max_concurrent_trials=4,  # Run 4 trials in parallel
)
```

### Multi-Node Cluster

1. Start Ray cluster:

```bash
# Head node
ray start --head --port=6379

# Worker nodes
ray start --address='head_node_ip:6379'
```

2. Connect and run:

```python
import ray
ray.init(address='auto')  # Connect to cluster

tuner = NeuroFMXRayTuner(...)
results = tuner.run()
```

### Fractional GPUs

Share GPUs across trials:

```python
tuner = NeuroFMXRayTuner(
    resources_per_trial={'gpu': 0.5},  # 2 trials per GPU
    max_concurrent_trials=8,
)
```

## Experiment Tracking

### MLflow Integration

```python
tuner = NeuroFMXRayTuner(
    use_mlflow=True,
    experiment_name='neurofmx_tuning',
)
```

View results:
```bash
mlflow ui --backend-store-uri ./mlruns
```

### Weights & Biases Integration

```python
tuner = NeuroFMXRayTuner(
    use_wandb=True,
    wandb_project='neurofmx',
    experiment_name='asha_optimization',
)
```

### Both MLflow and W&B

```python
tuner = NeuroFMXRayTuner(
    use_mlflow=True,
    use_wandb=True,
    wandb_project='neurofmx',
)
```

## Advanced Usage

### Custom Training Function

Implement your own training loop:

```python
from ray.air import session

def my_train_fn(config):
    """Custom training function."""
    # Setup
    model = create_model(config)
    train_loader = create_dataloader(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'])

    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer)

        # Validate
        val_loss, metrics = validate(model, val_loader)

        # Report to Ray Tune (REQUIRED)
        session.report({
            'training_iteration': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            **metrics,
        })

tuner = NeuroFMXRayTuner(train_fn=my_train_fn)
```

### Checkpoint Management

```python
tuner = NeuroFMXRayTuner(
    checkpoint_freq=10,           # Save every 10 epochs
    keep_checkpoints_num=3,       # Keep best 3 checkpoints
)

# Get best checkpoint
best_checkpoint = tuner.get_best_checkpoint_path()
```

### Resume from Previous Run

```python
tuner = NeuroFMXRayTuner(
    resume=True,
    local_dir='./ray_results',
    experiment_name='neurofmx_tuning',
)
```

### Analyze Results

```python
# Get results dataframe
df = tuner.results.get_dataframe()

# Filter and sort
best_trials = df.nsmallest(10, 'val_loss')

# Plot
tuner.plot_optimization_history('results.png')
```

## Best Practices

### 1. Start Small, Scale Up

```python
# Phase 1: Quick exploration (2 hours)
tuner_phase1 = NeuroFMXRayTuner(
    search_space=NeuroFMXSearchSpace.get_small_search_space(),
    search_algorithm='asha',
    num_samples=30,
    grace_period=3,
    max_epochs=10,
)

# Phase 2: Refine promising regions (8 hours)
tuner_phase2 = NeuroFMXRayTuner(
    search_space=refined_space,  # Based on phase 1 results
    search_algorithm='bayesian',
    num_samples=20,
    max_epochs=50,
)
```

### 2. Monitor Resource Usage

```bash
# Monitor Ray dashboard
ray dashboard
```

### 3. Use Appropriate Schedulers

| Scenario | Recommended Scheduler |
|----------|----------------------|
| Quick exploration | ASHA |
| Long training runs | PBT |
| Sample efficiency | Bayesian |
| Baseline | Random |

### 4. Tune in Stages

**Stage 1:** Architecture search
```python
search_space = NeuroFMXSearchSpace.get_architecture_search_space()
```

**Stage 2:** Training hyperparameters
```python
search_space = {
    'training': {...},
    'optimizer': {...},
}
```

**Stage 3:** Loss weights and regularization
```python
search_space = {
    'losses': {...},
    'model.dropout': tune.uniform(0.1, 0.3),
}
```

### 5. Set Realistic Budgets

| Budget | Strategy |
|--------|----------|
| 4 GPU-hours | Small space + ASHA + 20 trials |
| 20 GPU-hours | Default space + ASHA + 50 trials |
| 100 GPU-hours | Default space + PBT + 20 trials x 50 epochs |

### 6. Log Everything

```python
def train_fn(config):
    for epoch in range(num_epochs):
        # ... training ...

        session.report({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'val_r2': r2_score,
            'lr': current_lr,
            'gpu_memory_mb': torch.cuda.max_memory_allocated() / 1e6,
            'batch_time_ms': batch_time * 1000,
        })
```

## Troubleshooting

### Out of Memory

```python
# Reduce batch size or model size
search_space = {
    'model.d_model': tune.choice([256, 512]),  # Smaller models
    'training.batch_size': tune.choice([16, 32]),
}
```

### Slow Trials

```python
# Increase grace period to avoid wasting time on bad configs
tuner = NeuroFMXRayTuner(
    grace_period=3,  # Stop after 3 epochs if clearly bad
)
```

### Too Many Concurrent Trials

```python
# Limit concurrency
tuner = NeuroFMXRayTuner(
    max_concurrent_trials=2,  # Fewer parallel trials
)
```

## Examples

### Example 1: Quick Architecture Search

```python
from neuros_neurofm.optimization import NeuroFMXRayTuner, NeuroFMXSearchSpace

search_space = NeuroFMXSearchSpace.get_architecture_search_space()

tuner = NeuroFMXRayTuner(
    train_fn=my_train_fn,
    search_algorithm='asha',
    search_space=search_space,
    num_samples=50,
    max_concurrent_trials=8,
    resources_per_trial={'gpu': 0.5},
    max_epochs=20,
    grace_period=5,
)

results = tuner.run()
tuner.export_best_config('best_architecture.yaml')
```

### Example 2: Learning Rate Optimization with PBT

```python
search_space = {
    'training': {
        'lr': tune.loguniform(1e-5, 1e-3),
        'weight_decay': tune.loguniform(1e-6, 1e-2),
    },
}

tuner = NeuroFMXRayTuner(
    train_fn=my_train_fn,
    search_algorithm='pbt',
    search_space=search_space,
    num_samples=20,
    max_epochs=100,
)

results = tuner.run()
```

### Example 3: Full Pipeline with Tracking

```python
tuner = NeuroFMXRayTuner(
    train_fn=my_train_fn,
    search_algorithm='bayesian',
    num_samples=30,
    resources_per_trial={'cpu': 8, 'gpu': 1},
    metric='val_r2',
    mode='max',  # Maximize R²
    use_mlflow=True,
    use_wandb=True,
    wandb_project='neurofmx-optimization',
    experiment_name='full_pipeline',
)

results = tuner.run()
tuner.export_best_config('best_config.yaml')
tuner.plot_optimization_history('optimization.png')
```

## References

- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)
- [ASHA Paper](https://arxiv.org/abs/1810.05934)
- [PBT Paper](https://arxiv.org/abs/1711.09846)
- [NeuroFMX Documentation](../README.md)

## Support

For issues or questions:
- GitHub Issues: [neurOS Issues](https://github.com/your-repo/neuros/issues)
- Ray Community: [Ray Discourse](https://discuss.ray.io/)
