# NeuroFMX Evaluation System Guide

## Overview

The NeuroFMX evaluation system provides a comprehensive framework for benchmarking foundation models across diverse neural data tasks. It includes:

1. **Task Registry**: Centralized management of evaluation tasks with rich metadata
2. **Zero-Shot Evaluation**: Test frozen representations with linear probes
3. **Few-Shot Evaluation**: Measure adaptation with limited labeled data (K=1,5,10,25,50)
4. **Comprehensive Task Suite**: 18+ tasks spanning multiple species, modalities, and domains

## Architecture

```
neuros_neurofm/evaluation/
├── task_registry.py     # Task registration and discovery
├── zero_shot.py         # Zero-shot evaluation with linear probes
├── few_shot_eval.py     # Few-shot learning with LoRA adapters
├── metrics.py           # Evaluation metrics (R², correlation, BPS)
├── falcon.py            # FALCON benchmark integration
└── visualization.py     # Plotting and visualization tools

configs/eval/
└── eval_tasks.yaml      # Task definitions and metadata
```

## Quick Start

### 1. Load Task Registry

```python
from neuros_neurofm.evaluation import TaskRegistry

# Load from YAML config
registry = TaskRegistry(config_path="configs/eval/eval_tasks.yaml")

# List all tasks
print(f"Total tasks: {len(registry)}")

# Filter tasks
motor_tasks = registry.list_tasks(tags=["motor", "bci"])
monkey_tasks = registry.list_tasks(species=Species.MONKEY)
encoding_tasks = registry.list_tasks(task_type=TaskType.ENCODING)
```

### 2. Run Zero-Shot Evaluation

```python
from neuros_neurofm.evaluation import (
    ZeroShotEvaluator,
    ZeroShotConfig,
)

# Configure evaluation
config = ZeroShotConfig(
    probe_lr=1e-3,
    probe_epochs=100,
    early_stopping_patience=10,
    device="cuda",
)

# Create evaluator
evaluator = ZeroShotEvaluator(model, config)

# Evaluate on a task
task = registry.get("monkey_m1_reaching_decoding")
results = evaluator.run_zero_shot_evaluation(
    task,
    output_dir="results/zero_shot"
)

print(f"Test R²: {results['test_metric']:.4f}")
```

### 3. Run Few-Shot Evaluation

```python
from neuros_neurofm.evaluation import (
    FewShotEvaluator,
    FewShotConfig,
)

# Configure few-shot learning
config = FewShotConfig(
    k_shots=[1, 5, 10, 25, 50],
    n_episodes=100,
    adaptation_method="lora",
    lora_rank=8,
    adaptation_steps=100,
)

# Create evaluator
evaluator = FewShotEvaluator(model, config)

# Evaluate
results = evaluator.run_few_shot_evaluation(
    task,
    output_dir="results/few_shot"
)

# Print learning curve
for k in [1, 5, 10, 25, 50]:
    k_results = results['k_shot_results'][k]
    metric = k_results['accuracy']
    print(f"K={k}: {metric['mean']:.4f} ± {metric['std']:.4f}")
```

### 4. Run Full Evaluation Suite

```bash
# Run comprehensive evaluation
python examples/run_comprehensive_evaluation.py \
    --model_path checkpoints/neurofmx_pretrained.pt \
    --config configs/eval/eval_tasks.yaml \
    --output_dir results/evaluation/ \
    --zero_shot \
    --few_shot \
    --k_shots 1 5 10 25 50
```

## Task Registry

### Task Metadata

Each task includes rich metadata:

- **name**: Unique task identifier
- **task_type**: Classification, regression, encoding, forecasting
- **species**: Mouse, monkey, human, rat, zebrafish
- **modality**: Spikes, LFP, EEG, ECoG, fMRI, calcium
- **target**: What to predict (kinematics, sleep stage, etc.)
- **metric**: Evaluation metrics (R², accuracy, bits-per-spike)
- **difficulty**: Easy, medium, hard
- **tags**: Additional categorization

### Adding Custom Tasks

```python
from neuros_neurofm.evaluation import (
    TaskMetadata,
    EvaluationTask,
    TaskType,
    Species,
    Modality,
)

# Define metadata
metadata = TaskMetadata(
    name="custom_decoding_task",
    task_type=TaskType.REGRESSION,
    species=Species.MONKEY,
    modality=Modality.SPIKES,
    target="hand velocity",
    metric=["r2_score", "correlation"],
    output_dim=3,
    difficulty="medium",
)

# Create task
task = EvaluationTask(
    metadata=metadata,
    dataset=your_dataset,
)

# Split dataset
task.split_dataset(
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
)

# Register
registry.register(task)
```

## Evaluation Tasks

### Neural Decoding (8 tasks)

| Task | Species | Modality | Target | Difficulty |
|------|---------|----------|--------|------------|
| monkey_m1_reaching_decoding | Monkey | Spikes | Arm kinematics | Medium |
| monkey_pmd_grasp_classification | Monkey | Spikes | Grasp type | Medium |
| human_eeg_sleep_staging | Human | EEG | Sleep stage | Hard |
| human_ecog_speech_decoding | Human | ECoG | Speech acoustics | Hard |
| mouse_ca1_position_decoding | Mouse | Calcium | 2D position | Easy |
| human_eeg_bci_motor_imagery | Human | EEG | Motor imagery | Medium |
| human_eeg_epilepsy_detection | Human | EEG | Seizure presence | Medium |
| rat_hippocampus_memory_task | Rat | Spikes | Task choice | Easy |

### Neural Encoding (3 tasks)

| Task | Species | Modality | Target | Difficulty |
|------|---------|----------|--------|------------|
| mouse_v1_natural_images_encoding | Mouse | Spikes | Firing rates | Medium |
| monkey_it_face_encoding | Monkey | Spikes | Firing rates | Hard |
| human_fmri_narrative_encoding | Human | fMRI | Voxel responses | Hard |

### Forecasting (3 tasks)

| Task | Species | Modality | Target | Difficulty |
|------|---------|----------|--------|------------|
| monkey_m1_dynamics_forecast | Monkey | Spikes | Future rates | Medium |
| mouse_hippocampus_sequence_prediction | Mouse | Spikes | Sequences | Hard |
| monkey_pfc_working_memory_dynamics | Monkey | Spikes | Population state | Hard |

### Cross-Species Transfer (2 tasks)

| Task | Species | Modality | Target | Difficulty |
|------|---------|----------|--------|------------|
| cross_species_motor_transfer | Multi | Spikes | Movement velocity | Hard |
| cross_species_visual_encoding | Multi | Spikes | Firing rates | Hard |

### Advanced Tasks (2 tasks)

| Task | Species | Modality | Target | Difficulty |
|------|---------|----------|--------|------------|
| zebrafish_wholebrain_behavior_mapping | Zebrafish | Calcium | Swim vigor | Hard |
| monkey_pfc_working_memory_dynamics | Monkey | Spikes | Population state | Hard |

## Zero-Shot Evaluation

### How It Works

1. **Extract Representations**: Pass neural data through frozen NeuroFMX encoder
2. **Train Linear Probe**: Train linear classifier/regressor on representations
3. **Evaluate**: Test on held-out test set
4. **Layer Analysis**: Optionally probe different layers to find best features

### Configuration Options

```python
ZeroShotConfig(
    layers_to_probe=[0, 4, 8, 12, 16],  # Which layers to evaluate
    probe_lr=1e-3,                       # Linear probe learning rate
    probe_epochs=100,                     # Training epochs
    probe_batch_size=128,                 # Batch size
    l2_regularization=1e-4,               # L2 penalty
    early_stopping_patience=10,           # Early stopping
    use_layer_norm=True,                  # Normalize representations
    cache_representations=True,           # Cache extracted features
)
```

### Metrics

- **Regression**: R², Pearson correlation, MAE, RMSE
- **Classification**: Accuracy, F1-score, AUC-ROC
- **Encoding**: Bits-per-spike, correlation
- **Forecasting**: MAE, RMSE, per-timestep metrics

## Few-Shot Evaluation

### How It Works

1. **Episode Sampling**: Sample K support examples + N query examples
2. **Model Adaptation**: Fine-tune model on support set
3. **Evaluation**: Test on query set
4. **Repeat**: Run multiple episodes for statistical significance
5. **Confidence Intervals**: Bootstrap CI for robustness

### Adaptation Methods

#### 1. Linear Probe
```python
FewShotConfig(
    adaptation_method="linear_probe",
    freeze_backbone=True,
)
```
Only train final layer, freeze all other parameters.

#### 2. LoRA (Recommended)
```python
FewShotConfig(
    adaptation_method="lora",
    lora_rank=8,
    lora_alpha=16.0,
)
```
Low-rank adaptation: add trainable low-rank matrices to attention layers.

#### 3. Full Fine-Tuning
```python
FewShotConfig(
    adaptation_method="full_finetune",
    freeze_backbone=False,
)
```
Train all model parameters (expensive, risk of overfitting).

### Learning Curves

Few-shot evaluation generates learning curves showing performance vs. number of examples:

```
K=1:  0.45 ± 0.12  (few examples, high variance)
K=5:  0.62 ± 0.08
K=10: 0.71 ± 0.05
K=25: 0.78 ± 0.03
K=50: 0.82 ± 0.02  (more examples, lower variance)
```

### Configuration Options

```python
FewShotConfig(
    k_shots=[1, 5, 10, 25, 50],      # Shot counts to evaluate
    n_episodes=100,                   # Episodes per k-shot
    n_query_samples=100,              # Query set size
    adaptation_steps=100,             # Adaptation iterations
    adaptation_lr=1e-4,               # Adaptation learning rate
    bootstrap_samples=1000,           # For confidence intervals
)
```

## Output and Reports

### Directory Structure

```
results/evaluation/
├── zero_shot/
│   ├── monkey_m1_reaching_decoding_zero_shot_results.json
│   ├── monkey_m1_reaching_decoding_report.md
│   ├── ...
├── few_shot/
│   ├── monkey_m1_reaching_decoding_few_shot_results.json
│   ├── monkey_m1_reaching_decoding_learning_curves.md
│   ├── ...
└── comprehensive_evaluation_report.md
```

### JSON Results Format

```json
{
  "task_name": "monkey_m1_reaching_decoding",
  "task_type": "decoding",
  "species": "monkey",
  "modality": "spikes",
  "test_metric": 0.8234,
  "metric_name": "r2_score",
  "representation_dim": 768,
  "n_train_samples": 5000,
  "n_test_samples": 1000
}
```

### Markdown Reports

Automatically generated reports include:

- Task information and metadata
- Performance metrics with confidence intervals
- Learning curves (for few-shot)
- Comparison tables
- Summary statistics

## Integration with MLflow/W&B

### MLflow Integration

```python
import mlflow

mlflow.start_run()

# Log evaluation results
for task_name, result in zero_shot_results.items():
    mlflow.log_metric(f"zero_shot/{task_name}/r2", result['test_metric'])

# Log few-shot learning curves
for k in [1, 5, 10, 25, 50]:
    mlflow.log_metric(f"few_shot/{task_name}/k{k}", results[k]['mean'])

mlflow.end_run()
```

### Weights & Biases Integration

```python
import wandb

wandb.init(project="neurofmx-evaluation")

# Log results
for task_name, result in results.items():
    wandb.log({
        f"zero_shot/{task_name}": result['test_metric']
    })

# Log learning curve
wandb.log({
    "few_shot_learning_curve": wandb.plot.line_series(
        xs=[[1, 5, 10, 25, 50]],
        ys=[k_shot_values],
        keys=["accuracy"],
        title="Few-Shot Learning Curve",
    )
})
```

## Distributed Evaluation

For large-scale evaluation across multiple GPUs/nodes:

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize distributed
dist.init_process_group(backend="nccl")

# Wrap model
model = DDP(model, device_ids=[local_rank])

# Distribute tasks across ranks
rank = dist.get_rank()
world_size = dist.get_world_size()

tasks_per_rank = len(all_tasks) // world_size
my_tasks = all_tasks[rank * tasks_per_rank : (rank + 1) * tasks_per_rank]

# Evaluate assigned tasks
for task in my_tasks:
    evaluator.run_zero_shot_evaluation(task)

# Gather results
dist.barrier()
```

## Best Practices

### 1. Start with Zero-Shot
Always run zero-shot first to establish baseline performance.

### 2. Use Multiple Episodes
For few-shot, use ≥100 episodes for reliable statistics.

### 3. Bootstrap Confidence Intervals
Report mean ± CI rather than just mean ± std.

### 4. Compare Across Methods
Evaluate both LoRA and linear probe for few-shot.

### 5. Cross-Species Testing
Test on cross-species tasks to measure generalization.

### 6. Layer-wise Probing
For zero-shot, probe multiple layers to find best representations.

### 7. Cache Representations
Enable caching to speed up repeated evaluations.

## Troubleshooting

### Out of Memory
```python
# Reduce batch size
config.probe_batch_size = 64

# Enable gradient checkpointing
model.gradient_checkpointing_enable()

# Use CPU offloading for few-shot
config.device = "cpu"
```

### Slow Evaluation
```python
# Cache representations
config.cache_representations = True

# Reduce episodes
config.n_episodes = 50

# Use smaller K values
config.k_shots = [1, 5, 10]
```

### Poor Performance
```python
# Increase adaptation steps
config.adaptation_steps = 200

# Increase LoRA rank
config.lora_rank = 16

# Try different learning rate
config.adaptation_lr = 1e-3
```

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{neurofmx_evaluation,
  title={NeuroFMX Evaluation Framework},
  author={NeuroFMX Team},
  year={2025},
  url={https://github.com/your-repo/neuros}
}
```

## References

1. **Zero-Shot Learning**: "Evaluation of Neural Architectures Trained with Square Loss vs Cross-Entropy" (Beyer et al., 2020)
2. **Few-Shot Learning**: "Model-Agnostic Meta-Learning" (Finn et al., 2017)
3. **LoRA**: "Low-Rank Adaptation of Large Language Models" (Hu et al., 2021)
4. **FALCON Benchmark**: "A Large-Scale Neural Benchmark for Neural Population Dynamics" (Ye & Pandarinath, 2021)
5. **Neural Encoding Models**: "A Continuous Semantic Space Describes the Representation of Thousands of Object and Action Categories" (Huth et al., 2016)
