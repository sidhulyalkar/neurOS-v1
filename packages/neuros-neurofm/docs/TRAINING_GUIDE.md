# NeuroFMX Training Guide

Complete guide to training NeuroFMX foundation models from scratch.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Data Preparation](#data-preparation)
3. [Model Configuration](#model-configuration)
4. [Training Strategies](#training-strategies)
5. [Distributed Training](#distributed-training)
6. [Monitoring & Debugging](#monitoring--debugging)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Minimal Training Script

See the complete example at: [examples/01_complete_training_workflow.py](../examples/01_complete_training_workflow.py)

Key steps:
1. Create model with NeuroFMX
2. Setup WebDataset data loader
3. Configure training objectives
4. Train with curriculum learning
5. Monitor with MLflow/W&B

---

## Data Preparation

### WebDataset Format

NeuroFMX uses WebDataset for efficient streaming. See [WEBDATASET_GUIDE.md](WEBDATASET_GUIDE.md) for details.

Quick example:
```python
from neuros_neurofm.datasets import WebDatasetWriter

writer = WebDatasetWriter(output_dir='data/train', shard_name='train')
writer.write_sample(key='sample000000', modalities={'eeg': eeg_data})
writer.close()
```

---

## Model Configuration

### Model Sizes

- **Small** (125M): `d_model=512, n_layers=8`
- **Base** (350M): `d_model=768, n_layers=12`  
- **Large** (1B): `d_model=1024, n_layers=24`
- **Extra Large** (3B): `d_model=1536, n_layers=32`

See [API_REFERENCE.md](API_REFERENCE.md) for complete configuration options.

---

## Training Strategies

### 1. Curriculum Learning

Progressive training from simple to complex:
- Stage 1: Unimodal learning
- Stage 2: Pairwise cross-modal
- Stage 3: Full multimodal

See [examples/01_complete_training_workflow.py](../examples/01_complete_training_workflow.py)

### 2. Multi-Objective Training

Combine multiple objectives:
- Masked modeling
- Multi-horizon forecasting
- Diffusion denoising
- Cross-modal contrastive

### 3. Data Augmentation

- Modality dropout
- Time/channel masking
- Gaussian noise

---

## Distributed Training

### Single-Node Multi-GPU

```bash
torchrun --nproc_per_node=8 examples/02_distributed_training.py
```

### Multi-Node (SLURM)

```bash
sbatch submit_distributed.sh
```

See [examples/02_distributed_training.py](../examples/02_distributed_training.py) for complete FSDP setup.

---

## Monitoring & Debugging

### Experiment Tracking

- **MLflow**: Metrics, parameters, artifacts
- **Weights & Biases**: Real-time monitoring
- **TensorBoard**: Loss curves

### Checkpointing

```python
from neuros_neurofm.training import CheckpointManager

manager = CheckpointManager(checkpoint_dir='checkpoints/', save_top_k=3)
manager.save(model, optimizer, scheduler, global_step, metrics)
```

---

## Best Practices

1. **Start small**: Test with small model first
2. **Validate frequently**: Check val_loss every N steps  
3. **Clip gradients**: Prevent exploding gradients
4. **Use mixed precision**: 2x speedup with bf16
5. **Monitor GPU utilization**: Should be >90%

---

## Troubleshooting

### Out of Memory (OOM)

Solutions:
1. Reduce batch size
2. Enable gradient accumulation
3. Enable activation checkpointing
4. Use CPU offloading

### Slow Training

Solutions:
1. Increase num_workers
2. Use mixed precision (bf16)
3. Enable torch.compile()
4. Profile to find bottlenecks

### Loss Not Decreasing

Solutions:
1. Check learning rate (try 1e-4)
2. Verify data (check for NaNs)
3. Reduce model complexity
4. Check loss function

---

## Complete Example

See [examples/01_complete_training_workflow.py](../examples/01_complete_training_workflow.py) for a production-ready training script with all features.

---

## Next Steps

- **Evaluation**: [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md)
- **Interpretability**: [mechint_hooks_guide.md](mechint_hooks_guide.md)
- **Deployment**: [examples/05_deployment_inference.py](../examples/05_deployment_inference.py)

---

**Last Updated:** January 2025
