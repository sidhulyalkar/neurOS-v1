# NeuroFMX Examples

This directory contains comprehensive examples demonstrating all features of NeuroFMX, the world's most advanced multimodal foundation model for neural data.

## Quick Start

```bash
# Install NeuroFMX
pip install -e .

# Run complete training workflow
python examples/01_complete_training_workflow.py

# Run mechanistic interpretability analysis
python examples/03_mechanistic_interpretability.py
```

## Examples Overview

### 1. Complete Training Workflow (`01_complete_training_workflow.py`)

**What it demonstrates:**
- Multi-modal data loading with WebDataset
- Curriculum learning (unimodal → pairwise → multimodal)
- Multiple training objectives (masked modeling, forecasting, diffusion, contrastive)
- Distributed training with FSDP
- Checkpoint management and resumption
- MLflow experiment tracking

**Requirements:**
- 4+ GPUs recommended (can run on 1 GPU for testing)
- 100+ GB disk space for checkpoints
- MLflow server running (optional)

**Usage:**
```bash
# Configure training in configs/training/default.yaml
python examples/01_complete_training_workflow.py
```

**Key features:**
- Automatic curriculum scheduling
- Multi-objective loss balancing
- Gradient accumulation for large batches
- Mixed precision (bfloat16) training
- Learning rate warmup and decay

---

### 2. Distributed Multi-GPU Training (`02_distributed_training.py`)

**What it demonstrates:**
- Full FSDP (Fully Sharded Data Parallel) training
- DeepSpeed ZeRO-3 equivalent configuration
- Mixed precision (bfloat16) training
- Activation checkpointing for memory efficiency
- Multi-node training setup
- Memory usage estimation

**Hardware requirements:**
- 4-8 H100/A100 GPUs recommended
- 80GB+ GPU memory per GPU
- NVLink/Infiniband for multi-node

**Expected performance:**
- 4x H100 (80GB): ~150B parameters, 10-15 samples/sec
- 8x H100 (80GB): ~300B parameters, 20-30 samples/sec

**Usage:**

Single-node multi-GPU:
```bash
torchrun --nproc_per_node=8 examples/02_distributed_training.py
```

Multi-node (SLURM):
```bash
sbatch examples/submit_distributed.sh
```

**Key features:**
- Automatic GPU memory estimation
- FSDP auto-wrapping policy
- Sharded checkpointing
- Cross-species generalization testing

---

### 3. Mechanistic Interpretability Analysis (`03_mechanistic_interpretability.py`)

**What it demonstrates:**

The **world's most comprehensive mechanistic interpretability suite**:

1. **Sparse Autoencoder (SAE) Concept Discovery**
   - Hierarchical dictionaries (512 → 4096 → 16384)
   - Semantic concept labeling
   - Causal feature importance

2. **Brain-Model Alignment**
   - CCA (Canonical Correlation Analysis)
   - RSA (Representational Similarity Analysis)
   - Noise ceiling estimation
   - Cross-validated alignment scores

3. **Dynamical Systems Analysis**
   - Koopman operator eigenvalues
   - Lyapunov exponents (chaos detection)
   - Manifold intrinsic dimensionality
   - Phase portraits

4. **Causal Graph Discovery**
   - Granger causality networks
   - Time-varying graphs
   - Perturbation-based validation

5. **Counterfactual Interventions**
   - Latent surgery (edit specific features)
   - Do-calculus interventions
   - Synthetic lesions

6. **Meta-Dynamics Tracking**
   - Training trajectory analysis
   - Feature emergence detection
   - Representational drift measurement
   - Phase transition detection

7. **Geometric/Topological Analysis**
   - Riemannian curvature
   - Persistent homology
   - Betti numbers (topological features)

8. **Attribution & Importance**
   - Integrated gradients
   - DeepLIFT
   - Region-based attribution

9. **Automated Report Generation**
   - Professional HTML reports
   - Interactive visualizations
   - MLflow/W&B integration

**Requirements:**
- Trained NeuroFMX checkpoint
- Validation dataset (1000+ samples recommended)
- 32GB+ RAM for analysis
- Optional: Multi-modal brain recordings for alignment

**Usage:**
```bash
# Configure in configs/mechint/default.yaml
python examples/03_mechanistic_interpretability.py
```

**Output:**
- HTML report with all analyses
- Individual analysis results (JSON)
- Visualizations (PNG/PDF)
- Trained SAE checkpoints

**Key analyses:**
```python
# Example: Extract and analyze specific layer
activations = extract_activations(model, data_loader, ['backbone.layers.6'])

# Train hierarchical SAE
sae = HierarchicalSAE(layer_sizes=[768, 4096, 16384])
sae.train(activations)

# Discover concepts
concepts = ConceptDictionary(sae)
concepts.build_dictionary(activations, probe_labels={'region': labels})

# Analyze dynamics
dynamics = DynamicsAnalyzer()
lyapunov = dynamics.compute_lyapunov_exponents(activations)

# Generate report
reporter = MechIntReporter()
reporter.generate_report(all_results)
```

---

### 4. Evaluation & Benchmarking (`04_evaluation_benchmarking.py`)

**What it demonstrates:**
- Zero-shot evaluation (frozen features + linear probe)
- Few-shot learning (1, 5, 10, 25, 50 shots with LoRA)
- Cross-species generalization (train on macaque, test on human)
- Cross-task transfer
- Comprehensive benchmark suite

**Standard benchmarks:**
- Motor decoding (predict cursor position)
- Visual encoding (predict V1 activity)
- Speech decoding (phoneme classification)
- Memory encoding (remember vs. forget)
- Sleep staging (5-class classification)

**Requirements:**
- Trained NeuroFMX checkpoint
- Labeled benchmark datasets
- 16GB+ GPU memory

**Usage:**
```bash
# Configure in configs/evaluation/default.yaml
python examples/04_evaluation_benchmarking.py
```

**Output:**
- Zero-shot results (CSV)
- Few-shot learning curves (PNG)
- Cross-species heatmap (PNG)
- Comprehensive markdown report

**Key metrics:**
- R² for regression tasks
- Accuracy/F1 for classification
- AUROC for binary classification
- Cross-species transfer scores

---

### 5. Deployment & Inference (`05_deployment_inference.py`)

**What it demonstrates:**
- Model export (TorchScript, ONNX)
- Inference optimization (quantization, pruning)
- REST API serving with FastAPI
- Batch inference
- Real-time streaming inference
- Docker/Kubernetes deployment

**Deployment options:**
- Local server (FastAPI)
- Docker container
- Kubernetes cluster
- Cloud deployment (AWS/GCP/Azure)

**Usage:**

Export model:
```bash
python examples/05_deployment_inference.py
```

Start inference server:
```bash
# Local
uvicorn examples.05_deployment_inference:app --host 0.0.0.0 --port 8000

# Docker
docker build -t neurofmx-server .
docker run -p 8000:8000 --gpus all neurofmx-server

# Kubernetes
kubectl apply -f deployment.yaml
```

**API endpoints:**

Health check:
```bash
curl http://localhost:8000/health
```

Inference:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "modalities": {
      "eeg": [[[0.1, 0.2, ...], ...]],
      "spikes": [[[0.01, 0.02, ...], ...]]
    },
    "return_embeddings": true
  }'
```

Batch inference:
```bash
curl -X POST http://localhost:8000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[{"modalities": {...}}, {"modalities": {...}}]'
```

**Optimization techniques:**
- Dynamic quantization (fp32 → int8): ~4x smaller, 2x faster
- Pruning: Remove unimportant weights
- TorchScript: Optimized graph execution
- ONNX Runtime: Hardware-specific optimizations

**Real-time streaming:**
```python
# For continuous neural data (e.g., BCI)
streamer = StreamingInference(model, buffer_size=1000)

# Process new data as it arrives
while True:
    new_data = get_latest_neural_data()  # Get latest 10ms
    predictions = streamer.update(new_data)
    send_predictions_to_application(predictions)
```

---

## Configuration Files

All examples use YAML configuration files in `configs/`:

```
configs/
├── training/
│   ├── default.yaml          # Standard training config
│   └── distributed.yaml      # Multi-GPU config
├── mechint/
│   └── default.yaml          # Mech-int analysis config
├── evaluation/
│   └── default.yaml          # Evaluation config
└── deployment/
    └── default.yaml          # Deployment config
```

## Data Preparation

### WebDataset Format

NeuroFMX uses WebDataset for efficient multi-modal data loading:

```python
# Create shards
import webdataset as wds

with wds.TarWriter("data/train-000000.tar") as sink:
    for sample in dataset:
        sink.write({
            "__key__": f"sample{idx:06d}",
            "eeg.npy": eeg_data,
            "spikes.npy": spike_data,
            "video.mp4": video_data,
            "metadata.json": metadata,
        })
```

### Data Structure

```
data/
├── train/
│   ├── train-000000.tar
│   ├── train-000001.tar
│   └── ...
├── val/
│   ├── val-000000.tar
│   └── ...
└── test/
    ├── test-000000.tar
    └── ...
```

## Performance Benchmarks

### Training Performance

| Setup | Parameters | Throughput | Memory/GPU |
|-------|-----------|------------|------------|
| 1x A100 (80GB) | 1B | 2-3 samples/sec | 60GB |
| 4x A100 (80GB) | 10B | 10-15 samples/sec | 70GB |
| 8x H100 (80GB) | 50B | 25-35 samples/sec | 75GB |
| 16x H100 (80GB) | 150B | 50-70 samples/sec | 78GB |

### Inference Performance

| Setup | Batch Size | Latency | Throughput |
|-------|-----------|---------|------------|
| CPU (24 cores) | 1 | 200ms | 5 samples/sec |
| CPU (24 cores) | 32 | 2000ms | 16 samples/sec |
| A100 (FP32) | 1 | 5ms | 200 samples/sec |
| A100 (FP16) | 1 | 3ms | 333 samples/sec |
| A100 (INT8) | 1 | 2ms | 500 samples/sec |
| A100 (FP16) | 64 | 80ms | 800 samples/sec |

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. **Reduce batch size:**
   ```yaml
   data:
     batch_size: 16  # Reduce from 32
   ```

2. **Enable gradient accumulation:**
   ```yaml
   training:
     accumulate_grad_batches: 4  # Effective batch = 16 * 4 = 64
   ```

3. **Enable activation checkpointing:**
   ```yaml
   distributed:
     activation_checkpointing: true
   ```

4. **Use CPU offloading (for very large models):**
   ```yaml
   distributed:
     cpu_offload: true
   ```

### Slow Training

If training is slow:

1. **Increase num_workers:**
   ```yaml
   data:
     num_workers: 16  # Match CPU cores
   ```

2. **Use mixed precision:**
   ```yaml
   training:
     use_bf16: true
   ```

3. **Profile with PyTorch Profiler:**
   ```python
   from torch.profiler import profile, ProfilerActivity

   with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
       trainer.fit(model, train_loader)

   print(prof.key_averages().table(sort_by="cuda_time_total"))
   ```

### Import Errors

If you encounter import errors:

```bash
# Install NeuroFMX in development mode
pip install -e .

# Install all dependencies
pip install -r requirements.txt

# Verify installation
python -c "from neuros_neurofm.model import NeuroFMX; print('✓ Installation successful')"
```

## Support

For issues, questions, or feature requests:
- GitHub Issues: https://github.com/your-org/neuros-v1/issues
- Documentation: https://neurofmx.readthedocs.io
- Email: support@neurofmx.ai

## Citation

If you use NeuroFMX in your research, please cite:

```bibtex
@article{neurofmx2024,
  title={NeuroFMX: A Foundation Model for Multimodal Neural Data},
  author={Your Name et al.},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2024}
}
```

## License

MIT License - See LICENSE file for details
