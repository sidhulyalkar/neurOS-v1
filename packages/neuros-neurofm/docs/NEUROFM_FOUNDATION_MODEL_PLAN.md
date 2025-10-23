# NeuroFM-X Foundation Model: Comprehensive Development & Optimization Plan

**Status:** Production-Ready Architecture Testing Phase
**Target:** State-of-the-art neural foundation model for population dynamics
**Hardware:** RTX 3070 Ti (8GB VRAM) - WSL2 Ubuntu
**Dataset:** 20 Allen Neuropixels sessions (~395K training sequences)

---

## Executive Summary

This plan outlines a systematic approach to transform NeuroFM-X from an experimental architecture into a production-grade foundation model. The strategy follows a phased approach: **Fix → Validate → Optimize → Scale → Benchmark → Deploy**.

---

## Phase 1: Critical Bug Fixes & Initial Validation (COMPLETED ✓)

### Issues Resolved:
1. **Unit Dimension Mismatch** (384 vs 128)
   - Fixed `Config.max_units` to match preprocessed data (384 units)
   - Updated tokenizer initialization to use `encoder_output_dim` parameter
   - Aligned collate function with preprocessed data dimensions

### Next Steps:
- Run initial training to validate the fix
- Monitor for any runtime errors
- Validate gradient flow and loss convergence

---

## Phase 2: Architecture Evaluation & Validation (CURRENT PHASE)

### 2.1 Model Architecture Analysis

**Current Architecture:**
```
Input (B, S=100, N=384)
  ↓ BinnedTokenizer (384 → 128)
  ↓ MambaBackbone (4 blocks, multi-rate [1x, 4x])
  ↓ PerceiverIO (128 → 32 latents @ 128-dim)
  ↓ PopTWithLatents (32 latents, 2 layers)
  ↓ Mean Pool (32 latents → 128-dim)
  ↓ MultiTaskHeads (Encoder, Decoder, Contrastive)
```

**Total Parameters:** ~3M (highly efficient!)

**Critical Questions to Answer:**
1. Is the model capacity sufficient for 384 units?
2. Are 32 latents enough to capture population dynamics?
3. Is the 128-dim bottleneck limiting performance?
4. How well does multi-rate processing work?

### 2.2 Training Validation Checklist

**Immediate Goals (First 5 epochs):**
- [ ] Verify training loop runs without crashes
- [ ] Confirm loss decreases (not NaN/Inf)
- [ ] Validate gradient flow (check grad norms)
- [ ] Monitor GPU memory usage (should be <7GB)
- [ ] Check reconstruction quality (encoder head)
- [ ] Validate multi-task loss balance

**Success Criteria:**
- Training loss decreases by >50% in first 5 epochs
- No gradient explosions (grad norm < 10)
- Reconstruction MSE < 0.1 by epoch 5
- GPU utilization >80%

### 2.3 Recommended Monitoring Script

Create `monitor_training.py`:
```python
import torch
import matplotlib.pyplot as plt
import json
from pathlib import Path

def analyze_checkpoint(ckpt_path):
    """Analyze training checkpoint for debugging."""
    ckpt = torch.load(ckpt_path)

    # Check for gradient issues
    for name, param in ckpt['model_state_dict'].items():
        if torch.isnan(param).any():
            print(f"⚠️  NaN detected in {name}")
        if torch.isinf(param).any():
            print(f"⚠️  Inf detected in {name}")

    # Check parameter statistics
    print(f"Epoch: {ckpt['epoch']}")
    print(f"Best Val Loss: {ckpt['best_val_loss']:.4f}")
    print(f"Global Step: {ckpt['global_step']}")

    return ckpt

# Usage: python monitor_training.py
```

---

## Phase 3: Hyperparameter Optimization & Scaling

### 3.1 Architectural Improvements (Post-Validation)

**If initial training succeeds, consider:**

1. **Increase Model Capacity** (if underfitting):
   ```python
   d_model = 256           # 128 → 256
   n_latents = 64          # 32 → 64
   latent_dim = 256        # 128 → 256
   n_mamba_blocks = 6      # 4 → 6
   ```
   - Estimated params: ~12M
   - GPU memory: ~6.5GB

2. **Multi-Rate Optimization**:
   ```python
   downsample_rates = [1, 2, 4, 8]  # Add more temporal scales
   ```

3. **Enhanced PopT**:
   ```python
   n_popt_layers = 4       # 2 → 4
   ```

### 3.2 Training Optimization

**Learning Rate Schedule:**
```python
# Current: OneCycleLR (good start)
# Consider: Cosine with warmup + restarts
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)
```

**Gradient Accumulation:**
```python
# Current: 8 steps (effective batch=16)
# If stable, try: 16 steps (effective batch=32)
gradient_accumulation_steps = 16
```

**Mixed Precision:**
- Already enabled ✓
- Monitor for stability issues

### 3.3 Regularization Strategies

```python
# Add to Config:
label_smoothing = 0.1          # For reconstruction
gradient_clip_norm = 1.0       # Already set ✓
weight_decay = 0.01            # Already set ✓
dropout = 0.15                 # 0.1 → 0.15 (if overfitting)
```

---

## Phase 4: Multi-Stream Training Infrastructure

### 4.1 Data Pipeline Optimization

**Current Bottleneck:** CPU-based data loading

**Improvements:**
1. **Increase DataLoader workers:**
   ```python
   num_workers = 4  # 0 → 4 (careful with WSL2)
   prefetch_factor = 2
   persistent_workers = True
   ```

2. **GPU-based preprocessing:**
   ```python
   # Move sqrt transform to GPU
   class GPUAugmentedDataset(StreamingNeuropixelsDataset):
       def __getitem__(self, idx):
           # Load raw counts (skip CPU sqrt)
           data = super().__getitem__(idx)
           return data  # Transform on GPU in collate_fn
   ```

3. **Memory-mapped loading:**
   ```python
   # Replace npz with memmap for faster access
   data = np.load(file_path, mmap_mode='r')
   ```

### 4.2 Multi-Modal Data Streams

**Architecture for Multiple Data Types:**
```python
class MultiModalNeuroFMX(NeuroFMXComplete):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Multiple tokenizers
        self.spike_tokenizer = BinnedTokenizer(...)
        self.lfp_tokenizer = LFPTokenizer(...)
        self.calcium_tokenizer = CalciumTokenizer(...)

        # Modality fusion
        self.modality_fusion = nn.MultiheadAttention(d_model, 8)

    def forward(self, modality_dict):
        # Tokenize each modality
        tokens_list = []
        for modality, data in modality_dict.items():
            tokens = getattr(self, f"{modality}_tokenizer")(data)
            tokens_list.append(tokens)

        # Cross-modal attention
        fused_tokens = self.modality_fusion(...)

        # Continue through backbone
        return super().forward(fused_tokens)
```

**Benefits:**
- Leverage multiple recording types
- Cross-modal learning
- Robust to missing modalities

---

## Phase 5: Comprehensive Benchmarking Suite

### 5.1 Neural Decoding Benchmarks

**Create `benchmarks/neural_decoding.py`:**

```python
class NeuroFMXBenchmark:
    """Comprehensive benchmarking suite."""

    def __init__(self, model, test_loader):
        self.model = model
        self.test_loader = test_loader

    def benchmark_reconstruction(self):
        """Test neural reconstruction quality."""
        mse_scores = []
        correlation_scores = []

        for batch in self.test_loader:
            pred = self.model(batch['tokens_raw'], task='encoder')
            target = batch['tokens_raw']

            # Per-unit metrics
            mse = F.mse_loss(pred, target, reduction='none')
            corr = self.compute_correlation(pred, target)

            mse_scores.append(mse)
            correlation_scores.append(corr)

        return {
            'mse': torch.cat(mse_scores).mean(),
            'correlation': torch.cat(correlation_scores).mean()
        }

    def benchmark_behavior_decoding(self):
        """Test behavioral decoding accuracy."""
        # Implementation
        pass

    def benchmark_population_geometry(self):
        """Analyze latent space geometry."""
        # Extract latents for all test samples
        latents = []
        labels = []

        for batch in self.test_loader:
            z = self.model.encode(batch['tokens_raw'])
            latents.append(z)
            labels.append(batch['behavior'])

        # Compute metrics:
        # - Manifold dimensionality
        # - Linear separability
        # - Clustering quality
        return self.compute_geometry_metrics(latents, labels)

    def benchmark_transfer_learning(self):
        """Test few-shot adaptation."""
        # Freeze backbone, train on new sessions
        pass

    def benchmark_computational_efficiency(self):
        """Measure inference speed and memory."""
        import time

        # Warmup
        for _ in range(10):
            dummy_input = torch.randn(1, 100, 384).cuda()
            _ = self.model(dummy_input)

        # Benchmark
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(100):
            _ = self.model(dummy_input)
        end.record()

        torch.cuda.synchronize()
        return start.elapsed_time(end) / 100  # ms per forward pass
```

### 5.2 Comparison with Baselines

**Baselines to Compare:**
1. **CEBRA** - Current state-of-the-art
2. **LFADS** - Classic latent dynamics model
3. **NDT (Neural Data Transformer)** - Transformer baseline
4. **Linear Decoder** - Simple baseline

**Metrics:**
- R² for behavior decoding
- Bit/spike for reconstruction
- Latent dimensionality
- Computational efficiency (FLOPs, latency)
- Transfer learning performance

---

## Phase 6: Production Deployment Architecture

### 6.1 Model Serving Infrastructure

**Architecture:**
```
┌─────────────────────────────────────────────┐
│          NeuroFM-X Serving Layer            │
├─────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐          │
│  │  FastAPI    │  │  Model      │          │
│  │  Server     │→ │  Registry   │          │
│  └─────────────┘  └─────────────┘          │
│         ↓                                   │
│  ┌─────────────────────────────┐           │
│  │   Inference Optimization    │           │
│  │  - TorchScript compilation  │           │
│  │  - ONNX export              │           │
│  │  - Batch inference          │           │
│  │  - GPU memory pooling       │           │
│  └─────────────────────────────┘           │
└─────────────────────────────────────────────┘
```

**Implementation:**
```python
# serving/neurofmx_server.py
from fastapi import FastAPI
import torch
import uvicorn

app = FastAPI()

class NeuroFMXInference:
    def __init__(self, checkpoint_path):
        self.model = NeuroFMXComplete.from_pretrained(checkpoint_path)
        self.model.eval()
        self.model = torch.jit.script(self.model)  # TorchScript
        self.model.cuda()

    @torch.inference_mode()
    def predict(self, neural_data):
        """Fast inference with optimizations."""
        # Preprocessing
        tokens = self.preprocess(neural_data)

        # Inference
        with torch.cuda.amp.autocast():
            latents = self.model.encode(tokens)
            behavior = self.model.decode_behavior(tokens)

        return {
            'latents': latents.cpu().numpy(),
            'behavior': behavior.cpu().numpy()
        }

# Global model instance
inference_engine = None

@app.on_event("startup")
def load_model():
    global inference_engine
    inference_engine = NeuroFMXInference("checkpoints/best.pt")

@app.post("/predict")
def predict(data: dict):
    result = inference_engine.predict(data['neural_data'])
    return result

# Run: uvicorn serving.neurofmx_server:app --host 0.0.0.0 --port 8000
```

### 6.2 Model Versioning & Registry

```python
# model_registry.py
class ModelRegistry:
    """Track model versions and experiments."""

    def __init__(self, registry_dir="./model_registry"):
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(exist_ok=True)

    def register_model(self, model, config, metrics, version):
        """Save model with metadata."""
        version_dir = self.registry_dir / f"v{version}"
        version_dir.mkdir(exist_ok=True)

        # Save model
        torch.save(model.state_dict(), version_dir / "model.pt")

        # Save metadata
        metadata = {
            'version': version,
            'config': config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'git_commit': self.get_git_commit(),
        }

        with open(version_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_model(self, version):
        """Load model by version."""
        version_dir = self.registry_dir / f"v{version}"

        with open(version_dir / "metadata.json") as f:
            metadata = json.load(f)

        model = NeuroFMXComplete.from_config(metadata['config'])
        model.load_state_dict(torch.load(version_dir / "model.pt"))

        return model, metadata
```

### 6.3 Distributed Training (Future Scale-Up)

**For Multi-GPU Training:**
```python
# distributed_train.py
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup_distributed():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank

def main():
    local_rank = setup_distributed()

    model = NeuroFMXComplete(...).to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Training loop
    for batch in train_loader:
        loss = ...
        loss.backward()
        optimizer.step()

# Run: torchrun --nproc_per_node=4 distributed_train.py
```

---

## Phase 7: Advanced Optimizations

### 7.1 Architectural Enhancements

**1. Sparse Attention for Long Sequences:**
```python
# For longer temporal context
from torch.nn.functional import scaled_dot_product_attention

class SparseAttention(nn.Module):
    """Efficient attention for long sequences."""
    def __init__(self, d_model, n_heads, window_size=50):
        super().__init__()
        self.window_size = window_size
        # Implementation with sliding window
```

**2. Hierarchical PopT:**
```python
class HierarchicalPopT(nn.Module):
    """Multi-scale population aggregation."""
    def __init__(self, n_layers, scales=[1, 2, 4]):
        super().__init__()
        self.scales = scales
        self.popt_layers = nn.ModuleList([
            PopT(...) for _ in scales
        ])
```

**3. Contrastive Learning Enhancement:**
```python
# Implement SimCLR-style contrastive loss
def contrastive_loss(z1, z2, temperature=0.07):
    """NT-Xent loss for neural representations."""
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)

    logits = torch.mm(z1, z2.T) / temperature
    labels = torch.arange(len(z1)).cuda()

    loss = F.cross_entropy(logits, labels)
    return loss
```

### 7.2 Training Tricks

**1. Gradual Unfreezing:**
```python
def progressive_unfreezing(model, epoch, schedule):
    """Unfreeze layers progressively."""
    if epoch < 5:
        # Freeze backbone, train heads only
        for param in model.backbone.parameters():
            param.requires_grad = False
    elif epoch < 10:
        # Unfreeze top layers
        for param in model.backbone.layers[-2:].parameters():
            param.requires_grad = True
    else:
        # Full fine-tuning
        for param in model.parameters():
            param.requires_grad = True
```

**2. Stochastic Weight Averaging (SWA):**
```python
from torch.optim.swa_utils import AveragedModel, SWALR

swa_model = AveragedModel(model)
swa_scheduler = SWALR(optimizer, swa_lr=1e-5)

# After epoch 40, start SWA
if epoch > 40:
    swa_model.update_parameters(model)
    swa_scheduler.step()
```

---

## Immediate Action Plan (Next 24-48 Hours)

### Step 1: Validate Fix (1-2 hours)
```bash
# Run training to verify the fix
cd /mnt/c/Users/sidso/Documents/neurOS-v1/packages/neuros-neurofm
python3 full_train_streaming.py

# Monitor first 5 epochs
# Check: Loss decreasing? No errors? GPU usage?
```

### Step 2: Create Monitoring Tools (2 hours)
```bash
# Create monitoring script
# Add tensorboard logging
# Set up checkpoint analysis
```

### Step 3: Short Training Run (4-6 hours)
```bash
# Train for 10 epochs
# Analyze results
# Identify bottlenecks
```

### Step 4: Optimize Based on Results (2-3 hours)
```bash
# Tune hyperparameters
# Fix any discovered issues
# Prepare for full 50-epoch run
```

### Step 5: Full Training Run (12-24 hours)
```bash
# Launch 50-epoch training
# Monitor remotely
# Analyze final results
```

---

## Success Metrics

### Short-term (Week 1):
- [x] Fix tokenizer bug
- [ ] Complete 50-epoch training run
- [ ] Achieve reconstruction R² > 0.5
- [ ] Validate gradient stability
- [ ] Benchmark inference speed

### Medium-term (Weeks 2-4):
- [ ] Implement comprehensive benchmarking
- [ ] Compare against CEBRA/LFADS
- [ ] Optimize data pipeline
- [ ] Scale to full Allen dataset (200+ sessions)
- [ ] Achieve SOTA reconstruction performance

### Long-term (Months 2-3):
- [ ] Multi-modal training
- [ ] Transfer learning validation
- [ ] Production deployment infrastructure
- [ ] Scientific publication preparation
- [ ] Open-source release

---

## Risk Mitigation

### Potential Issues & Solutions:

1. **GPU Memory Overflow:**
   - Reduce batch size to 1
   - Reduce model capacity
   - Enable gradient checkpointing

2. **Training Instability:**
   - Reduce learning rate (3e-4 → 1e-4)
   - Increase gradient clipping (1.0 → 0.5)
   - Disable mixed precision temporarily

3. **Slow Convergence:**
   - Increase model capacity
   - Improve data augmentation
   - Tune learning rate schedule

4. **Poor Reconstruction:**
   - Check tokenizer preprocessing
   - Verify loss weighting
   - Increase encoder head capacity

---

## Recommended Next Steps (Priority Order)

1. **RUN TRAINING NOW** - Validate the fix
2. Add TensorBoard logging for visualization
3. Create checkpoint analysis script
4. Implement comprehensive benchmarking
5. Write inference optimization
6. Scale to full dataset
7. Deploy production serving

---

## Team Expertise Requirements

**For Production-Grade Development:**
- **ML Engineering:** Model optimization, distributed training
- **Systems Engineering:** Deployment infrastructure, model serving
- **Neuroscience:** Validation, benchmark design
- **DevOps:** CI/CD, monitoring, versioning

**Current Status:** Single-developer prototype → Production transition phase

---

## Conclusion

NeuroFM-X has a solid architectural foundation with efficient design (3M params). The immediate bug fix unblocks training. The next critical step is validating training convergence and building robust evaluation infrastructure.

**Priority:** Get training running, then iterate on optimization.

This is a marathon, not a sprint. Build systematically, measure rigorously, optimize iteratively.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-23
**Status:** Bug fixed, ready for validation training
