# NeuroFM-X Implementation Plan

## Executive Summary

**NeuroFM-X** is a state-of-the-art neural foundational model that combines cutting-edge architectures for multi-modal neural data processing, transfer learning, and generation. This document outlines the complete implementation plan for integrating NeuroFM-X into neurOS-v1.

**Key Innovations:**
- **Selective State-Space Models (Mamba)** - Linear-time long-sequence modeling (5x faster than Transformers)
- **Perceiver-IO Fusion** - Efficient multi-modal integration
- **Population Transformer (PopT)** - Cross-session neural alignment
- **Latent Diffusion** - Realistic neural data generation and imputation
- **Adapter-based Transfer** - Few-shot learning with frozen core
- **CEBRA Contrastive Learning** - Behavior-aligned latent spaces

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        NeuroFM-X Architecture                    │
│                                                                  │
│  Input Data                                                      │
│  ├─ Spikes (event tokens: Δt, unit_id, session_id)             │
│  ├─ Calcium (binned ΔF/F traces)                               │
│  ├─ iEEG/LFP (continuous signals)                               │
│  ├─ Behavior (kinematics, EMG, audio)                          │
│  └─ Stimuli (video features, task metadata)                    │
│                    ↓                                             │
│  ┌────────────────────────────────────────────────────┐        │
│  │           Tokenizers & Encoders                     │        │
│  │  • Spikes-as-tokens (POYO-style)                   │        │
│  │  • Population bins (10-20ms)                       │        │
│  │  • 1D conv downsampler for LFP                     │        │
│  └────────────────────────────────────────────────────┘        │
│                    ↓                                             │
│  ┌────────────────────────────────────────────────────┐        │
│  │      Mamba/SSM Backbone (Linear Complexity)         │        │
│  │  • d_model=768, n_blocks=16                        │        │
│  │  • Multi-rate streams: 5ms | 20ms | 80ms          │        │
│  │  • Cross-scale gating every 2 blocks               │        │
│  └────────────────────────────────────────────────────┘        │
│                    ↓                                             │
│  ┌────────────────────────────────────────────────────┐        │
│  │        Perceiver-IO Fusion Hub                      │        │
│  │  • 512-dim latents, 128 slots                      │        │
│  │  • Cross-attention to all modalities               │        │
│  └────────────────────────────────────────────────────┘        │
│                    ↓                                             │
│  ┌────────────────────────────────────────────────────┐        │
│  │          PopT Population Aggregator                 │        │
│  │  • 3 layers, width 512                             │        │
│  │  • Permutation-invariant across neurons            │        │
│  └────────────────────────────────────────────────────┘        │
│                    ↓                                             │
│  ┌────────────────────────────────────────────────────┐        │
│  │    Generative Prior (Latent Diffusion)             │        │
│  │  • Forecasting: 1-2 second horizon                │        │
│  │  • Conditional generation                          │        │
│  │  • Imputation & augmentation                       │        │
│  └────────────────────────────────────────────────────┘        │
│                    ↓                                             │
│  ┌────────────────────────────────────────────────────┐        │
│  │             Multi-Task Heads                        │        │
│  │  • Decoding (velocity, EMG, phonemes)             │        │
│  │  • Encoding (spike prediction)                     │        │
│  │  • Contrastive (CEBRA-style)                      │        │
│  └────────────────────────────────────────────────────┘        │
│                    ↓                                             │
│  ┌────────────────────────────────────────────────────┐        │
│  │            Transfer Adapters                        │        │
│  │  • Unit-ID embeddings (new neurons)                │        │
│  │  • Session/Region stitchers                        │        │
│  │  • LoRA (r=8) on last blocks                       │        │
│  └────────────────────────────────────────────────────┘        │
│                    ↓                                             │
│  Output: Predictions, Latents, Generated Data                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Foundation (Weeks 1-2)
**Goal:** Set up infrastructure and data pipelines

#### Tasks:
1. **Package Structure**
   - Create `packages/neuros-neurofm/` subdirectory
   - Set up PyTorch Lightning framework
   - Configure Hydra for experiment management
   - Add dependencies: `mamba-ssm`, `pytorch-lightning>=2.4`

2. **Data Integration**
   - Implement NWB readers/writers in `neuros/io/nwb_io.py`
   - Create data loaders for IBL, Allen, DANDI
   - Build validation pipeline for metadata
   - Test NWB round-trip (write → read → validate)

3. **Base Classes**
   - Extend `BaseFoundationModel` for NeuroFM-X
   - Create Lightning module skeleton
   - Set up configuration system

**Deliverables:**
- ✅ Package structure
- ✅ NWB data pipeline
- ✅ Configuration system
- ✅ Unit tests for data loading

---

### Phase 2: Tokenizers & Encoders (Week 3)
**Goal:** Convert neural data to model-ready tokens

#### Components:

1. **Spikes-as-Tokens**
   ```python
   # Event representation
   Token = (Δt, unit_id, session_id, metadata)
   # Learned embeddings for each field
   # Output: token sequence for SSM
   ```

2. **Population Bins**
   ```python
   # Time-binned counts (10-20ms bins)
   # Shape: (batch, time, n_neurons)
   # Support masking for masked modeling
   ```

3. **Continuous Signal Encoder**
   ```python
   # 1D conv downsampler for iEEG/LFP
   # Or STFT → learnable encoder
   # Output: fixed-rate features
   ```

**Deliverables:**
- ✅ `neurofm/tokenizers/spike_tokenizer.py`
- ✅ `neurofm/tokenizers/binned_encoder.py`
- ✅ `neurofm/tokenizers/lfp_encoder.py`
- ✅ Unit tests for each tokenizer

---

### Phase 3: SSM Backbone (Weeks 4-5)
**Goal:** Implement efficient Mamba-based backbone

#### Architecture:
```python
class MambaBackbone(nn.Module):
    """
    Selective State-Space Model backbone.

    Key Features:
    - Linear complexity O(L) vs O(L²) for attention
    - Multi-rate processing (5ms, 20ms, 80ms)
    - Cross-scale gating
    - Handles sequences up to millions of timesteps
    """
    def __init__(self, d_model=768, n_blocks=16, d_state=64):
        # Multi-rate SSM blocks
        self.streams = nn.ModuleList([
            MambaBlock(d_model, rate=5),   # 5ms
            MambaBlock(d_model, rate=20),  # 20ms
            MambaBlock(d_model, rate=80)   # 80ms
        ])

        # Cross-scale gates
        self.gates = nn.ModuleList([
            CrossScaleGate(d_model)
            for _ in range(n_blocks // 2)
        ])
```

**Implementation:**
- Use `mamba-ssm` library for core layers
- Build multi-rate streams with different dilation rates
- Add cross-scale attention gates
- Optimize for long sequences (tested up to 1M steps)

**Deliverables:**
- ✅ `neurofm/backbones/mamba_backbone.py`
- ✅ `neurofm/backbones/ssm_blocks.py`
- ✅ Performance benchmarks vs Transformer
- ✅ Long-sequence tests (100K+ steps)

---

### Phase 4: Perceiver-IO Fusion (Week 6)
**Goal:** Multi-modal latent fusion

#### Architecture:
```python
class PerceiverFusion(nn.Module):
    """
    Perceiver-IO style fusion for multiple modalities.

    Complexity: O(L*M) instead of O(L²) where M << L
    """
    def __init__(self, latent_dim=512, latent_slots=128):
        # Learnable latent array
        self.latents = nn.Parameter(
            torch.randn(latent_slots, latent_dim)
        )

        # Cross-attention to each modality
        self.cross_attns = nn.ModuleDict({
            'neural': CrossAttention(latent_dim),
            'behavior': CrossAttention(latent_dim),
            'video': CrossAttention(latent_dim),
        })
```

**Key Features:**
- Latent queries attend to each modality
- Scalable to N modalities
- Shared latent space for all inputs

**Deliverables:**
- ✅ `neurofm/fusion/perceiver_io.py`
- ✅ Multi-modal integration tests
- ✅ Computational efficiency validation

---

### Phase 5: PopT Aggregator (Week 7)
**Goal:** Population-level neural representation

#### Architecture:
```python
class PopulationTransformer(nn.Module):
    """
    Population-level aggregator (PopT).

    Features:
    - Permutation-invariant across neurons
    - Handles variable channel counts
    - Cross-session alignment
    """
    def __init__(self, layers=3, width=512):
        # Treats each neuron as a token
        self.channel_encoder = nn.Linear(temporal_dim, width)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(width, nhead=8),
            num_layers=layers
        )
        self.pooler = nn.AdaptiveAvgPool1d(1)
```

**Self-Supervised Objectives:**
- Channel-level reconstruction
- Ensemble-level consistency
- Cross-session alignment

**Deliverables:**
- ✅ `neurofm/population/popt.py`
- ✅ Self-supervised pretraining code
- ✅ Variable neuron count tests

---

### Phase 6: Latent Diffusion (Week 8)
**Goal:** Generative modeling for forecasting/imputation

#### Architecture:
```python
class LatentDiffusion(nn.Module):
    """
    Latent diffusion model for neural data.

    Two-stage:
    1. Encoder: neural data → latents
    2. Diffusion: conditional DDPM on latents
    """
    def __init__(self, latent_dim=256, horizon_s=1.5):
        # VAE-style encoder
        self.encoder = ConvEncoder(latent_dim)

        # Conditional UNet diffusion
        self.diffusion = ConditionalUNet(
            latent_dim,
            timesteps=1000
        )

    def forward(self, x, condition=None):
        # Encode to latents
        z = self.encoder(x)

        # Diffusion process
        z_noisy = self.add_noise(z)
        z_pred = self.diffusion(z_noisy, condition)

        return z_pred
```

**Applications:**
- Neural forecasting (1-2 second horizon)
- Data imputation (missing neurons)
- Data augmentation for training

**Deliverables:**
- ✅ `neurofm/priors/latent_diffusion.py`
- ✅ Conditional generation tests
- ✅ Forecasting benchmarks

---

### Phase 7: Multi-Task Heads (Week 9)
**Goal:** Task-specific output layers

#### Heads:

1. **Decoding Head**
   ```python
   # Predict behavior from neural data
   # Tasks: velocity, EMG, phonemes, etc.
   class DecodingHead(nn.Module):
       def __init__(self, latent_dim, n_tasks):
           self.task_heads = nn.ModuleDict({
               'velocity': nn.Linear(latent_dim, 3),
               'emg': nn.Linear(latent_dim, 8),
               'phoneme': nn.Linear(latent_dim, 39)
           })
   ```

2. **Encoding Head**
   ```python
   # Predict neural activity (held-out neurons)
   # Poisson/Bernoulli likelihood
   class EncodingHead(nn.Module):
       def __init__(self, latent_dim, max_neurons):
           self.spike_predictor = nn.Linear(latent_dim, max_neurons)
           # Poisson rate λ = exp(output)
   ```

3. **Contrastive Head**
   ```python
   # CEBRA-style behavior alignment
   class ContrastiveHead(nn.Module):
       def __init__(self, latent_dim, projection_dim=128):
           self.projector = nn.Sequential(
               nn.Linear(latent_dim, projection_dim),
               nn.ReLU(),
               nn.Linear(projection_dim, projection_dim)
           )
   ```

**Deliverables:**
- ✅ `neurofm/heads/decoding.py`
- ✅ `neurofm/heads/encoding.py`
- ✅ `neurofm/heads/contrastive.py`
- ✅ Multi-task training tests

---

### Phase 8: Adapters (Week 10)
**Goal:** Transfer learning components

#### Adapter Types:

1. **Unit-ID Adapter**
   ```python
   # Learn embeddings for new neurons
   # Core model stays frozen
   class UnitAdapter(nn.Module):
       def __init__(self, n_units, embed_dim):
           self.unit_embeddings = nn.Embedding(n_units, embed_dim)

       def forward(self, unit_ids):
           return self.unit_embeddings(unit_ids)
   ```

2. **Session/Region Stitchers**
   ```python
   # Linear input/output adaptation
   class SessionStitcher(nn.Module):
       def __init__(self, in_dim, out_dim):
           self.input_proj = nn.Linear(in_dim, out_dim)
           self.output_proj = nn.Linear(out_dim, in_dim)
   ```

3. **LoRA Layers**
   ```python
   # Low-rank adaptation on backbone
   class LoRALayer(nn.Module):
       def __init__(self, dim, rank=8):
           self.lora_A = nn.Linear(dim, rank, bias=False)
           self.lora_B = nn.Linear(rank, dim, bias=False)
           self.scaling = rank
   ```

**Deliverables:**
- ✅ `neurofm/adapters/unit_adapter.py`
- ✅ `neurofm/adapters/stitchers.py`
- ✅ `neurofm/adapters/lora.py`
- ✅ Few-shot transfer tests

---

### Phase 9: Training Pipeline (Weeks 11-12)
**Goal:** PyTorch Lightning training framework

#### Training Stages:

**Stage 1: Self-Supervised Pretraining**
```python
# Objectives:
# 1. Masked reconstruction (Poisson/Bernoulli)
# 2. Contrastive learning (CEBRA)
# 3. Latent diffusion auxiliary

Loss = λ1*masked_recon + λ2*contrastive + λ3*diffusion
```

**Stage 2: Task Head Training**
```python
# Add supervised tasks
# Freeze core, train heads
Loss += λ4*task_loss
```

**Stage 3: Adapter Fine-tuning**
```python
# Freeze core + heads
# Train only adapters on new data
```

**Configuration System:**
```yaml
# configs/model/neurofm_x.yaml
model:
  backbone: mamba2
  d_model: 768
  n_blocks: 16
  fusion: perceiver_io

# configs/train/pretrain.yaml
train:
  optimizer: adamw
  lr: 3e-4
  max_steps: 300000
  loss_weights:
    masked: 1.0
    contrastive: 0.2
    diffusion: 0.5
```

**Deliverables:**
- ✅ `neurofm/model.py` - Full model assembly
- ✅ `neurofm/lightning_module.py` - Training logic
- ✅ `scripts/train_neurofm.py` - Entry point
- ✅ Hydra configs for all experiments

---

### Phase 10: Evaluation Suite (Week 13)
**Goal:** Comprehensive benchmarking

#### Benchmarks:

1. **FALCON** - Few-shot robustness
   ```python
   # Varying calibration time: 0, 1, 5, 10, 30 minutes
   # Measure decode accuracy vs time
   ```

2. **Cross-Subject Transfer**
   ```python
   # Freeze core, train adapters on new subject
   # Measure performance drop
   ```

3. **Forecasting/Imputation**
   ```python
   # Negative log-likelihood
   # Bits-per-spike
   # CRPS (continuous ranked probability score)
   ```

4. **Computational Efficiency**
   ```python
   # Throughput: samples/sec
   # Latency: ms per prediction
   # Memory: peak GPU usage
   ```

**Deliverables:**
- ✅ `scripts/evaluate_neurofm.py`
- ✅ `neurofm/eval/falcon.py`
- ✅ `neurofm/eval/metrics.py`
- ✅ Benchmark reports (CSV/HTML)

---

### Phase 11: neurOS Integration (Week 14)
**Goal:** Seamless integration with neurOS pipeline

#### Integration Points:

1. **Model Registration**
   ```python
   # Register in neurOS model registry
   from neuros.models.base_model import BaseModel

   class NeuroFMXModel(BaseModel):
       def __init__(self, config):
           self.neurofm = NeuroFMX.load_from_checkpoint(...)

       def predict(self, X):
           return self.neurofm.forward(X)
   ```

2. **Real-Time Pipeline**
   ```python
   # Use in neurOS pipelines
   pipeline = Pipeline(
       driver=BrainFlowDriver(),
       model=NeuroFMXModel(config),
       fs=250.0
   )

   metrics = await pipeline.run(duration=60.0)
   ```

3. **Data Drivers**
   ```python
   # NWB data driver
   class NWBDriver(BaseDriver):
       def __init__(self, nwb_file):
           self.nwb = load_nwb(nwb_file)

       def get_data(self):
           return self.nwb.get_spikes()
   ```

**Deliverables:**
- ✅ `neuros/models/neurofm_model.py`
- ✅ `neuros/drivers/nwb_driver.py`
- ✅ Integration tests
- ✅ Real-time demo

---

### Phase 12: Tutorials & Documentation (Weeks 15-16)
**Goal:** Comprehensive user guides

#### Tutorials:

1. **Tutorial 1: Data Preparation**
   - Loading IBL/Allen datasets
   - NWB conversion
   - Data validation

2. **Tutorial 2: Pretraining**
   - Self-supervised learning
   - Monitoring metrics
   - Checkpoint management

3. **Tutorial 3: Fine-tuning**
   - Adapter training
   - Few-shot learning
   - Task-specific heads

4. **Tutorial 4: Evaluation**
   - FALCON benchmarks
   - Cross-subject transfer
   - Forecasting experiments

5. **Tutorial 5: Real-Time Inference**
   - neurOS pipeline integration
   - Online adaptation
   - Performance optimization

6. **Tutorial 6: Advanced Topics**
   - Custom tokenizers
   - Ablation studies
   - Multi-GPU training

**Documentation:**
- Architecture guide
- API reference
- Model cards
- Dataset cards
- Troubleshooting guide

**Deliverables:**
- ✅ 6 Jupyter notebooks
- ✅ Comprehensive documentation
- ✅ Model/dataset cards
- ✅ Example scripts

---

## Testing Strategy

### Unit Tests
```python
# neurofm/tests/unit/
test_tokenizers.py       # Token generation correctness
test_mamba_backbone.py   # Forward/backward passes
test_perceiver.py        # Multi-modal fusion
test_popt.py            # Population aggregation
test_diffusion.py       # Generative modeling
test_adapters.py        # Transfer components
```

### Integration Tests
```python
# neurofm/tests/integration/
test_end_to_end.py      # Full training loop
test_nwb_pipeline.py    # Data loading → inference
test_checkpoint.py      # Save/load consistency
test_neuros_integration.py  # neurOS compatibility
```

### Regression Tests
```python
# neurofm/tests/regression/
test_performance.py     # Latency benchmarks
test_accuracy.py        # Known dataset scores
test_memory.py         # GPU memory usage
```

**Target:** 90%+ test coverage

---

## Performance Targets

| Metric | Target | Baseline | Status |
|--------|--------|----------|--------|
| **Pretraining throughput** | >1000 samples/s | 500 s/s | 🎯 |
| **Inference latency** | <50ms | 100ms | 🎯 |
| **Few-shot accuracy** | >0.80 R² | 0.65 R² | 🎯 |
| **Transfer efficiency** | <10% accuracy drop | 30% drop | 🎯 |
| **Forecasting NLL** | <3.0 bits/spike | 5.0 b/s | 🎯 |
| **Sequence length** | 1M timesteps | 10K steps | 🎯 |
| **Memory (pretrain)** | <24GB | 40GB | 🎯 |

---

## Datasets

### Primary Datasets:

1. **IBL Repeated Site**
   - Motor cortex recordings
   - Multiple sessions per subject
   - Behavioral tasks

2. **Allen Brain Observatory**
   - Visual cortex calcium imaging
   - Neuropixels recordings
   - Standard stimuli

3. **DANDI Archive**
   - Public iEEG datasets
   - Speech/handwriting tasks
   - Multi-area recordings

### Data Processing:
- Convert all to NWB format
- Standardize metadata
- Create train/val/test splits
- Generate both event and binned views

---

## Computational Requirements

### Development:
- **CPU:** Testing, small experiments
- **1x A100 (40GB):** Pretraining on subsets
- **4x A100/H100:** Full-scale pretraining

### Inference:
- **CPU:** Real-time BCI (<50ms latency)
- **GPU:** Batch processing, forecasting

### Optimization:
- Mixed precision (bf16)
- Gradient accumulation
- Distributed data parallel (DDP)
- Checkpoint sharding

---

## Timeline Summary

| Weeks | Phase | Deliverable |
|-------|-------|-------------|
| 1-2 | Foundation | Package + data pipeline |
| 3 | Tokenizers | Neural data encoders |
| 4-5 | SSM Backbone | Mamba implementation |
| 6 | Fusion | Perceiver-IO |
| 7 | PopT | Population aggregator |
| 8 | Diffusion | Generative prior |
| 9 | Heads | Multi-task outputs |
| 10 | Adapters | Transfer learning |
| 11-12 | Training | Full pipeline |
| 13 | Evaluation | Benchmarks |
| 14 | Integration | neurOS compatibility |
| 15-16 | Docs | Tutorials + guides |

**Total: 16 weeks** for complete implementation

---

## Success Criteria

### Technical:
- ✅ All unit tests pass (90%+ coverage)
- ✅ Smoke runs complete on CPU
- ✅ Full pretrain converges on small dataset
- ✅ FALCON benchmark shows few-shot gains
- ✅ Cross-subject transfer works with adapters
- ✅ Deterministic inference verified

### Scientific:
- ✅ Match or exceed POYO/NDT baselines
- ✅ Demonstrate OOD generalization
- ✅ Show efficiency gains (5x vs Transformer)
- ✅ Realistic neural generation (bits-per-spike)

### Engineering:
- ✅ Export to ONNX/TorchScript
- ✅ Real-time neurOS integration
- ✅ Comprehensive documentation
- ✅ Reproducible experiments

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| SSM implementation complexity | Use `mamba-ssm` library, fallback to attention |
| Data availability | Start with public IBL/Allen, expand later |
| Computational cost | Progressive training, smaller ablations first |
| Integration issues | Early neurOS compatibility tests |
| Performance gaps | Systematic ablations to identify bottlenecks |

---

## Next Steps

**Immediate Actions:**
1. ✅ Set up `packages/neuros-neurofm/` directory structure
2. ✅ Install dependencies (`mamba-ssm`, `pytorch-lightning`)
3. ✅ Implement base NWB data loaders
4. ✅ Create configuration system (Hydra)
5. ✅ Write first unit tests

**This Week:**
- Build neural tokenizers
- Implement Mamba backbone skeleton
- Set up training infrastructure
- Create initial tutorials

---

**Let's build the future of neural foundation models! 🧠🚀**
