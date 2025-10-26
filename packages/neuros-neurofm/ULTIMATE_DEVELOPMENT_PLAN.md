# NeuroFMX Ultimate Development Plan
## Parallel Development Roadmap for Foundation Model Excellence

**Version:** 1.0
**Date:** 2025-10-25
**Status:** Production-Ready Foundation (85-90% Complete)

---

## Executive Summary

This plan synthesizes insights from:
1. **mechint_plan.xml** - Structured 10-phase development roadmap
2. **chatGPT_mechint_plan.md** - Current state analysis and cloud training blueprint
3. **Codebase exploration** - Comprehensive implementation status assessment

**Key Finding:** The NeuroFMX codebase is substantially complete with 40+ production-ready modules. We can immediately begin **parallel development** across 5 independent workstreams to achieve foundation model status within **6-8 weeks**.

---

## Current State Summary

### What's Production-Ready ✅
- **9+ Tokenizers** for all neural/behavioral modalities
- **Mamba SSM Backbone** with O(L) complexity
- **Perceiver-IO Fusion** for cross-modal integration
- **3 Loss Categories** (contrastive, adversarial, multi-task)
- **Advanced Interpretability** (4 modules, 40+ analysis functions)
- **Transfer Learning** (LoRA, adapters, few-shot, continual learning)
- **Full Training Pipeline** (PyTorch Lightning + config-driven)
- **Cloud Infrastructure** (Kubernetes + Terraform for 2 providers)
- **Docker Deployment** (multi-stage, optimized)
- **Data Acquisition** (7 public datasets ready)

### What Needs Enhancement ⚠️
- **Distributed Training** - FSDP/DeepSpeed configuration (10% work)
- **Benchmark Suite** - Automated baseline comparisons (15% work)
- **Data Pipeline** - WebDataset sharding for scale (20% work)
- **Evaluation Matrix** - Cross-species/task transfer grid (25% work)
- **Mech-Int Expansion** - SAE features, dynamics analysis (30% work)

---

## Architecture: 5 Parallel Workstreams

The plan divides work into **5 independent workstreams** that can be developed in parallel by different team members or AI agents, with minimal cross-dependencies.

```
┌─────────────────────────────────────────────────────────────────┐
│                    WORKSTREAM ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  WS1: Infrastructure   WS2: Data Pipeline   WS3: Training      │
│  & Scaling             & Tokenization       Objectives         │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐   │
│  │ FSDP/DeepS   │     │ WebDataset   │     │ Curriculum   │   │
│  │ Ray Tune     │────▶│ Sharding     │────▶│ Multi-stage  │   │
│  │ Checkpoints  │     │ Lazy Loading │     │ Objective Mix│   │
│  └──────────────┘     └──────────────┘     └──────────────┘   │
│         │                     │                     │          │
│         │                     │                     │          │
│         ▼                     ▼                     ▼          │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              WS4: Mechanistic Interpretability           │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ SAE      │  │ Dynamics │  │ Patching │  │ Alignment│ │ │
│  │  │ Features │  │ Analysis │  │ & Probes │  │ CCA/RSA  │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│  └──────────────────────────────────────────────────────────┘ │
│                              │                                 │
│                              ▼                                 │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │         WS5: Evaluation & Benchmarking Matrix            │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │ │
│  │  │ Cross-   │  │ Baseline │  │ Transfer │  │ Few-shot │ │ │
│  │  │ Species  │  │ Comparion│  │ Matrices │  │ Eval     │ │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Workstream 1: Infrastructure & Scaling (Weeks 1-3)

**Owner:** DevOps/Infra Lead
**Dependencies:** None (can start immediately)
**Goal:** Enable distributed training at H100 scale with fault tolerance

### Phase 1.1: Distributed Training Configuration (Week 1)
**Deliverables:**
- [ ] `configs/distributed/fsdp.yaml` - PyTorch FSDP configuration
- [ ] `configs/distributed/deepspeed.yaml` - DeepSpeed ZeRO-3 configuration
- [ ] `training/fsdp_trainer.py` - FSDP-wrapped training loop
- [ ] `training/deepspeed_trainer.py` - DeepSpeed integration

**Implementation Tasks:**
```python
# configs/distributed/fsdp.yaml
strategy:
  name: fsdp
  cpu_offload: false
  mixed_precision: bf16
  activation_checkpointing: true
  sharding_strategy: FULL_SHARD  # ZeRO-3 equivalent
  backward_prefetch: BACKWARD_PRE

# Integrate with existing training/train.py
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
```

**Tests:**
- [ ] `tests/test_fsdp_training.py` - Multi-GPU shape validation
- [ ] `tests/test_deepspeed_training.py` - ZeRO-3 gradient flow

### Phase 1.2: Ray Tune Hyperparameter Search (Week 2)
**Deliverables:**
- [ ] `optimization/ray_tune_search.py` - Replace existing `hyperparameter_search.py`
- [ ] `configs/tune/search_spaces.yaml` - Hyperparameter bounds
- [ ] `reports/scaling_laws.ipynb` - Analysis notebook

**Implementation Tasks:**
```python
from ray import tune
from ray.tune.schedulers import ASHAScheduler, PopulationBasedTraining

search_space = {
    "model.d_model": tune.choice([256, 512, 1024]),
    "model.n_layers": tune.randint(4, 16),
    "model.mamba_d_state": tune.choice([16, 32, 64]),
    "training.lr": tune.loguniform(1e-5, 1e-3),
    "training.fusion_freq": tune.choice([1, 2, 4]),
    "losses.mask_ratio": tune.uniform(0.15, 0.75),
}

scheduler = ASHAScheduler(
    time_attr="training_iteration",
    max_t=100,
    grace_period=10,
    reduction_factor=3,
)
```

**Tests:**
- [ ] `tests/test_ray_tune.py` - Mock trials, early stopping

### Phase 1.3: Checkpointing & Resumability (Week 3)
**Deliverables:**
- [ ] `training/checkpoint_manager.py` - Unified checkpoint handler
- [ ] `datasets/resumable_iterator.py` - Shard cursor tracking
- [ ] `configs/checkpoint_policy.yaml` - Save frequency, retention

**Implementation Tasks:**
```python
class CheckpointManager:
    def __init__(self, checkpoint_dir, save_every_n_steps=1000):
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps

    def save(self, model, optimizer, scheduler, global_step, data_cursor):
        """Save model + optimizer + data cursor for exact resumption"""
        torch.save({
            'global_step': global_step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'data_cursor': data_cursor,  # shard_idx, sample_offset
        }, f"{self.checkpoint_dir}/step_{global_step}.pt")
```

**Tests:**
- [ ] `tests/test_checkpoint_resume.py` - Training resumption validation

---

## Workstream 2: Data Pipeline & Tokenization (Weeks 1-4)

**Owner:** Data Engineering Lead
**Dependencies:** None
**Goal:** Scalable multi-modal data pipeline with WebDataset sharding

### Phase 2.1: WebDataset Sharding (Weeks 1-2)
**Deliverables:**
- [ ] `datasets/webdataset_writer.py` - Shard creation utilities
- [ ] `datasets/webdataset_loader.py` - Iterable dataset with resumption
- [ ] `scripts/convert_to_shards.py` - Convert NWB → WebDataset
- [ ] `docs/DATA_REGISTRY.md` - Dataset catalog and schema

**Implementation Tasks:**
```python
import webdataset as wds

class NeuroWebDatasetWriter:
    def __init__(self, output_dir, shard_size=1000):
        self.output_dir = output_dir
        self.shard_size = shard_size

    def write_shards(self, dataset, modalities):
        """Convert dataset to WebDataset tar shards"""
        for shard_idx, samples in enumerate(self.batch_samples(dataset)):
            with wds.TarWriter(f"{self.output_dir}/shard_{shard_idx:06d}.tar") as sink:
                for sample_id, data in samples:
                    sample_dict = {"__key__": sample_id}
                    for modality in modalities:
                        sample_dict[f"{modality}.pyd"] = data[modality]
                    sink.write(sample_dict)
```

**Shard Structure:**
```
s3://neurofmx/shards/
├── eeg/
│   ├── train/
│   │   ├── shard_000000.tar  (1000 samples, ~50MB)
│   │   ├── shard_000001.tar
│   │   └── ...
│   └── val/
├── ecog/
├── spikes/
├── lfp/
├── fmri/
├── video/
└── multimodal/  # Pre-aligned multi-modal samples
```

**Tests:**
- [ ] `tests/test_webdataset_io.py` - Shard write/read validation
- [ ] `tests/test_data_resumption.py` - Cursor-based resumption

### Phase 2.2: Enhanced Tokenizers with Temporal Anchors (Weeks 2-3)
**Deliverables:**
- [ ] Update all tokenizers to export `(tokens, t0, dt, mask)` format
- [ ] `tokenizers/temporal_alignment.py` - Cross-modal time alignment
- [ ] `tests/test_temporal_alignment.py` - Alignment validation

**Implementation Tasks:**
```python
# All tokenizers should return this format:
@dataclass
class TokenizedSequence:
    tokens: torch.Tensor        # (B, T, D)
    t0: float                   # Start time (seconds)
    dt: float                   # Sampling interval (seconds)
    mask: torch.Tensor          # (B, T) - valid positions
    metadata: dict              # Modality-specific info

# Example: EEGTokenizer
def forward(self, x, timestamps):
    tokens = self.conv_layers(x)
    return TokenizedSequence(
        tokens=tokens,
        t0=timestamps[0],
        dt=self.sampling_rate,
        mask=self.create_mask(x),
        metadata={"n_channels": x.shape[1], "band_powers": self.compute_bands(x)}
    )
```

**Tests:**
- [ ] `tests/test_tokenizer_contracts.py` - Shape/type validation for all 9 tokenizers

### Phase 2.3: Extended Dataset Coverage (Week 4)
**Deliverables:**
- [ ] `datasets/eye_tracking.py` - Eye-tracking loader
- [ ] `datasets/pose.py` - DeepLabCut/SLEAP pose data
- [ ] `datasets/physio.py` - HRV/EDA/respiration
- [ ] `datasets/task_metadata.py` - Task annotations/labels

**Dataset Additions:**
- **Eye-tracking:** Fixations, saccades, pupil diameter
- **Pose:** Joint positions, velocities, accelerations
- **Physio:** Heart rate variability, skin conductance
- **Task metadata:** Trial types, outcomes, timestamps

**Tests:**
- [ ] `tests/test_new_tokenizers.py` - Eye/pose/physio tokenizer validation

---

## Workstream 3: Training Objectives & Curriculum (Weeks 1-4)

**Owner:** ML Research Lead
**Dependencies:** WS2 Phase 2.2 (tokenizer format)
**Goal:** Multi-stage training curriculum with diverse objectives

### Phase 3.1: Enhanced Loss Functions (Weeks 1-2)
**Deliverables:**
- [ ] `losses/masked_modeling.py` - Per-modality masked token modeling
- [ ] `losses/forecasting.py` - Multi-horizon predictive coding
- [ ] `losses/diffusion.py` - Denoising diffusion prior for neural segments
- [ ] `losses/loss_registry.py` - Unified loss configuration

**Implementation Tasks:**
```python
# losses/masked_modeling.py
class MaskedModalityLoss(nn.Module):
    def __init__(self, modality, mask_ratio=0.15, mask_strategy='random'):
        super().__init__()
        self.modality = modality
        self.mask_ratio = mask_ratio
        self.mask_strategy = mask_strategy  # random, block, adaptive

    def forward(self, tokens, reconstructed, mask):
        """Mask tokens and predict them"""
        masked_tokens, mask_indices = self.apply_mask(tokens)
        loss = F.mse_loss(reconstructed[mask_indices], tokens[mask_indices])
        return loss

# losses/forecasting.py
class MultiHorizonForecastLoss(nn.Module):
    def __init__(self, horizons=[100, 250, 500, 1000]):  # milliseconds
        super().__init__()
        self.horizons = horizons

    def forward(self, current_state, future_states, dt):
        """Predict multiple future time points"""
        losses = []
        for horizon in self.horizons:
            steps_ahead = int(horizon / dt)
            pred = self.forecast_head(current_state, steps_ahead)
            loss = F.mse_loss(pred, future_states[steps_ahead])
            losses.append(loss)
        return sum(losses) / len(losses)
```

**Tests:**
- [ ] `tests/test_masked_modeling.py` - Masking strategies validation
- [ ] `tests/test_forecasting.py` - Multi-horizon prediction accuracy

### Phase 3.2: Training Curriculum (Weeks 2-3)
**Deliverables:**
- [ ] `configs/curriculum/unimodal.yaml` - Stage 1: Single modalities
- [ ] `configs/curriculum/pairwise.yaml` - Stage 2: Pairwise fusion
- [ ] `configs/curriculum/multimodal.yaml` - Stage 3: Full multimodal
- [ ] `training/curriculum_scheduler.py` - Stage transition logic

**Curriculum Stages:**
```yaml
# Stage 1: Unimodal (Weeks 1-2 of training)
stage_1:
  name: unimodal
  modalities: [eeg, ecog, lfp, spikes, fmri]
  objectives:
    - masked_modeling: 1.0
    - reconstruction: 0.5
  fusion: none

# Stage 2: Pairwise (Weeks 3-4)
stage_2:
  name: pairwise
  modality_pairs:
    - [eeg, video]
    - [spikes, lfp]
    - [fmri, task_metadata]
  objectives:
    - masked_modeling: 0.5
    - contrastive_alignment: 1.0
    - reconstruction: 0.3
  fusion: early

# Stage 3: Full Multimodal (Weeks 5+)
stage_3:
  name: multimodal
  modalities: all
  objectives:
    - masked_modeling: 0.3
    - contrastive_alignment: 0.8
    - forecasting: 0.5
    - diffusion: 0.2
  fusion: mid+late
```

**Tests:**
- [ ] `tests/test_curriculum.py` - Stage transitions, objective weights

### Phase 3.3: Modality Dropout & Augmentation (Week 4)
**Deliverables:**
- [ ] `augmentation/modality_dropout.py` - Random modality exclusion
- [ ] `augmentation/neural_augment.py` - Time/channel masking, SpecAugment
- [ ] `configs/augmentation.yaml` - Augmentation policies

**Implementation Tasks:**
```python
class ModalityDropout(nn.Module):
    def __init__(self, dropout_prob=0.1):
        super().__init__()
        self.dropout_prob = dropout_prob

    def forward(self, modality_dict):
        """Randomly drop entire modalities during training"""
        if not self.training:
            return modality_dict

        available_modalities = list(modality_dict.keys())
        num_to_keep = max(1, int(len(available_modalities) * (1 - self.dropout_prob)))
        kept_modalities = random.sample(available_modalities, num_to_keep)
        return {k: v for k, v in modality_dict.items() if k in kept_modalities}

class NeuralAugmentation(nn.Module):
    """SpecAugment-style for neural signals"""
    def __init__(self, time_mask_param=20, channel_mask_param=5):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.channel_mask_param = channel_mask_param

    def forward(self, x):
        # Time masking: zero out random time segments
        # Channel masking: zero out random channels
        # Gaussian noise injection
        return augmented_x
```

**Tests:**
- [ ] `tests/test_augmentation.py` - Augmentation effect validation

---

## Workstream 4: Mechanistic Interpretability Suite (Weeks 1-6)

**Owner:** Interpretability Research Lead
**Dependencies:** Minimal (works with existing models)
**Goal:** Comprehensive mech-int toolkit for understanding learned representations

### Phase 4.1: Sparse Autoencoders (SAE) for Feature Discovery (Weeks 1-2)
**Deliverables:**
- [ ] `interpretability/sae_training.py` - SAE training on frozen model
- [ ] `interpretability/sae_visualization.py` - Feature dictionary visualizations
- [ ] `interpretability/feature_analysis.py` - Feature attribution and ranking
- [ ] `reports/sae_features/` - Per-modality feature catalogs

**Implementation Tasks:**
```python
class SparseAutoencoder(nn.Module):
    """Learn sparse overcomplete feature dictionary"""
    def __init__(self, input_dim, hidden_dim, sparsity_coef=1e-3):
        super().__init__()
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.sparsity_coef = sparsity_coef

    def forward(self, x):
        h = F.relu(self.encoder(x))  # Sparse activations
        x_hat = self.decoder(h)

        recon_loss = F.mse_loss(x_hat, x)
        sparsity_loss = self.sparsity_coef * h.abs().mean()
        return x_hat, recon_loss + sparsity_loss, h

def train_sae_on_layer(model, layer_name, dataset):
    """Train SAE on specific layer's activations"""
    activations = []

    # Collect activations
    def hook_fn(module, input, output):
        activations.append(output.detach())

    handle = model.get_submodule(layer_name).register_forward_hook(hook_fn)
    # ... training loop ...
    handle.remove()
```

**Visualizations:**
- Feature activation heatmaps over time
- Top-activating samples per feature
- Feature co-occurrence matrices
- Cross-modality feature alignment

**Tests:**
- [ ] `tests/test_sae.py` - Sparsity, reconstruction quality

### Phase 4.2: Activation Patching & Causal Tracing (Weeks 2-3)
**Deliverables:**
- [ ] `interpretability/patching.py` - Extended from `circuit_discovery.py`
- [ ] `interpretability/causal_tracing.py` - Interventional analysis
- [ ] `interpretability/ablation_suite.py` - Head/block ablations
- [ ] `reports/circuit_analysis/` - Automated circuit reports

**Implementation Tasks:**
```python
class ActivationPatcher:
    def __init__(self, model):
        self.model = model
        self.cache = {}

    def run_with_patch(self, input_clean, input_corrupted, layer, pos):
        """Replace activations from corrupted run with clean run"""
        # 1. Forward pass on clean input, cache activations
        with self.cache_activations(layer):
            output_clean = self.model(input_clean)

        # 2. Forward pass on corrupted input, patch at layer/pos
        def patch_hook(module, input, output):
            output[:, pos] = self.cache[layer][:, pos]
            return output

        handle = self.model.get_submodule(layer).register_forward_hook(patch_hook)
        output_patched = self.model(input_corrupted)
        handle.remove()

        # 3. Measure effect on downstream task
        return self.measure_effect(output_clean, output_patched)

def generate_circuit_report(model, dataset, task):
    """Automatically identify minimal circuits for task"""
    patcher = ActivationPatcher(model)

    importance_matrix = np.zeros((model.n_layers, model.d_model))

    for layer in range(model.n_layers):
        for head in range(model.n_heads):
            effect = patcher.ablate_head(layer, head, dataset, task)
            importance_matrix[layer, head] = effect

    # Plot heatmap, identify minimal circuit
    minimal_circuit = find_minimal_subgraph(importance_matrix, threshold=0.1)
    return minimal_circuit, importance_matrix
```

**Tests:**
- [ ] `tests/test_patching.py` - Patch fidelity, effect measurement

### Phase 4.3: Model-to-Brain Alignment (Weeks 3-4)
**Deliverables:**
- [ ] `interpretability/alignment/cca.py` - Canonical Correlation Analysis
- [ ] `interpretability/alignment/rsa.py` - Representational Similarity Analysis
- [ ] `interpretability/alignment/pls.py` - Partial Least Squares
- [ ] `interpretability/alignment/metrics.py` - Noise ceiling correction

**Implementation Tasks:**
```python
from sklearn.cross_decomposition import CCA

class ModelBrainAlignment:
    def __init__(self, n_components=10):
        self.cca = CCA(n_components=n_components)

    def align_representations(self, model_activations, neural_data):
        """Compute CCA between model and brain representations"""
        # model_activations: (n_samples, n_model_units)
        # neural_data: (n_samples, n_brain_units)

        self.cca.fit(model_activations, neural_data)

        # Transform both to shared space
        model_canonical, brain_canonical = self.cca.transform(
            model_activations, neural_data
        )

        # Compute canonical correlations
        correlations = [
            np.corrcoef(model_canonical[:, i], brain_canonical[:, i])[0, 1]
            for i in range(self.cca.n_components)
        ]

        return correlations, model_canonical, brain_canonical

def compute_noise_ceiling(neural_data, num_splits=10):
    """Estimate maximum achievable correlation given neural noise"""
    # Split-half reliability
    correlations = []
    for _ in range(num_splits):
        half1, half2 = split_trials(neural_data)
        r = np.corrcoef(half1.mean(axis=0), half2.mean(axis=0))[0, 1]
        correlations.append(r)

    noise_ceiling = np.mean(correlations)
    return noise_ceiling
```

**Metrics:**
- CCA canonical correlations
- RSA correlation (model RDM vs brain RDM)
- Noise-ceiling normalized scores
- Layer-wise alignment profiles

**Tests:**
- [ ] `tests/test_alignment.py` - Synthetic data with known alignment

### Phase 4.4: Dynamical Systems Analysis (Weeks 4-5)
**Deliverables:**
- [ ] `interpretability/dynamics/koopman.py` - Koopman operator estimation
- [ ] `interpretability/dynamics/lyapunov.py` - Stability analysis
- [ ] `interpretability/dynamics/manifold.py` - Slow manifold identification
- [ ] `interpretability/dynamics/controllability.py` - Control analysis

**Implementation Tasks:**
```python
class DynamicalAnalysis:
    def __init__(self, model):
        self.model = model

    def estimate_koopman_operator(self, trajectories):
        """Estimate linear operator for nonlinear dynamics"""
        # Collect state transitions: x[t] -> x[t+1]
        X = trajectories[:, :-1]  # (n_trials, T-1, d_model)
        Y = trajectories[:, 1:]   # (n_trials, T-1, d_model)

        # Solve: K @ X ≈ Y
        K = torch.linalg.lstsq(X.reshape(-1, X.shape[-1]),
                                Y.reshape(-1, Y.shape[-1])).solution

        # Eigendecomposition for mode analysis
        eigenvalues, eigenvectors = torch.linalg.eig(K)

        return K, eigenvalues, eigenvectors

    def compute_lyapunov_exponents(self, trajectories, dt=0.01):
        """Estimate local Lyapunov exponents"""
        # Measure divergence of nearby trajectories
        exponents = []

        for traj in trajectories:
            # Perturb initial condition slightly
            traj_perturbed = traj + torch.randn_like(traj) * 1e-3

            # Measure exponential divergence rate
            distance = torch.norm(traj - traj_perturbed, dim=-1)
            log_distance = torch.log(distance + 1e-8)

            # Fit exponential: log(d) = λt
            t = torch.arange(len(traj)) * dt
            lambda_estimate = torch.polyfit(t, log_distance, deg=1)[0]
            exponents.append(lambda_estimate)

        return torch.tensor(exponents).mean()

    def identify_slow_manifold(self, trajectories):
        """Find low-dimensional slow manifold via PCA"""
        # Flatten all trajectories
        all_states = trajectories.reshape(-1, trajectories.shape[-1])

        # PCA to find slow modes
        U, S, V = torch.pca_lowrank(all_states, q=10)

        # Slow modes = top eigenvectors
        slow_modes = V[:, :3]  # 3D manifold

        return slow_modes, S
```

**Visualizations:**
- Koopman mode decomposition
- Phase portraits in slow manifold
- Lyapunov spectrum
- Controllability/observability matrices

**Tests:**
- [ ] `tests/test_dynamics.py` - Known dynamical systems (Lorenz, etc.)

### Phase 4.5: Cross-Domain Concept Mining (Week 6)
**Deliverables:**
- [ ] `interpretability/concept_mining.py` - Shared feature discovery
- [ ] `interpretability/transfer_features.py` - Cross-task/species features
- [ ] `reports/concept_catalog/` - Ranked feature library

**Implementation Tasks:**
```python
def mine_shared_concepts(model, datasets_dict):
    """Find features that transfer across tasks/species"""
    # datasets_dict = {'mouse_reach': dataset1, 'monkey_reach': dataset2, ...}

    # 1. Train SAEs on each dataset
    saes = {}
    for name, dataset in datasets_dict.items():
        sae = train_sae_on_layer(model, 'backbone.layers.6', dataset)
        saes[name] = sae

    # 2. Align feature spaces across datasets (CCA/Procrustes)
    reference_features = saes['mouse_reach'].encoder.weight

    aligned_features = {}
    for name, sae in saes.items():
        W = sae.encoder.weight
        # Procrustes alignment: find rotation R
        U, _, Vt = torch.linalg.svd(W.T @ reference_features)
        R = U @ Vt
        aligned_features[name] = W @ R

    # 3. Cluster aligned features to find shared concepts
    all_features = torch.cat(list(aligned_features.values()), dim=0)
    from sklearn.cluster import AgglomerativeClustering

    clustering = AgglomerativeClustering(n_clusters=100)
    labels = clustering.fit_predict(all_features.detach().cpu().numpy())

    # 4. Rank clusters by causal importance across tasks
    concept_scores = []
    for cluster_id in range(100):
        cluster_features = [i for i, l in enumerate(labels) if l == cluster_id]

        # Measure downstream task performance when ablating cluster
        importance = measure_cluster_importance(model, cluster_features, datasets_dict)
        concept_scores.append(importance)

    return concept_scores, labels, aligned_features
```

**Tests:**
- [ ] `tests/test_concept_mining.py` - Synthetic multi-task data

---

## Workstream 5: Evaluation & Benchmarking Matrix (Weeks 2-6)

**Owner:** Evaluation Lead
**Dependencies:** WS1 (trained models), WS3 (objectives)
**Goal:** Comprehensive evaluation across species, tasks, and modalities

### Phase 5.1: Evaluation Task Registry (Week 2)
**Deliverables:**
- [ ] `evaluation/task_registry.py` - Unified task registration system
- [ ] `configs/eval/eval_tasks.yaml` - Task definitions
- [ ] `evaluation/metrics_library.py` - Extended metrics beyond current metrics.py

**Implementation Tasks:**
```yaml
# configs/eval/eval_tasks.yaml
tasks:
  - name: mouse_reach_decoding
    species: mouse
    modality: [spikes, lfp]
    target: kinematics
    metric: r2_score
    splits: [train, val, test]
    n_trials: 100

  - name: human_eeg_sleep_staging
    species: human
    modality: [eeg]
    target: sleep_stage
    metric: accuracy
    splits: [train, val, test]
    n_trials: 500

  - name: monkey_fmri_image_encoding
    species: monkey
    modality: [fmri]
    target: image_features
    metric: encoding_r2
    splits: [train, val, test]
    n_trials: 200
```

**Task Categories:**
1. **Neural Reconstruction**: Predict neural activity from other modalities
2. **Behavior Decoding**: Predict behavior from neural activity
3. **Encoding Models**: Predict neural responses to stimuli
4. **State Classification**: Sleep stages, behavioral states, task epochs
5. **Forecasting**: Future neural/behavioral state prediction
6. **Cross-Modal Alignment**: Temporal synchronization across modalities

**Tests:**
- [ ] `tests/test_task_registry.py` - Task loading and validation

### Phase 5.2: Zero-Shot & Few-Shot Evaluation (Weeks 3-4)
**Deliverables:**
- [ ] `evaluation/zero_shot.py` - No fine-tuning evaluation
- [ ] `evaluation/few_shot_eval.py` - K-shot learning curves (K=1,5,10,25,50)
- [ ] `evaluation/transfer_matrix.py` - Cross-task/species transfer grid

**Implementation Tasks:**
```python
def evaluate_zero_shot(model, task):
    """Evaluate frozen model with linear probe"""
    # Extract frozen representations
    with torch.no_grad():
        representations = model.encode(task.data)

    # Train linear probe only
    probe = LinearProbe(input_dim=model.d_model, output_dim=task.n_classes)
    probe.fit(representations, task.labels)

    # Evaluate on test set
    test_reps = model.encode(task.test_data)
    predictions = probe(test_reps)

    return task.metric(predictions, task.test_labels)

def evaluate_few_shot(model, task, shots=[1, 5, 10, 25, 50]):
    """Evaluate with K-shot fine-tuning"""
    results = {}

    for k in shots:
        # Sample k examples per class
        support_set = task.sample_support(k)

        # Fine-tune with LoRA adapter
        adapter = LoRAAdapter(model)
        adapter.fit(support_set, max_steps=100)

        # Evaluate on test set
        predictions = adapter(task.test_data)
        results[f"{k}-shot"] = task.metric(predictions, task.test_labels)

    return results

def compute_transfer_matrix(model, tasks):
    """Compute cross-task transfer performance matrix"""
    n_tasks = len(tasks)
    transfer_matrix = np.zeros((n_tasks, n_tasks))

    for i, source_task in enumerate(tasks):
        # Fine-tune on source task
        model_finetuned = finetune(model, source_task)

        for j, target_task in enumerate(tasks):
            # Evaluate on target task
            score = evaluate_zero_shot(model_finetuned, target_task)
            transfer_matrix[i, j] = score

    return transfer_matrix, tasks
```

**Visualizations:**
- Transfer matrices (heatmaps)
- Few-shot learning curves
- Gap-to-supervised headroom analysis

**Tests:**
- [ ] `tests/test_few_shot_eval.py` - Synthetic few-shot tasks

### Phase 5.3: Baseline Comparisons (Weeks 4-5)
**Deliverables:**
- [ ] `baselines/cebra_baseline.py` - CEBRA comparison
- [ ] `baselines/lfads_baseline.py` - LFADS comparison
- [ ] `baselines/ndt_baseline.py` - Neural Data Transformer comparison
- [ ] `reports/baseline_comparison.md` - Performance comparison report

**Baselines to Compare:**
1. **CEBRA**: Contrastive learning for neural data
2. **LFADS**: Latent Factor Analysis via Dynamical Systems
3. **NDT**: Neural Data Transformer
4. **Standard supervised**: Task-specific models

**Metrics:**
- R² for regression tasks
- Accuracy for classification
- Bits-per-spike for neural reconstruction
- Inference speed (samples/sec)
- Model parameters
- Training time

**Tests:**
- [ ] `tests/test_baselines.py` - Baseline model loading and evaluation

### Phase 5.4: Automated Evaluation Pipeline (Week 6)
**Deliverables:**
- [ ] `evaluation/auto_eval.py` - Automated evaluation runner
- [ ] `reports/transfer_matrices/` - Generated heatmaps
- [ ] `reports/performance_cards/` - Per-task scorecards
- [ ] `reports/scaling_curves/` - Performance vs compute/data

**Implementation Tasks:**
```python
class AutoEvaluator:
    def __init__(self, model, task_registry, output_dir):
        self.model = model
        self.task_registry = task_registry
        self.output_dir = output_dir

    def run_full_evaluation(self):
        """Run all evaluations and generate reports"""
        results = {}

        # 1. Zero-shot evaluation
        for task in self.task_registry.tasks:
            results[task.name] = {
                'zero_shot': evaluate_zero_shot(self.model, task),
                'few_shot': evaluate_few_shot(self.model, task),
            }

        # 2. Transfer matrix
        transfer_matrix = compute_transfer_matrix(self.model, self.task_registry.tasks)

        # 3. Baseline comparison
        baseline_results = {}
        for baseline_name in ['cebra', 'lfads', 'ndt']:
            baseline_model = load_baseline(baseline_name)
            baseline_results[baseline_name] = evaluate_model(baseline_model, self.task_registry)

        # 4. Generate reports
        self.generate_performance_cards(results)
        self.generate_transfer_heatmap(transfer_matrix)
        self.generate_comparison_table(results, baseline_results)

        return results
```

**Report Outputs:**
- `performance_cards/{task}.md` - Individual task results
- `transfer_matrices/heatmap.png` - Cross-task transfer
- `baseline_comparison.csv` - Tabular comparison
- `scaling_curves.png` - Model size vs performance

**Tests:**
- [ ] `tests/test_auto_eval.py` - Pipeline execution

---

## Integration & Testing Strategy

### Continuous Integration Across Workstreams

**Weekly Integration Points:**
```
Week 1: WS1 + WS2 → Test distributed data loading
Week 2: WS2 + WS3 → Test curriculum with new tokenizers
Week 3: WS1 + WS3 → Test FSDP with new objectives
Week 4: WS4 + WS5 → Test interpretability on evaluation tasks
Week 5: All → Full system integration test
Week 6: All → End-to-end foundation model training
```

### Testing Pyramid

```
                    ┌──────────────────┐
                    │  E2E Tests (5%)  │
                    │  - Full training │
                    │  - Cloud deploy  │
                    └──────────────────┘
                 ┌─────────────────────────┐
                 │  Integration Tests (20%)│
                 │  - Multi-GPU training   │
                 │  - Data pipeline        │
                 │  - Eval suite          │
                 └─────────────────────────┘
          ┌───────────────────────────────────────┐
          │      Unit Tests (75%)                │
          │  - Tokenizer shapes                  │
          │  - Loss functions                    │
          │  - Model components                  │
          │  - Metrics                           │
          └───────────────────────────────────────┘
```

**Test Categories:**
1. **Unit Tests** (75% coverage target)
   - [ ] All tokenizers: shape, dtype, mask validation
   - [ ] All loss functions: gradient flow, numerical stability
   - [ ] Model components: forward/backward pass
   - [ ] Metrics: known ground truth validation

2. **Integration Tests** (20% coverage)
   - [ ] Multi-GPU training: FSDP, gradient synchronization
   - [ ] Data pipeline: WebDataset loading, resumption
   - [ ] Curriculum: stage transitions, objective weighting
   - [ ] Evaluation: task loading, metric computation

3. **End-to-End Tests** (5% coverage)
   - [ ] Full training run: quick_test.yaml (2-3 hours)
   - [ ] Cloud deployment: Kubernetes cluster provisioning
   - [ ] Inference: API server request/response
   - [ ] Interpretability: Full mech-int analysis pipeline

### Automated Testing Workflow

```bash
# Pre-commit hooks
pre-commit run --all-files

# Unit tests (run on every commit)
pytest tests/unit/ -v --cov=neuros_neurofm --cov-report=html

# Integration tests (run nightly)
pytest tests/integration/ -v --timeout=300

# E2E tests (run weekly)
pytest tests/e2e/ -v --timeout=3600

# Cloud tests (run on release)
pytest tests/cloud/ -v --cloud-provider=coreweave
```

---

## Timeline & Milestones

### 6-Week Development Plan

```
┌─────────────────────────────────────────────────────────────────┐
│                    GANTT CHART TIMELINE                         │
├─────────────────────────────────────────────────────────────────┤
│ Workstream               │ W1 │ W2 │ W3 │ W4 │ W5 │ W6 │       │
├─────────────────────────┼────┼────┼────┼────┼────┼────┤       │
│ WS1: Infrastructure      │████│████│████│    │    │    │       │
│   - FSDP/DeepSpeed      │████│    │    │    │    │    │       │
│   - Ray Tune            │    │████│    │    │    │    │       │
│   - Checkpointing       │    │    │████│    │    │    │       │
├─────────────────────────┼────┼────┼────┼────┼────┼────┤       │
│ WS2: Data Pipeline       │████│████│████│████│    │    │       │
│   - WebDataset          │████│████│    │    │    │    │       │
│   - Tokenizers          │    │████│████│    │    │    │       │
│   - New datasets        │    │    │    │████│    │    │       │
├─────────────────────────┼────┼────┼────┼────┼────┼────┤       │
│ WS3: Objectives          │████│████│████│████│    │    │       │
│   - New losses          │████│████│    │    │    │    │       │
│   - Curriculum          │    │████│████│    │    │    │       │
│   - Augmentation        │    │    │    │████│    │    │       │
├─────────────────────────┼────┼────┼────┼────┼────┼────┤       │
│ WS4: Interpretability    │████│████│████│████│████│████│       │
│   - SAE features        │████│████│    │    │    │    │       │
│   - Patching            │    │████│████│    │    │    │       │
│   - Brain alignment     │    │    │████│████│    │    │       │
│   - Dynamics            │    │    │    │████│████│    │       │
│   - Concept mining      │    │    │    │    │    │████│       │
├─────────────────────────┼────┼────┼────┼────┼────┼────┤       │
│ WS5: Evaluation          │    │████│████│████│████│████│       │
│   - Task registry       │    │████│    │    │    │    │       │
│   - Few-shot eval       │    │    │████│████│    │    │       │
│   - Baselines           │    │    │    │████│████│    │       │
│   - Auto pipeline       │    │    │    │    │    │████│       │
├─────────────────────────┼────┼────┼────┼────┼────┼────┤       │
│ Integration Testing      │    │    │    │    │████│████│       │
│ Full System Training     │    │    │    │    │    │████│       │
└─────────────────────────┴────┴────┴────┴────┴────┴────┘       │
                                                                 │
Legend: ████ = Active Development                               │
└─────────────────────────────────────────────────────────────────┘
```

### Key Milestones

**Week 1:**
- ✅ FSDP configuration complete
- ✅ WebDataset shard writer implemented
- ✅ Masked modeling loss implemented
- ✅ SAE training pipeline functional

**Week 2:**
- ✅ Ray Tune hyperparameter search running
- ✅ All tokenizers export temporal anchors
- ✅ Training curriculum stages defined
- ✅ Activation patching suite complete
- ✅ Evaluation task registry operational

**Week 3:**
- ✅ Checkpoint resumption working
- ✅ Enhanced tokenizers with alignment
- ✅ Curriculum scheduler implemented
- ✅ Brain alignment metrics (CCA/RSA) functional
- ✅ Few-shot evaluation pipeline running

**Week 4:**
- ✅ New datasets integrated (eye, pose, physio)
- ✅ Augmentation policies active
- ✅ Dynamical analysis tools complete
- ✅ Baseline comparisons running

**Week 5:**
- ✅ Full integration testing across all workstreams
- ✅ Concept mining functional
- ✅ Automated evaluation pipeline operational

**Week 6:**
- ✅ End-to-end foundation model training (200+ sessions)
- ✅ Complete evaluation matrix generated
- ✅ All interpretability reports produced
- 🎯 **Foundation Model Release Ready**

---

## Cloud Training Blueprint

### Infrastructure Setup (Week 0, before development)

**Option 1: CoreWeave (Managed Kubernetes + H100 HGX)**
```bash
# 1. Apply Terraform configuration
cd infra/coreweave
terraform init
terraform apply

# 2. Deploy Ray cluster
kubectl apply -f infra/k8s/00-namespace.yaml
kubectl apply -f infra/k8s/01-nvidia-device-plugin.yaml
kubectl apply -f infra/k8s/10-kuberay-operator.yaml
kubectl apply -f infra/k8s/20-raycluster-neurofmx.yaml

# 3. Verify cluster
kubectl get pods -n neurofmx
# Expected: 1 head + 8 workers, all Running

# 4. Setup storage
kubectl apply -f infra/k8s/03-storage-pvc.yaml
# Creates 500GB checkpoint PVC + 2TB data PVC
```

**Option 2: Crusoe Cloud (H100 on-demand)**
```bash
cd infra/crusoe
terraform init
terraform apply

# Outputs Crusoe API credentials and cluster endpoint
```

### Training Configurations

**Small-Scale Test (RTX 3070 Ti, Local)**
```yaml
# configs/local_test.yaml
model:
  d_model: 256
  n_layers: 6
  n_heads: 4

training:
  batch_size: 8
  gradient_accumulation_steps: 4
  max_steps: 10000

data:
  num_sessions: 10
  modalities: [spikes, lfp]

# Expected time: 4-6 hours
# Expected memory: ~8GB GPU
```

**Medium-Scale Foundation (A100 x8, Cloud)**
```yaml
# configs/cloud_foundation.yaml
model:
  d_model: 1024
  n_layers: 16
  n_heads: 16

training:
  strategy: fsdp
  devices: 8
  batch_size: 32
  gradient_accumulation_steps: 2
  max_steps: 100000

data:
  num_sessions: 200
  modalities: [eeg, ecog, lfp, spikes, fmri, video, audio]

# Expected time: 24-40 hours
# Expected cost: $200-400 (depending on provider)
```

**Large-Scale Foundation (H100 HGX x8, Cloud)**
```yaml
# configs/cloud_h100_foundation.yaml
model:
  d_model: 2048
  n_layers: 24
  n_heads: 32

training:
  strategy: fsdp
  precision: bf16
  devices: 8
  batch_size: 64
  gradient_accumulation_steps: 1
  max_steps: 500000

data:
  num_sessions: 500+
  modalities: all

# Expected time: 48-80 hours
# Expected cost: $1000-2000
# Expected model size: 500M-1B parameters
```

### Cost Optimization

**Strategies:**
1. **Spot/Preemptible Instances**: 50-70% cost reduction
   - Use checkpoint resumption (WS1 Phase 1.3)
   - Expect ~1-2 interruptions per 24-hour run

2. **Mixed Precision (bf16)**: 2x throughput
   - Already implemented in Lightning module
   - Zero accuracy loss in testing

3. **Gradient Accumulation**: Larger effective batch sizes
   - Reduces communication overhead
   - Enables training on smaller GPU clusters

4. **Flash Attention**: 3-4x memory reduction
   - Already in dependencies
   - Enables larger models on same hardware

5. **Activation Checkpointing**: 40-60% memory reduction
   - Trade compute for memory
   - Enables 2x larger models

---

## Success Metrics

### Foundation Model Quality

**Core Performance Targets:**
| Metric | Baseline | Target | Stretch |
|--------|----------|--------|---------|
| **Neural Reconstruction R²** | 0.30 | 0.50 | 0.65 |
| **Behavior Decoding Accuracy** | 65% | 80% | 90% |
| **Cross-Species Transfer** | 0.40 | 0.60 | 0.75 |
| **Few-Shot (5-shot) Accuracy** | 55% | 75% | 85% |
| **Zero-Shot Transfer** | 0.25 | 0.45 | 0.60 |

**Interpretability Targets:**
| Metric | Target |
|--------|--------|
| **SAE Feature Sparsity** | <5% activations per feature |
| **Circuit Ablation Effect** | >20% performance drop for critical circuits |
| **Brain Alignment (CCA)** | >0.60 correlation with V1 for visual tasks |
| **Noise Ceiling Achievement** | >80% of noise ceiling |

**System Performance Targets:**
| Metric | Target |
|--------|--------|
| **Training Throughput** | >1000 samples/sec on A100 x8 |
| **Inference Latency** | <10ms per sample (GPU) |
| **Model Size** | 150M-1B parameters |
| **Training Time** | <48 hours to foundation model |

---

## Risk Mitigation

### Technical Risks

**Risk 1: WebDataset Sharding Complexity**
- *Probability:* Medium
- *Impact:* High (blocks distributed training)
- *Mitigation:*
  - Start with small shards (1000 samples)
  - Test resumption early (Week 1)
  - Fallback to existing NWB loader if needed

**Risk 2: FSDP Communication Overhead**
- *Probability:* Medium
- *Impact:* Medium (slower training)
- *Mitigation:*
  - Profile with PyTorch profiler
  - Tune `sharding_strategy` (FULL_SHARD vs SHARD_GRAD_OP)
  - Use gradient accumulation to reduce sync frequency

**Risk 3: Multi-Modal Alignment Failure**
- *Probability:* Low
- *Impact:* High (degrades cross-modal learning)
- *Mitigation:*
  - Validate temporal alignment in unit tests
  - Start with well-aligned modalities (video + spikes)
  - Use curriculum to stabilize early training

**Risk 4: Interpretability Tools Scalability**
- *Probability:* Medium
- *Impact:* Low (nice-to-have features)
- *Mitigation:*
  - Use activation checkpointing for memory
  - Run interpretability on smaller models first
  - Parallelize SAE training across layers

### Operational Risks

**Risk 5: Cloud Cost Overruns**
- *Probability:* Medium
- *Impact:* High (budget constraints)
- *Mitigation:*
  - Set billing alerts ($500, $1000, $1500)
  - Use spot instances with auto-resumption
  - Run small-scale tests first (quick_test.yaml)
  - Monitor GPU utilization (target >80%)

**Risk 6: Data Acquisition Failures**
- *Probability:* Low
- *Impact:* Medium (delays training)
- *Mitigation:*
  - Test all 7 download scripts in Week 0
  - Have fallback datasets (synthetic data)
  - Pre-download critical datasets to S3

---

## Resource Requirements

### Team Allocation

**Option A: 5-Person Team (6 weeks)**
- **WS1 Lead:** DevOps engineer (distributed systems)
- **WS2 Lead:** Data engineer (data pipelines, NWB)
- **WS3 Lead:** ML researcher (self-supervised learning)
- **WS4 Lead:** Interpretability researcher (mech-int)
- **WS5 Lead:** Evaluation engineer (benchmarking)

**Option B: 2-Person Team + AI Agents (8 weeks)**
- **Lead 1:** ML researcher (WS1 + WS3)
- **Lead 2:** Interpretability engineer (WS4 + WS5)
- **AI Agents:** WS2 (data pipeline - well-defined tasks)

**Option C: 1-Person + Heavy AI Agent Use (10 weeks)**
- **Solo developer** coordinates all workstreams
- **AI agents** handle 60-70% of implementation
- **Human** focuses on architecture decisions, testing, debugging

### Compute Requirements

**Development Phase (Weeks 1-5):**
- **Local:** RTX 3070 Ti or better (for quick tests)
- **Cloud:** Minimal (1-2 A100s for integration tests)
- **Estimated cost:** $50-100

**Training Phase (Week 6):**
- **Option 1:** 8x A100 (80GB) for 40 hours = $800-1200
- **Option 2:** 8x H100 HGX for 24 hours = $1200-1800
- **Storage:** S3/GCS for sharded data (~500GB) = $10-20/month

**Total Budget Estimate:** $1500-2500 for full foundation model

---

## Deliverables Checklist

### Code Deliverables (50+ files)

**Infrastructure (WS1):**
- [ ] `configs/distributed/fsdp.yaml`
- [ ] `configs/distributed/deepspeed.yaml`
- [ ] `training/fsdp_trainer.py`
- [ ] `training/deepspeed_trainer.py`
- [ ] `optimization/ray_tune_search.py`
- [ ] `training/checkpoint_manager.py`
- [ ] `datasets/resumable_iterator.py`

**Data Pipeline (WS2):**
- [ ] `datasets/webdataset_writer.py`
- [ ] `datasets/webdataset_loader.py`
- [ ] `scripts/convert_to_shards.py`
- [ ] `tokenizers/temporal_alignment.py`
- [ ] `datasets/eye_tracking.py`
- [ ] `datasets/pose.py`
- [ ] `datasets/physio.py`

**Training Objectives (WS3):**
- [ ] `losses/masked_modeling.py`
- [ ] `losses/forecasting.py`
- [ ] `losses/diffusion.py`
- [ ] `losses/loss_registry.py`
- [ ] `training/curriculum_scheduler.py`
- [ ] `augmentation/modality_dropout.py`
- [ ] `augmentation/neural_augment.py`

**Interpretability (WS4):**
- [ ] `interpretability/sae_training.py`
- [ ] `interpretability/sae_visualization.py`
- [ ] `interpretability/feature_analysis.py`
- [ ] `interpretability/patching.py`
- [ ] `interpretability/causal_tracing.py`
- [ ] `interpretability/ablation_suite.py`
- [ ] `interpretability/alignment/cca.py`
- [ ] `interpretability/alignment/rsa.py`
- [ ] `interpretability/alignment/pls.py`
- [ ] `interpretability/dynamics/koopman.py`
- [ ] `interpretability/dynamics/lyapunov.py`
- [ ] `interpretability/dynamics/manifold.py`
- [ ] `interpretability/concept_mining.py`

**Evaluation (WS5):**
- [ ] `evaluation/task_registry.py`
- [ ] `evaluation/zero_shot.py`
- [ ] `evaluation/few_shot_eval.py`
- [ ] `evaluation/transfer_matrix.py`
- [ ] `baselines/cebra_baseline.py`
- [ ] `baselines/lfads_baseline.py`
- [ ] `baselines/ndt_baseline.py`
- [ ] `evaluation/auto_eval.py`

**Tests (100+ test cases):**
- [ ] `tests/test_fsdp_training.py`
- [ ] `tests/test_deepspeed_training.py`
- [ ] `tests/test_ray_tune.py`
- [ ] `tests/test_webdataset_io.py`
- [ ] `tests/test_tokenizer_contracts.py`
- [ ] `tests/test_masked_modeling.py`
- [ ] `tests/test_curriculum.py`
- [ ] `tests/test_sae.py`
- [ ] `tests/test_patching.py`
- [ ] `tests/test_alignment.py`
- [ ] `tests/test_dynamics.py`
- [ ] `tests/test_task_registry.py`
- [ ] `tests/test_baselines.py`

### Documentation Deliverables

**Technical Documentation:**
- [ ] `docs/DATA_REGISTRY.md` - Dataset catalog
- [ ] `docs/TOKENIZER_SPEC.md` - Tokenizer contracts
- [ ] `docs/LOSS_FUNCTIONS.md` - Objective descriptions
- [ ] `docs/INTERPRETABILITY_GUIDE.md` - Mech-int toolkit usage
- [ ] `docs/EVALUATION_SUITE.md` - Benchmark descriptions
- [ ] `docs/DISTRIBUTED_TRAINING.md` - FSDP/DeepSpeed guide

**Research Artifacts:**
- [ ] `reports/scaling_laws.ipynb` - Scaling curve analysis
- [ ] `reports/sae_features/` - Feature visualizations
- [ ] `reports/circuit_analysis/` - Discovered circuits
- [ ] `reports/transfer_matrices/` - Cross-task heatmaps
- [ ] `reports/baseline_comparison.md` - Performance tables
- [ ] `reports/performance_cards/` - Per-task scorecards

**Deployment Guides:**
- [ ] `infra/CLOUD_DEPLOYMENT.md` - Step-by-step cloud setup
- [ ] `infra/COST_OPTIMIZATION.md` - Budget management
- [ ] `MODEL_CARD.md` - Pre-trained model card

---

## Post-Development Roadmap (Weeks 7-12)

### Week 7-8: Model Refinement
- Hyperparameter sweep with Ray Tune (search 1000+ configs)
- Curriculum optimization (adjust stage transition points)
- Data quality filtering (identify and remove low-quality sessions)

### Week 9-10: Advanced Features
- **Meta-Learning:** MAML for rapid task adaptation
- **Continual Learning:** EWC for sequential task learning
- **Model Distillation:** Compress to smaller inference models

### Week 11-12: Release Preparation
- Pre-train 3 model sizes: Small (20M), Medium (150M), Large (500M)
- Generate comprehensive model cards
- Write academic paper draft
- Open-source release (Apache 2.0 license)

---

## Conclusion

This **Ultimate Development Plan** provides a comprehensive roadmap to transform NeuroFMX from its current 85-90% complete state into a **world-class foundation model** for neural data.

**Key Advantages:**
1. ✅ **Parallel Development:** 5 independent workstreams minimize bottlenecks
2. ✅ **Incremental Testing:** Weekly integration points catch issues early
3. ✅ **Risk Mitigation:** Fallback plans for every major risk
4. ✅ **Cost Efficiency:** Cloud training optimized for <$2000 total
5. ✅ **Comprehensive Evaluation:** Rigorous benchmarking ensures quality

**Expected Outcomes (Week 6):**
- 🎯 Foundation model trained on 200+ sessions across 7+ modalities
- 🎯 Full interpretability suite with 40+ analysis methods
- 🎯 Comprehensive evaluation matrix (cross-species, cross-task)
- 🎯 Production-ready deployment (Docker + Kubernetes)
- 🎯 Academic-quality benchmarks vs CEBRA, LFADS, NDT

**Timeline:** 6-8 weeks to foundation model release
**Budget:** $1500-2500 (cloud training + storage)
**Team:** 1-5 people (scales with AI agent usage)

---

**Let's make this foundational model outstanding!** 🚀🧠
