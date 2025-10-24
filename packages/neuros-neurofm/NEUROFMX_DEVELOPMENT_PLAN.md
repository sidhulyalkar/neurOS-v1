# NeuroFMx Comprehensive Development Plan
## Vision: A World-Changing Multimodal Neural Foundation Model with Mechanistic Interpretability

**Last Updated:** 2025-10-23
**Status:** Active Development - Parallel Implementation Phase

---

## Executive Summary

This plan details the development of NeuroFMx into a comprehensive foundation model that:
- Integrates 8+ neural data modalities (spikes, LFP, calcium, EEG, fMRI, ECoG, EMG, iEEG)
- Learns unified latent representations across species and recording techniques
- Provides mechanistic interpretability through circuit discovery and neuron analysis
- Scales to H100 HGX cloud infrastructure for large-scale training
- Enables few-shot transfer learning and continual adaptation

---

## Phase 1: Data Infrastructure (Parallel Execution)

### 1.1 Multi-Modal Data Acquisition Scripts

**Priority: HIGH** | **Can Execute in Parallel**

#### A. IBL Dataset (Spikes + Behavior)
**File:** `scripts/data_acquisition/download_ibl.py`

```python
# Key Features:
- Use ONE API or DANDI for NWB access
- Extract spike times, unit metadata, wheel movements, choices
- Bin at 10ms resolution
- Save format: .npz with {spikes, behavior, metadata}
- Target: 20-30 sessions across decision tasks
```

#### B. Allen 2-Photon Calcium Imaging
**File:** `scripts/data_acquisition/download_allen_2p.py`

```python
# Key Features:
- AllenSDK Brain Observatory 2P API
- Extract dF/F traces for all ROIs
- Downsample to 10Hz
- Include stimulus information (images, gratings)
- Save: {calcium, behavior, stimulus_ids, timestamps}
- Target: 10-15 sessions across V1, LM, AL
```

#### C. LFP/iEEG from DANDI
**File:** `scripts/data_acquisition/download_lfp_ieeg.py`

```python
# Key Features:
- DANDI archive access (UCLA seizure, NYU iEEG)
- Extract LFP from Allen Neuropixels (already have spikes)
- Filter 1-100Hz, downsample to 250Hz
- Window into 1s segments (250 samples)
- Save: {lfp, channels, timestamps, behavioral_state}
```

#### D. Human EEG (Cognitive Tasks)
**File:** `scripts/data_acquisition/download_eeg.py`

```python
# Key Features:
- PhysioNet EEG Motor Movement/Imagery Dataset
- Temple University EEG Corpus (TUEG)
- MNE-Python for preprocessing
- Standard 10-20 system channels
- Epoch extraction aligned to events/tasks
- Save: {eeg, channels, events, task_labels}
- Target: 50+ subjects, multiple task conditions
```

#### E. ECoG (Intracranial Human Recordings)
**File:** `scripts/data_acquisition/download_ecog.py`

```python
# Key Features:
- OpenNeuro ECoG datasets
- Miller Lab ECoG data (movement/speech)
- High-resolution spatial coverage
- 200-1000Hz sampling
- Save: {ecog, electrode_positions, behavior, speech/movement_labels}
```

#### F. fMRI (Human Connectome Project)
**File:** `scripts/data_acquisition/download_fmri.py`

```python
# Key Features:
- HCP dataset (resting-state + task fMRI)
- Parcellate using Schaefer/Glasser atlas
- Extract ROI timeseries (e.g., 400 regions)
- TR ~0.7s for HCP
- Save: {fmri_timeseries, roi_labels, task_conditions, confounds}
- Target: 10-20 subjects, multiple runs
```

#### G. EMG (Muscle Activity)
**File:** `scripts/data_acquisition/download_emg.py`

```python
# Key Features:
- Ninapro database (hand movements)
- BCI Competition datasets with EMG
- Multiple muscle channels
- Aligned with movement kinematics
- Save: {emg, channels, movements, kinematics}
```

### 1.2 Unified Data Preprocessing Pipeline
**File:** `src/neuros_neurofm/preprocessing/unified_pipeline.py`

```python
class UnifiedPreprocessor:
    """
    Standardizes all modalities to common format:
    - Temporal alignment
    - Normalization (z-score within modality)
    - Sequence chunking
    - Quality control (artifact detection)
    - Metadata preservation
    """

    def preprocess_modality(self, data, modality_type):
        # Modality-specific preprocessing
        # Return standardized format
        pass
```

---

## Phase 2: Model Architecture Enhancement

### 2.1 MultiModalNeuroFMX Implementation
**File:** `src/neuros_neurofm/models/multimodal_neurofmx.py`

```python
class MultiModalNeuroFMX(nn.Module):
    """
    Enhanced architecture with:
    - Modality-specific tokenizers (8+ types)
    - Learned modality embeddings
    - Cross-modal Perceiver fusion
    - Shared Mamba SSM backbone
    - Multi-task heads (decoder, encoder, contrastive, forecast)
    - Domain adversarial discriminator
    """

    def __init__(self, config):
        self.tokenizers = nn.ModuleDict({
            'spike': SpikeTokenizer(...),
            'lfp': LFPTokenizer(...),
            'calcium': CalciumTokenizer(...),
            'eeg': EEGTokenizer(...),
            'ecog': ECoGTokenizer(...),
            'fmri': fMRITokenizer(...),
            'emg': EMGTokenizer(...)
        })

        # Learned modality embeddings
        self.modality_embeddings = nn.ParameterDict({
            mod: nn.Parameter(torch.randn(1, 1, d_model))
            for mod in self.tokenizers.keys()
        })

        # Cross-modal fusion via Perceiver
        self.perceiver_fusion = PerceiverIO(...)

        # Shared temporal backbone
        self.mamba_backbone = MambaSSM(...)

        # Domain adversarial for species alignment
        self.domain_discriminator = DomainDiscriminator(...)

    def forward(self, modality_dict, species_labels=None):
        # Process each modality
        tokens_list = []
        for modality, data in modality_dict.items():
            tokens = self.tokenizers[modality](data)
            tokens = tokens + self.modality_embeddings[modality]
            tokens_list.append(tokens)

        # Cross-modal fusion
        fused = self.perceiver_fusion(tokens_list)

        # Temporal processing
        features = self.mamba_backbone(fused)

        # Multi-task outputs
        outputs = {
            'latents': features,
            'decoder': self.behavior_decoder(features),
            'encoder': self.neural_encoder(features),
            'contrastive': self.contrastive_head(features)
        }

        # Domain adversarial (if training cross-species)
        if species_labels is not None:
            outputs['domain_logits'] = self.domain_discriminator(features)

        return outputs
```

### 2.2 Modality-Specific Tokenizers
**Directory:** `src/neuros_neurofm/models/tokenizers/`

Each tokenizer converts raw data to (batch, seq_len, d_model) embeddings:

- **EEGTokenizer**: 1D conv over channels + temporal encoding
- **fMRITokenizer**: ROI-wise MLP + spatial graph encoding (optional)
- **ECoGTokenizer**: Similar to EEG but higher freq support
- **EMGTokenizer**: Multi-channel temporal conv
- **CalciumTokenizer**: Temporal binning + feedforward
- **LFPTokenizer**: Spectral features (STFT) + conv

### 2.3 Tri-Modal Contrastive Learning
**File:** `src/neuros_neurofm/losses/contrastive_loss.py`

```python
class TriModalContrastiveLoss(nn.Module):
    """
    Aligns neural activity, behavior, and stimulus in shared space.

    Positive pairs:
    - (neural_t, behavior_t) at same timestep
    - (neural_t, stimulus_t) when stimulus present
    - (behavior_t, stimulus_t) causal relationship

    Uses InfoNCE with temperature scaling
    """

    def forward(self, neural_emb, behavior_emb, stimulus_emb):
        # Compute pairwise similarities
        # Sample positives and negatives
        # Return InfoNCE loss
        pass
```

---

## Phase 3: Mechanistic Interpretability Framework

### 3.1 Neuron Activation Analysis
**File:** `src/neuros_neurofm/interpretability/neuron_analysis.py`

```python
class NeuronActivationAnalyzer:
    """
    Analyzes individual neurons in latent space and model layers.

    Methods:
    - identify_selective_neurons(stimulus_type)
    - compute_tuning_curves(neuron_id, variable)
    - find_behavior_predictive_units()
    - analyze_population_geometry()
    """

    def compute_feature_attribution(self, neuron_id, input_data):
        """Use Integrated Gradients to attribute neuron activation"""
        pass

    def find_monosemantic_units(self):
        """Identify neurons that respond to single concepts"""
        pass
```

### 3.2 Circuit Discovery via Causal Intervention
**File:** `src/neuros_neurofm/interpretability/circuit_discovery.py`

```python
class CircuitDiscovery:
    """
    Discovers computational circuits through interventions.

    Techniques:
    - Activation patching
    - Path patching
    - Ablation experiments
    - Causal mediation analysis
    """

    def activation_patching(self, layer, neuron_subset, clean_input, corrupted_input):
        """
        Replace activations from corrupted input with clean input
        to identify causal neurons for specific computations
        """
        pass

    def discover_behavior_circuit(self, behavior_type):
        """
        Find minimal circuit responsible for behavior decoding
        """
        pass
```

### 3.3 Sparse Autoencoder for Latent Decomposition
**File:** `src/neuros_neurofm/interpretability/sparse_autoencoder.py`

```python
class SparseAutoencoder(nn.Module):
    """
    Decomposes polysemantic neurons into monosemantic features.

    Inspired by Anthropic's interpretability work:
    - L1 sparsity penalty
    - Overcomplete basis (e.g., 64 latents -> 512 features)
    - Learns interpretable feature dictionary
    """

    def __init__(self, latent_dim=512, dictionary_size=4096):
        self.encoder = nn.Linear(latent_dim, dictionary_size)
        self.decoder = nn.Linear(dictionary_size, latent_dim)
        self.sparsity_weight = 0.01

    def forward(self, latents):
        features = F.relu(self.encoder(latents))
        reconstruction = self.decoder(features)

        # L1 sparsity
        sparsity_loss = self.sparsity_weight * features.abs().mean()
        recon_loss = F.mse_loss(reconstruction, latents)

        return reconstruction, features, recon_loss + sparsity_loss
```

### 3.4 Gradient-Based Attribution
**File:** `src/neuros_neurofm/interpretability/attribution.py`

```python
class AttributionMethods:
    """
    Gradient-based methods to understand model decisions.
    """

    @staticmethod
    def integrated_gradients(model, input_data, baseline, target_output):
        """Attribute output to input features"""
        pass

    @staticmethod
    def attention_rollout(model, input_data):
        """Aggregate attention across layers (if using attention)"""
        pass

    @staticmethod
    def grad_cam_temporal(model, input_data, target_neuron):
        """Temporal GradCAM for sequence models"""
        pass
```

### 3.5 Latent Space Visualization
**File:** `src/neuros_neurofm/interpretability/latent_viz.py`

```python
class LatentSpaceVisualizer:
    """
    Visualize and analyze learned representations.
    """

    def plot_latent_manifold(self, latents, labels, method='umap'):
        """2D/3D projection with labels"""
        pass

    def compute_geometry_metrics(self, latents, labels):
        """
        - Linear separability
        - Cluster quality (silhouette score)
        - Manifold dimensionality
        - Cross-condition distances
        """
        pass

    def analyze_cross_modal_alignment(self, spike_latents, eeg_latents, fmri_latents):
        """
        Measure how well different modalities align for same cognitive state
        """
        pass
```

### 3.6 Feature Steering
**File:** `src/neuros_neurofm/interpretability/steering.py`

```python
class FeatureSteering:
    """
    Intervene on latent features to control model behavior.
    """

    def steer_toward_behavior(self, latents, target_behavior, strength=1.0):
        """
        Add steering vector to push latents toward desired behavior
        """
        pass

    def amplify_feature(self, latents, feature_id, factor=2.0):
        """
        Scale specific feature to see impact on decoding
        """
        pass
```

---

## Phase 4: Training Infrastructure

### 4.1 Multi-Task Training Loop
**File:** `training/train_multimodal.py`

```python
class MultiModalTrainer:
    """
    Handles complex multi-task, multi-modal training.

    Features:
    - Dynamic loss balancing (uncertainty weighting)
    - Modality sampling strategies
    - Gradient accumulation
    - Mixed precision (AMP)
    - Distributed training (DDP)
    """

    def __init__(self, config):
        self.model = MultiModalNeuroFMX(config)
        self.dataloaders = self._setup_dataloaders()
        self.loss_weights = self._init_loss_weights()

    def train_step(self, batch_dict):
        """
        batch_dict can contain any subset of modalities
        """
        outputs = self.model(batch_dict['inputs'])

        # Compute individual losses
        loss_decoder = self._compute_decoder_loss(outputs, batch_dict)
        loss_encoder = self._compute_encoder_loss(outputs, batch_dict)
        loss_contrastive = self._compute_contrastive_loss(outputs, batch_dict)
        loss_domain = self._compute_domain_loss(outputs, batch_dict)

        # Dynamic weighting (uncertainty-based)
        total_loss = (
            self.loss_weights['decoder'] * loss_decoder +
            self.loss_weights['encoder'] * loss_encoder +
            self.loss_weights['contrastive'] * loss_contrastive +
            self.loss_weights['domain'] * loss_domain
        )

        return total_loss, {
            'decoder': loss_decoder.item(),
            'encoder': loss_encoder.item(),
            'contrastive': loss_contrastive.item(),
            'domain': loss_domain.item()
        }

    def _update_loss_weights(self, loss_history):
        """Adaptive weighting based on loss scales"""
        pass
```

### 4.2 Cloud Training Setup (H100 HGX)
**Directory:** `infra/`

Already provided in cloud_training_instruction.xml:
- Terraform configs for CoreWeave/Crusoe
- KubeRay deployment
- S3 checkpoint storage
- NCCL tuning for NVLink

**Action Items:**
1. Deploy Terraform infrastructure
2. Build Docker image with all dependencies
3. Configure distributed training
4. Set up checkpoint syncing

---

## Phase 5: Evaluation & Benchmarking

### 5.1 Comprehensive Benchmark Suite
**File:** `evaluation/benchmark_suite.py`

```python
class NeuroFMxBenchmark:
    """
    Evaluate model on standard neuroscience tasks.
    """

    def run_all_benchmarks(self, model, test_datasets):
        results = {}

        # Decoding benchmarks
        results['behavior_r2'] = self.benchmark_behavior_decoding(model)
        results['movement_acc'] = self.benchmark_movement_classification(model)

        # Reconstruction
        results['spike_correlation'] = self.benchmark_neural_reconstruction(model)

        # Latent space quality
        results['clustering_score'] = self.benchmark_latent_clustering(model)
        results['cross_modal_alignment'] = self.benchmark_cross_modal(model)

        # Transfer learning
        results['few_shot_transfer'] = self.benchmark_few_shot_learning(model)

        # Interpretability
        results['circuit_discovery'] = self.benchmark_circuit_identification(model)

        return results
```

### 5.2 Cross-Modal Generalization Tests
**File:** `evaluation/cross_modal_eval.py`

```python
def test_cross_modal_transfer():
    """
    Train linear probe on one modality, test on another.

    Examples:
    - Train on spikes, test on EEG (cross-species)
    - Train on calcium, test on fMRI (cross-resolution)
    - Train on motor cortex EMG, test on ECoG (cross-invasiveness)
    """
    pass

def test_zero_shot_alignment():
    """
    Can model align new modality without training?
    """
    pass
```

---

## Phase 6: Advanced Features

### 6.1 LoRA Adapters for Few-Shot Learning
**File:** `src/neuros_neurofm/adapters/lora.py`

```python
class LoRAAdapter(nn.Module):
    """
    Low-Rank Adaptation for efficient fine-tuning.
    Freeze backbone, learn small rank updates.
    """

    def __init__(self, base_layer, rank=8):
        self.lora_A = nn.Parameter(torch.randn(base_layer.in_features, rank))
        self.lora_B = nn.Parameter(torch.randn(rank, base_layer.out_features))
        self.base_layer = base_layer
        self.base_layer.requires_grad_(False)

    def forward(self, x):
        base_out = self.base_layer(x)
        lora_out = x @ self.lora_A @ self.lora_B
        return base_out + lora_out
```

### 6.2 Continual Learning Framework
**File:** `src/neuros_neurofm/continual/ewc.py`

```python
class ElasticWeightConsolidation:
    """
    Prevent catastrophic forgetting via Fisher Information.
    """

    def compute_fisher(self, model, dataset):
        """Compute parameter importance"""
        pass

    def ewc_loss(self, model, old_params, fisher):
        """Penalty for changing important params"""
        pass
```

---

## Implementation Timeline & Parallelization Strategy

### Week 1-2: Data Infrastructure (PARALLEL)
- **Team A**: EEG, fMRI, EMG acquisition scripts
- **Team B**: IBL, 2P calcium, ECoG acquisition scripts
- **Team C**: LFP/iEEG, unified preprocessing pipeline

### Week 3-4: Model Architecture (PARALLEL)
- **Team A**: MultiModalNeuroFMX core + tokenizers
- **Team B**: Loss functions (contrastive, domain adversarial)
- **Team C**: Training loop infrastructure

### Week 5-6: Interpretability (PARALLEL)
- **Team A**: Neuron analysis, circuit discovery
- **Team B**: Sparse autoencoder, attribution methods
- **Team C**: Visualization tools, feature steering

### Week 7-8: Cloud Deployment & Training
- Deploy infrastructure
- Initial training run ($500 pilot)
- Monitor and debug

### Week 9-10: Evaluation & Iteration
- Benchmark suite execution
- Analysis and refinement
- Prepare for large-scale training

### Week 11+: Scale-Up & Continual Learning
- Full multi-modal training (H100 HGX)
- Implement continual learning
- Community release and documentation

---

## Key Success Metrics

1. **Multi-Modal Integration**: Successfully process 8+ modalities simultaneously
2. **Cross-Modal Transfer**: >70% performance retention when transferring between modalities
3. **Interpretability**: Identify interpretable circuits for 5+ behavioral tasks
4. **Few-Shot Learning**: Achieve 80%+ performance with <10% of task-specific data
5. **Latent Alignment**: Demonstrate aligned representations across species/modalities

---

## Novel Contributions

1. **First foundation model** integrating invasive + non-invasive + imaging data
2. **Mechanistic interpretability** built-in from ground up
3. **Tri-modal contrastive learning** (neural + behavior + stimulus)
4. **Cross-species alignment** via domain adversarial training
5. **Continual learning** for real-time adaptation

---

## Next Steps (Immediate Actions)

1. ✅ Review and approve this plan
2. 🚀 Begin parallel data acquisition script development
3. 🏗️ Set up cloud infrastructure (Terraform)
4. 🧠 Implement MultiModalNeuroFMX skeleton
5. 🔬 Create interpretability framework structure

---

**Let's build this world-changing model!** 🧠🚀
