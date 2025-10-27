# NeuroFMX Revolutionary Expansion Plan

## Executive Summary

This document synthesizes two advanced expansion proposals into a unified implementation strategy that will make NeuroFMX the world's most advanced mechanistic interpretability suite for neural foundation models:

1. **Fractal Geometry Integration** - Comprehensive fractal analysis of neural dynamics
2. **Advanced Mechanistic Interpretability** - Latent circuits, biophysical modeling, DUNL disentanglement

**Goal:** Transform NeuroFMX from "world-class" to "truly revolutionary" by integrating cutting-edge neuroscience methods.

**Total New Code:** ~15,000 lines across 20+ new modules
**Development Time:** 5 parallel workstreams
**Innovation Level:** Beyond state-of-the-art

---

## Table of Contents

1. [Conceptual Integration](#conceptual-integration)
2. [Architecture Overview](#architecture-overview)
3. [Parallel Workstreams](#parallel-workstreams)
4. [Module Specifications](#module-specifications)
5. [Integration Points](#integration-points)
6. [Testing Strategy](#testing-strategy)
7. [Documentation Plan](#documentation-plan)
8. [Commit Strategy](#commit-strategy)

---

## Conceptual Integration

### Key Synergies Between Fractal Geometry and Advanced Mech-Int

1. **Scale-Free Dynamics**
   - Fractal metrics reveal power-law scaling in neural activity
   - Latent circuits can be analyzed for fractal dimensionality
   - Multi-scale modeling benefits from fractional dynamics

2. **Biophysical Plausibility**
   - Fractional Ornstein-Uhlenbeck processes model ion channel noise
   - Dale's law + fractal branching = realistic dendrite growth
   - 1/f spectral priors match biological neural spectra

3. **Interpretability Enhancement**
   - DUNL disentanglement + fractal probes = interpretable scale-dependent features
   - Activation patching + fractal ablation = causal scale analysis
   - Latent circuits with fractal regularization = biologically plausible circuits

4. **Multi-Modal Understanding**
   - Fractal metrics unify across modalities (EEG, fMRI, spikes)
   - Biophysical models bridge micro-macro scales
   - Virtual Brain Integration provides whole-brain context

### Unified Framework

```
┌─────────────────────────────────────────────────────────┐
│         NeuroFMX Foundation Model (Existing)             │
│  [Mamba SSM] [Perceiver-IO] [PopT] [Latent Diffusion]  │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
┌────────▼─────────┐  ┌─────────▼──────────┐
│  FRACTAL LAYER   │  │  CIRCUIT LAYER     │
│                  │  │                    │
│ • Metrics        │  │ • Latent RNNs      │
│ • Regularizers   │  │ • DUNL             │
│ • Stimuli        │  │ • Biophysical      │
│ • Simulators     │  │ • Patching         │
└────────┬─────────┘  └─────────┬──────────┘
         │                       │
         └───────────┬───────────┘
                     │
         ┌───────────▼────────────┐
         │   INTEGRATION LAYER     │
         │                         │
         │ • Multi-scale modeling  │
         │ • Fractal circuits      │
         │ • Biophysical priors    │
         │ • Virtual Brain         │
         └─────────────────────────┘
```

---

## Architecture Overview

### New Directory Structure

```
src/neuros_neurofm/
├── interpretability/
│   ├── fractals/                    # NEW: Fractal geometry suite
│   │   ├── __init__.py
│   │   ├── metrics.py               # Higuchi, DFA, Hurst, spectral
│   │   ├── regularizers.py          # 1/f prior, multifractal smoothness
│   │   ├── probes.py                # Latent FD tracker, attention-fractal
│   │   ├── stimuli.py               # fBm, cascades, 1/f noise
│   │   ├── simulators.py            # fOU, dendrite growth
│   │   └── plotting.py              # Fractal visualizations
│   │
│   ├── circuits/                    # NEW: Latent circuit inference
│   │   ├── __init__.py
│   │   ├── latent_rnn.py            # Low-dim RNN models
│   │   ├── dunl.py                  # Deconvolutional unrolling
│   │   ├── feature_viz.py           # Optimal input synthesis
│   │   └── circuit_extraction.py    # E/I circuit diagrams
│   │
│   ├── biophysical/                 # NEW: Biophysical modeling
│   │   ├── __init__.py
│   │   ├── spiking_nets.py          # Differentiable SNNs
│   │   ├── neuron_models.py         # HH, Izhikevich, LIF
│   │   ├── dales_law.py             # E/I separation
│   │   └── synaptic_models.py       # STDP, STP
│   │
│   ├── interventions/               # NEW: Advanced patching/ablation
│   │   ├── __init__.py
│   │   ├── activation_patching.py   # Systematic patching
│   │   ├── ablation_suite.py        # Lesion studies
│   │   ├── scale_ablation.py        # Fractal scale manipulation
│   │   └── virtual_lesions.py       # Reversible interventions
│   │
│   └── multiscale/                  # NEW: Multi-scale modeling
│       ├── __init__.py
│       ├── lfp_generation.py        # LFP from spikes
│       ├── interarea_coupling.py    # Cross-region interactions
│       ├── virtual_brain.py         # VBI integration
│       └── scale_bridge.py          # Micro-macro bridge
│
├── models/
│   └── mamba_fractional.py          # MODIFIED: Fractional kernels
│
└── losses/
    └── fractal_priors.py            # NEW: Fractal regularization
```

### Total New Modules: 25

---

## Parallel Workstreams

### Workstream 1: Fractal Geometry Foundation (Priority: CRITICAL)

**Modules:**
- `fractals/metrics.py` (600 lines)
- `fractals/regularizers.py` (400 lines)
- `fractals/stimuli.py` (500 lines)
- `fractals/simulators.py` (450 lines)
- `fractals/probes.py` (550 lines)
- `fractals/plotting.py` (400 lines)

**Total:** ~2,900 lines

**Key Deliverables:**
- GPU-accelerated fractal dimension estimators
- Multifractal spectrum computation
- 1/f spectral prior for training
- fBm and cascade generators
- Fractional OU simulator

**Dependencies:** None (standalone)

---

### Workstream 2: Latent Circuit Inference (Priority: HIGH)

**Modules:**
- `circuits/latent_rnn.py` (700 lines)
- `circuits/dunl.py` (800 lines)
- `circuits/feature_viz.py` (500 lines)
- `circuits/circuit_extraction.py` (600 lines)

**Total:** ~2,600 lines

**Key Deliverables:**
- Low-dimensional RNN inference from heterogeneous responses
- DUNL-style deconvolutional unrolling
- Optimal stimulus synthesis (feature visualization)
- Automated E/I circuit diagram extraction

**Dependencies:** Existing SAE, dynamics modules

---

### Workstream 3: Biophysical Modeling (Priority: HIGH)

**Modules:**
- `biophysical/spiking_nets.py` (900 lines)
- `biophysical/neuron_models.py` (700 lines)
- `biophysical/dales_law.py` (400 lines)
- `biophysical/synaptic_models.py` (500 lines)

**Total:** ~2,500 lines

**Key Deliverables:**
- Differentiable spiking neural networks (forward-Euler)
- Hodgkin-Huxley, Izhikevich, LIF implementations
- Dale's law enforcement (separate E/I populations)
- STDP and short-term plasticity

**Dependencies:** None (standalone)

---

### Workstream 4: Advanced Interventions (Priority: MEDIUM)

**Modules:**
- `interventions/activation_patching.py` (600 lines)
- `interventions/ablation_suite.py` (550 lines)
- `interventions/scale_ablation.py` (450 lines)
- `interventions/virtual_lesions.py` (500 lines)

**Total:** ~2,100 lines

**Key Deliverables:**
- Systematic activation patching (layer-by-layer)
- Comprehensive ablation studies
- Scale-specific ablation (fractal integration)
- Reversible virtual lesions

**Dependencies:** Workstream 1 (fractal probes), existing counterfactuals

---

### Workstream 5: Multi-Scale Integration (Priority: MEDIUM)

**Modules:**
- `multiscale/lfp_generation.py` (600 lines)
- `multiscale/interarea_coupling.py` (550 lines)
- `multiscale/virtual_brain.py` (700 lines)
- `multiscale/scale_bridge.py` (500 lines)

**Total:** ~2,350 lines

**Key Deliverables:**
- LFP generation from spiking activity
- Inter-area coupling metrics
- Virtual Brain Integration (TVB compatibility)
- Micro-macro scale bridging

**Dependencies:** Workstream 3 (biophysical models)

---

### Workstream 6: Core Model Enhancements (Priority: CRITICAL)

**Modules:**
- `models/mamba_fractional.py` (400 lines, modification)
- `losses/fractal_priors.py` (500 lines)
- `tokenizers/scale_encodings.py` (450 lines)

**Total:** ~1,350 lines

**Key Deliverables:**
- Fractional kernels in Mamba SSM
- Fractal regularization losses
- Wavelet scattering encodings

**Dependencies:** Workstream 1 (fractal metrics)

---

## Module Specifications

### 1. Fractal Metrics (`fractals/metrics.py`)

**Purpose:** GPU-accelerated fractal dimension and scaling estimators

**Key Classes:**
```python
class HiguchiFractalDimension:
    """Higuchi fractal dimension (HFD) for time series"""
    def compute(self, X: Tensor, k_max: int = 10) -> Tensor:
        # Returns: fractal dimension [batch_size]

class DetrendedFluctuationAnalysis:
    """DFA scaling exponent α"""
    def compute(self, X: Tensor, min_window: int = 10,
                max_window: int = 100) -> Tuple[Tensor, Tensor]:
        # Returns: (alpha, fluctuation_curve)

class HurstExponent:
    """Hurst exponent H (0.5=Brownian, >0.5=persistent, <0.5=anti-persistent)"""
    def compute(self, X: Tensor, method: str = 'rs') -> Tensor:
        # Methods: 'rs' (R/S), 'dfa', 'wavelets'

class SpectralSlope:
    """Power spectral density slope β in 1/f^β"""
    def compute(self, X: Tensor, freq_range: Tuple[float, float]) -> Tensor:
        # Returns: beta (spectral slope)

class GraphFractalDimension:
    """Graph fractal dimension via box-covering"""
    def compute(self, adj_matrix: Tensor, min_box: int = 2,
                max_box: int = 10) -> Tensor:
        # Returns: fractal dimension

class MultifractalSpectrum:
    """Multifractal spectrum τ(q) and f(α)"""
    def compute(self, X: Tensor, q_range: Tensor) -> Dict[str, Tensor]:
        # Returns: {'tau': τ(q), 'alpha': α, 'f_alpha': f(α)}
```

**Features:**
- All methods GPU-compatible (PyTorch)
- Batched computation (process multiple sequences simultaneously)
- Automatic differentiation support (for gradient-based optimization)
- Configurable parameters (window sizes, k_max, etc.)

**Validation:**
- Test on synthetic fractional Brownian motion (known H)
- Test on 1/f^β signals (known β)
- Compare to reference implementations (NOLDS, antropy)

---

### 2. Fractal Regularizers (`fractals/regularizers.py`)

**Purpose:** Training losses that enforce fractal properties

**Key Classes:**
```python
class SpectralPrior(nn.Module):
    """1/f spectral prior for latent dynamics"""
    def __init__(self, target_beta: float = 1.0, weight: float = 0.01):
        self.target_beta = target_beta

    def forward(self, latents: Tensor) -> Tensor:
        # Compute PSD, fit slope β, penalize deviation from target_beta
        # Loss = weight * |β - target_beta|^2

class MultifractalSmoothness(nn.Module):
    """Penalize non-smooth multifractal spectra"""
    def forward(self, latents: Tensor) -> Tensor:
        # Compute τ(q), penalize high curvature
        # Encourages smooth multifractal spectra

class GraphFractalityPrior(nn.Module):
    """Encourage fractal graph structure"""
    def __init__(self, target_dim: float = 2.0):
        self.target_dim = target_dim

    def forward(self, attention_weights: Tensor) -> Tensor:
        # Treat attention as graph adjacency
        # Penalize deviation from target fractal dimension
```

**Integration:**
```python
# In training loop
loss = (
    reconstruction_loss
    + spectral_prior(latents)
    + multifractal_smoothness(latents)
    + graph_fractality_prior(attention_weights)
)
```

---

### 3. Fractal Stimuli (`fractals/stimuli.py`)

**Purpose:** Generate fractal test signals for probing models

**Key Classes:**
```python
class FractionalBrownianMotion:
    """fBm generator with specified Hurst exponent"""
    def generate(self, n_samples: int, H: float, n_dims: int = 1) -> Tensor:
        # Davies-Harte method or spectral synthesis

class ColoredNoise:
    """1/f^β noise generator"""
    def generate(self, n_samples: int, beta: float) -> Tensor:
        # Spectral synthesis: S(f) ∝ 1/f^β

class MultiplicativeCascade:
    """Multifractal cascade model"""
    def generate(self, n_levels: int, branching: int = 2,
                 weights: Optional[Tensor] = None) -> Tensor:
        # Iterative cascade construction

class FractalPatterns:
    """2D fractal patterns (for visual stimuli)"""
    def generate_mandelbrot(self, size: int, zoom: float) -> Tensor:
    def generate_julia(self, size: int, c: complex) -> Tensor:
    def generate_ifs(self, size: int, transforms: List) -> Tensor:
```

**Use Cases:**
- Test model responses to scale-invariant stimuli
- Validate fractal metrics on known signals
- Data augmentation with fractal properties

---

### 4. Fractal Simulators (`fractals/simulators.py`)

**Purpose:** Biophysically-inspired fractal dynamical systems

**Key Classes:**
```python
class FractionalOU:
    """Fractional Ornstein-Uhlenbeck process"""
    def __init__(self, alpha: float = 0.8, theta: float = 1.0,
                 sigma: float = 0.1):
        # dX_t = θ(μ - X_t)dt + σ dB^H_t
        # where B^H is fractional Brownian motion

    def simulate(self, n_steps: int, dt: float = 0.001) -> Tensor:
        # Euler-Maruyama with fBm increments

class DendriteGrowthSimulator:
    """Fractal dendrite growth with target dimension"""
    def __init__(self, target_fd: float = 1.7, branching_prob: float = 0.3):
        self.target_fd = target_fd

    def grow(self, n_iterations: int) -> Tuple[Tensor, Tensor]:
        # Returns: (positions, connectivity_matrix)
        # Uses stochastic L-systems or DLA

class FractalNetworkModel:
    """Network with power-law degree distribution"""
    def __init__(self, n_nodes: int, gamma: float = 2.5):
        self.gamma = gamma  # P(k) ∝ k^(-γ)

    def generate(self) -> Tensor:
        # Returns: adjacency matrix
        # Configuration model or preferential attachment
```

**Applications:**
- Biophysical priors for network structure
- Realistic synthetic data generation
- Testing circuit inference algorithms

---

### 5. Latent Circuit Inference (`circuits/latent_rnn.py`)

**Purpose:** Infer low-dimensional RNN circuits from heterogeneous neural responses

**Key Classes:**
```python
class LatentCircuitModel(nn.Module):
    """Low-dimensional RNN explaining high-D responses (Langdon & Engel 2025)"""
    def __init__(self, n_latent: int = 10, n_observed: int = 100,
                 enforce_dales: bool = True):
        self.latent_rnn = nn.RNN(n_latent, n_latent, nonlinearity='tanh')
        self.readout = nn.Linear(n_latent, n_observed)

        if enforce_dales:
            self.ei_mask = self._create_ei_mask()

    def forward(self, inputs: Tensor, hidden: Optional[Tensor] = None):
        # z_t = W_rec @ tanh(z_{t-1}) + W_in @ u_t
        # x_t = W_out @ z_t  (observed responses)

    def extract_circuit(self) -> Dict:
        """Extract E/I connectivity"""
        # Returns: {
        #   'W_rec': recurrent weights,
        #   'E_indices': excitatory neurons,
        #   'I_indices': inhibitory neurons,
        #   'coupling_strength': connection strength matrix
        # }

class CircuitFitter:
    """Fit latent circuit to observed data"""
    def fit(self, neural_responses: Tensor, stimuli: Tensor,
            n_latent: int = 10, n_epochs: int = 1000) -> LatentCircuitModel:
        # Objective: min ||X - f(Z; W)||^2 + λ||W||_1
        # Returns fitted LatentCircuitModel
```

**Innovation:** This allows extracting interpretable low-dimensional circuits from the foundation model's high-dimensional representations.

---

### 6. DUNL Disentanglement (`circuits/dunl.py`)

**Purpose:** Deconvolutional unrolling for mixed selectivity decomposition

**Key Classes:**
```python
class DUNLModel(nn.Module):
    """Deconvolutional Unrolled Neural Learning"""
    def __init__(self, n_neurons: int, n_factors: int = 20,
                 n_iterations: int = 10, sparsity: float = 0.1):
        # Unrolled iterative sparse coding
        # x ≈ D @ s  where D is dictionary, s is sparse code

        self.dictionaries = nn.ModuleList([
            nn.Linear(n_factors, n_neurons) for _ in range(n_iterations)
        ])
        self.thresholds = nn.Parameter(torch.ones(n_iterations) * sparsity)

    def forward(self, neural_activity: Tensor) -> Tuple[Tensor, Tensor]:
        # Iterative soft-thresholding
        # Returns: (sparse_codes, reconstructed_activity)

    def extract_factors(self) -> Tensor:
        """Extract interpretable factors"""
        # Each column of D is a factor
        # Factors represent mixed selectivity components

class MixedSelectivityAnalyzer:
    """Decompose mixed selectivity into interpretable components"""
    def decompose(self, responses: Tensor, conditions: Tensor) -> Dict:
        # Returns: {
        #   'task_component': task-related variance,
        #   'motor_component': movement-related variance,
        #   'cognitive_component': decision-related variance,
        #   'interaction_terms': nonlinear interactions
        # }
```

**Application:** Separate "what" vs "how" in neural representations (e.g., stimulus identity vs task context).

---

### 7. Biophysical Spiking Networks (`biophysical/spiking_nets.py`)

**Purpose:** Differentiable spiking neural networks for biophysical constraints

**Key Classes:**
```python
class LeakyIntegrateFireNeuron(nn.Module):
    """Differentiable LIF neuron"""
    def __init__(self, tau_mem: float = 10.0, v_threshold: float = 1.0,
                 v_reset: float = 0.0):
        self.tau_mem = tau_mem
        self.v_threshold = v_threshold

    def forward(self, input_current: Tensor, dt: float = 1.0) -> Tuple[Tensor, Tensor]:
        # dV/dt = (-V + I) / τ
        # Spike if V > θ, then reset to v_reset
        # Returns: (spikes, membrane_voltage)
        # Uses surrogate gradient for backprop

class IzhikevichNeuron(nn.Module):
    """Izhikevich's model (rich spiking dynamics)"""
    def __init__(self, a: float = 0.02, b: float = 0.2,
                 c: float = -65.0, d: float = 8.0):
        # dv/dt = 0.04v^2 + 5v + 140 - u + I
        # du/dt = a(bv - u)

class HodgkinHuxleyNeuron(nn.Module):
    """Full HH conductance-based model"""
    def __init__(self, g_Na: float = 120.0, g_K: float = 36.0,
                 g_L: float = 0.3):
        # I_ion = g_Na * m^3 * h * (V - E_Na) + ...

class SpikingNeuralNetwork(nn.Module):
    """Full SNN with configurable neuron types"""
    def __init__(self, n_neurons: int, neuron_type: str = 'lif',
                 connectivity: Optional[Tensor] = None,
                 enforce_dales: bool = True):
        # Build network with specified connectivity
        # Enforce Dale's law if requested
```

**Surrogate Gradients:**
```python
class SurrogateGradient(torch.autograd.Function):
    """Smooth approximation for spike gradient"""
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return (input > 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        # Sigmoid derivative as surrogate
        grad = grad_output * torch.sigmoid(input) * (1 - torch.sigmoid(input))
        return grad
```

---

### 8. Dale's Law Enforcement (`biophysical/dales_law.py`)

**Purpose:** Constrain networks to biologically plausible E/I separation

**Key Classes:**
```python
class DalesLawConstraint:
    """Enforce excitatory/inhibitory separation"""
    def __init__(self, n_neurons: int, ei_ratio: float = 0.8):
        # ei_ratio: fraction of excitatory neurons (typically 80%)
        self.n_exc = int(n_neurons * ei_ratio)
        self.n_inh = n_neurons - self.n_exc

    def apply(self, weight_matrix: nn.Parameter):
        """Constrain weights to be non-negative (E) or non-positive (I)"""
        with torch.no_grad():
            # Excitatory neurons: all outgoing weights ≥ 0
            weight_matrix[:self.n_exc, :] = weight_matrix[:self.n_exc, :].clamp(min=0)
            # Inhibitory neurons: all outgoing weights ≤ 0
            weight_matrix[self.n_exc:, :] = weight_matrix[self.n_exc:, :].clamp(max=0)

class DalesLinear(nn.Module):
    """Linear layer with Dale's law constraint"""
    def __init__(self, in_features: int, out_features: int, ei_ratio: float = 0.8):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        self.constraint = DalesLawConstraint(out_features, ei_ratio)

    def forward(self, x: Tensor) -> Tensor:
        self.constraint.apply(self.weight)
        return F.linear(x, self.weight)
```

**Application:** Make learned circuits biologically plausible by enforcing sign constraints.

---

### 9. Activation Patching (`interventions/activation_patching.py`)

**Purpose:** Systematic causal intervention testing

**Key Classes:**
```python
class ActivationPatcher:
    """Patch activations from source to target computation"""
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks = {}

    def patch_layer(self, layer_name: str, source_input: Tensor,
                    target_input: Tensor) -> Tensor:
        """
        1. Run source_input through model, save activation at layer_name
        2. Run target_input through model, but replace layer_name activation
           with saved activation from source
        3. Return final output
        """
        # Measures causal effect of layer_name on output

    def systematic_patching(self, layers: List[str],
                            source_inputs: Tensor,
                            target_inputs: Tensor) -> pd.DataFrame:
        """Patch all layers systematically, return effect size matrix"""
        # Returns DataFrame: rows=layers, cols=output_dimensions, values=effect_size
```

**Use Case:** Identify which layers are causally responsible for specific behaviors.

---

### 10. Multi-Scale LFP Generation (`multiscale/lfp_generation.py`)

**Purpose:** Generate local field potentials from spiking activity

**Key Classes:**
```python
class LFPGenerator:
    """Generate LFP from spiking neural activity"""
    def __init__(self, kernel_type: str = 'exponential', tau: float = 10.0):
        # LFP(t) = ∫ K(t-s) * Spikes(s) ds
        # Kernel options: 'exponential', 'alpha', 'realistic'

    def generate(self, spike_trains: Tensor, sampling_rate: float = 1000.0) -> Tensor:
        # Convolve spikes with kernel
        # Returns: continuous LFP signal

class CurrentSourceDensity:
    """CSD analysis for laminar probe data"""
    def compute_csd(self, lfp: Tensor, electrode_spacing: float) -> Tensor:
        # CSD = -σ * d²LFP/dz²  (second spatial derivative)

class MultiScaleBridge:
    """Bridge micro (spikes) and macro (LFP/EEG) scales"""
    def spike_to_lfp(self, spikes: Tensor) -> Tensor:
        # Local integration

    def lfp_to_eeg(self, lfp: Tensor, forward_model: Tensor) -> Tensor:
        # Volume conduction (forward model)
```

**Application:** Connect foundation model representations to different recording modalities.

---

### 11. Virtual Brain Integration (`multiscale/virtual_brain.py`)

**Purpose:** Integration with The Virtual Brain (TVB) for whole-brain modeling

**Key Classes:**
```python
class VirtualBrainInterface:
    """Interface to TVB for large-scale network modeling"""
    def __init__(self, connectome: str = 'desikan'):
        # Load structural connectivity (e.g., DTI-based)
        self.connectivity = self._load_connectome(connectome)

    def create_neural_mass_model(self, model_type: str = 'jansen-rit') -> Dict:
        # Options: 'jansen-rit', 'wilson-cowan', 'kuramoto'
        # Returns TVB-compatible model specification

    def embed_foundation_model(self, neurofmx_model: nn.Module,
                               region_mapping: Dict[str, int]):
        """
        Embed NeuroFMX as local dynamics in TVB nodes
        - region_mapping: {'V1': 0, 'V2': 1, ...}
        """
        # Replace TVB's neural mass with NeuroFMX representations

    def simulate(self, duration: float, dt: float = 0.001) -> Dict:
        """
        Run whole-brain simulation
        Returns: {
            'time': time vector,
            'bold': simulated BOLD signals,
            'lfp': simulated LFPs,
            'connectivity_dynamics': time-varying functional connectivity
        }
        """
```

**Innovation:** First foundation model with full-brain context via TVB integration.

---

### 12. Fractional Mamba SSM (`models/mamba_fractional.py`)

**Purpose:** Modify Mamba SSM to use fractional (power-law) kernels

**Modifications:**
```python
class FractionalMambaBlock(nn.Module):
    """Mamba with fractional impulse response"""
    def __init__(self, d_model: int, d_state: int = 16,
                 alpha: float = 0.8, learnable_alpha: bool = True):
        super().__init__()

        # Original Mamba: h(t) = exp(-λt)
        # Fractional Mamba: h(t) = t^(-α) / Γ(1-α)  (power-law)

        if learnable_alpha:
            self.alpha = nn.Parameter(torch.tensor(alpha))
        else:
            self.alpha = alpha

    def compute_kernel(self, seq_len: int) -> Tensor:
        """Compute fractional kernel"""
        t = torch.arange(1, seq_len + 1, dtype=torch.float32)
        # h(t) = t^(-α) / Γ(1-α)
        kernel = t.pow(-self.alpha) / torch.exp(torch.lgamma(1 - self.alpha))
        return kernel

    def forward(self, x: Tensor) -> Tensor:
        # Convolve input with fractional kernel (via FFT)
        kernel = self.compute_kernel(x.size(1))
        # Efficient convolution in frequency domain
        x_fft = torch.fft.rfft(x, dim=1)
        kernel_fft = torch.fft.rfft(kernel, n=x.size(1))
        output = torch.fft.irfft(x_fft * kernel_fft, n=x.size(1))
        return output
```

**Benefit:** Power-law kernels match biological neural dynamics (1/f noise, long-range temporal dependencies).

---

## Integration Points

### 1. Training Pipeline Integration

**Fractal Priors in Loss Function:**
```python
from neuros_neurofm.losses.fractal_priors import SpectralPrior, MultifractalSmoothness
from neuros_neurofm.losses import CombinedLoss

loss_fn = CombinedLoss(
    objectives=[
        MaskedModelingLoss(),
        ForecastingLoss(),
        SpectralPrior(target_beta=1.0, weight=0.01),  # NEW
        MultifractalSmoothness(weight=0.005),          # NEW
    ]
)
```

**Dale's Law in Model:**
```python
from neuros_neurofm.interpretability.biophysical import DalesLinear

# Replace standard linear layers in readout heads
class TaskReadout(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        self.classifier = DalesLinear(d_model, n_classes, ei_ratio=0.8)
```

---

### 2. Interpretability Pipeline Integration

**Fractal Analysis in Mech-Int Callback:**
```python
from neuros_neurofm.interpretability import MechIntCallback
from neuros_neurofm.interpretability.fractals import HiguchiFractalDimension, MultifractalSpectrum

callback = MechIntCallback(
    analyses=[
        'sae',           # Existing
        'alignment',     # Existing
        'dynamics',      # Existing
        'fractal_dims',  # NEW: Fractal analysis
        'latent_circuits',  # NEW: Circuit inference
    ],
    analysis_configs={
        'fractal_dims': {
            'metrics': ['higuchi', 'dfa', 'multifractal'],
            'layers': ['mamba_0', 'mamba_6', 'mamba_11']
        },
        'latent_circuits': {
            'n_latent': 10,
            'enforce_dales': True
        }
    }
)

trainer = pl.Trainer(callbacks=[callback])
```

---

### 3. Evaluation Integration

**Biophysical Constraints in Evaluation:**
```python
from neuros_neurofm.evaluation import BenchmarkSuite
from neuros_neurofm.interpretability.biophysical import SpikingNeuralNetwork

# Evaluate on spiking neural decoding
benchmark = BenchmarkSuite(
    tasks=[
        'motor_decoding',
        'visual_encoding',
        'spiking_prediction',  # NEW: Predict spiking activity
    ],
    decoders={
        'spiking_prediction': SpikingNeuralNetwork(n_neurons=100, neuron_type='lif')
    }
)
```

---

### 4. Real-Time Inference Integration

**Fractal Monitoring in Deployment:**
```python
from neuros_neurofm.interpretability.fractals import HiguchiFractalDimension

class RealtimeMonitor:
    def __init__(self):
        self.fd_estimator = HiguchiFractalDimension()

    def monitor_stream(self, latent_stream: Tensor):
        # Compute rolling fractal dimension
        fd = self.fd_estimator.compute(latent_stream[-1000:])  # Last 1000 timesteps

        if fd < 1.2:  # Anomaly detection
            logger.warning(f"Low complexity detected: FD={fd:.2f}")
```

---

## Testing Strategy

### Unit Tests (per module)

**Fractal Metrics (`tests/test_fractal_metrics.py`):**
```python
def test_higuchi_fd_on_fbm():
    """Test Higuchi FD on fractional Brownian motion with known H"""
    H = 0.7
    fbm = FractionalBrownianMotion().generate(n_samples=1000, H=H)
    fd = HiguchiFractalDimension().compute(fbm)
    expected_fd = 2 - H  # Theoretical FD = 2 - H
    assert abs(fd - expected_fd) < 0.1

def test_dfa_on_pink_noise():
    """Test DFA on 1/f noise"""
    noise = ColoredNoise().generate(n_samples=2000, beta=1.0)
    alpha, _ = DetrendedFluctuationAnalysis().compute(noise)
    assert 0.9 < alpha < 1.1  # α ≈ 1 for 1/f noise
```

**Latent Circuits (`tests/test_latent_circuits.py`):**
```python
def test_circuit_inference_on_synthetic():
    """Test circuit inference on synthetic low-D RNN data"""
    # Generate synthetic data from known RNN
    true_rnn = LatentCircuitModel(n_latent=5, n_observed=50)
    inputs = torch.randn(100, 20, 5)  # [batch, time, input_dim]
    true_responses, _ = true_rnn(inputs)

    # Fit model
    fitter = CircuitFitter()
    inferred_rnn = fitter.fit(true_responses, inputs, n_latent=5)

    # Check recovery
    true_W = true_rnn.latent_rnn.weight_hh_l0.data
    inferred_W = inferred_rnn.latent_rnn.weight_hh_l0.data
    correlation = torch.corrcoef(torch.stack([true_W.flatten(), inferred_W.flatten()]))[0, 1]
    assert correlation > 0.8
```

**Biophysical Models (`tests/test_biophysical.py`):**
```python
def test_lif_spiking():
    """Test LIF neuron produces spikes"""
    neuron = LeakyIntegrateFireNeuron(tau_mem=10.0, v_threshold=1.0)
    current = torch.ones(100) * 1.5  # Suprathreshold current
    spikes, voltage = neuron(current, dt=1.0)
    assert spikes.sum() > 0  # Should spike

def test_dales_law_enforcement():
    """Test Dale's law constraint"""
    layer = DalesLinear(in_features=10, out_features=20, ei_ratio=0.8)
    _ = layer(torch.randn(5, 10))  # Forward pass applies constraint

    # Check excitatory neurons have non-negative weights
    assert (layer.weight[:16, :] >= 0).all()
    # Check inhibitory neurons have non-positive weights
    assert (layer.weight[16:, :] <= 0).all()
```

---

### Integration Tests

**Full Fractal Pipeline (`tests/test_fractal_integration.py`):**
```python
def test_fractal_training_integration():
    """Test training with fractal priors"""
    model = NeuroFMX(d_model=256, n_layers=4)
    loss_fn = CombinedLoss([
        MaskedModelingLoss(),
        SpectralPrior(target_beta=1.0, weight=0.01)
    ])

    # Train for a few steps
    optimizer = torch.optim.Adam(model.parameters())
    for batch in dataloader:
        outputs = model(batch)
        loss = loss_fn(outputs, batch)
        loss.backward()
        optimizer.step()

    # Check spectral slope is closer to target
    latents = model.encode(batch)
    slope = SpectralSlope().compute(latents)
    assert abs(slope - 1.0) < 0.5  # Closer to 1/f
```

**Circuit + Fractal Analysis (`tests/test_circuit_fractal.py`):**
```python
def test_fractal_circuit_inference():
    """Test inferring circuits with fractal regularization"""
    responses = load_test_data('multiunit_recordings.pt')

    # Infer circuit
    fitter = CircuitFitter()
    circuit = fitter.fit(responses, stimuli, n_latent=8)

    # Analyze fractal properties of inferred circuit
    W_rec = circuit.latent_rnn.weight_hh_l0.data
    graph_fd = GraphFractalDimension().compute(W_rec.abs())

    assert 1.5 < graph_fd < 2.5  # Reasonable fractal dimension
```

---

### Performance Benchmarks

**Fractal Metrics Speed Test:**
```python
def benchmark_fractal_metrics():
    """Benchmark fractal metrics on GPU"""
    X = torch.randn(128, 1000, device='cuda')  # [batch, time]

    import time
    start = time.time()
    fd = HiguchiFractalDimension().compute(X)
    elapsed = time.time() - start

    print(f"Higuchi FD: {elapsed:.3f}s for 128 sequences")
    assert elapsed < 1.0  # Should be fast on GPU
```

---

## Documentation Plan

### New Documentation Files

1. **`docs/FRACTAL_GUIDE.md`** - Comprehensive fractal geometry guide
   - Introduction to fractal analysis
   - All metrics explained with math
   - Usage examples
   - Interpretation guidelines

2. **`docs/CIRCUIT_INFERENCE_GUIDE.md`** - Latent circuit inference
   - Theory behind latent RNNs
   - DUNL explanation
   - Step-by-step tutorial
   - Case studies

3. **`docs/BIOPHYSICAL_MODELING.md`** - Biophysical constraints
   - Spiking neural networks
   - Neuron models (LIF, Izhikevich, HH)
   - Dale's law
   - Multi-scale integration

4. **`docs/ADVANCED_INTERVENTIONS.md`** - Activation patching & ablation
   - Causal analysis methods
   - Patching protocols
   - Interpreting results
   - Best practices

5. **`docs/MULTISCALE_INTEGRATION.md`** - Multi-scale modeling
   - LFP generation
   - Virtual Brain integration
   - Cross-scale analysis

### Updated Documentation

- **`docs/API_REFERENCE.md`** - Add all new modules
- **`docs/TRAINING_GUIDE.md`** - Add fractal priors section
- **`examples/README.md`** - Add new examples

### New Examples

6. **`examples/06_fractal_analysis.py`** - Complete fractal analysis workflow
7. **`examples/07_circuit_inference.py`** - Latent circuit extraction
8. **`examples/08_biophysical_constraints.py`** - Training with bio constraints
9. **`examples/09_multiscale_modeling.py`** - Micro-macro integration

---

## Commit Strategy

### Phase 1: Fractal Foundation (Commit 1)
```bash
git add src/neuros_neurofm/interpretability/fractals/
git add tests/test_fractal_*.py
git add docs/FRACTAL_GUIDE.md
git commit -m "feat(fractals): add comprehensive fractal geometry suite

- Implement Higuchi FD, DFA, Hurst, spectral slope, multifractal spectrum
- Add graph fractal dimension
- GPU-accelerated batched computation
- Fractal regularizers: 1/f prior, multifractal smoothness
- Fractal stimuli: fBm, colored noise, cascades
- Fractional OU simulator, dendrite growth
- 30+ tests with 95%+ coverage
- Complete documentation with examples

This establishes world's most comprehensive fractal analysis suite for neural foundation models."
```

### Phase 2: Circuit Inference (Commit 2)
```bash
git add src/neuros_neurofm/interpretability/circuits/
git add tests/test_circuit_*.py
git add docs/CIRCUIT_INFERENCE_GUIDE.md
git commit -m "feat(circuits): add latent circuit inference and DUNL disentanglement

- Latent RNN inference from heterogeneous responses (Langdon & Engel 2025)
- DUNL deconvolutional unrolling for mixed selectivity
- Feature visualization (optimal input synthesis)
- Automated E/I circuit extraction
- Mixed selectivity decomposition
- 25+ tests validating circuit recovery
- Complete guide with case studies

First foundation model with automated circuit inference from learned representations."
```

### Phase 3: Biophysical Modeling (Commit 3)
```bash
git add src/neuros_neurofm/interpretability/biophysical/
git add tests/test_biophysical_*.py
git add docs/BIOPHYSICAL_MODELING.md
git commit -m "feat(biophysical): add differentiable spiking networks and Dale's law

- Differentiable LIF, Izhikevich, Hodgkin-Huxley neurons
- Surrogate gradients for backprop through spikes
- Dale's law enforcement (E/I separation)
- STDP and short-term plasticity
- Full spiking neural network class
- 20+ tests with biological validation
- Comprehensive biophysical modeling guide

Makes NeuroFMX the first foundation model with full biophysical constraints."
```

### Phase 4: Advanced Interventions (Commit 4)
```bash
git add src/neuros_neurofm/interpretability/interventions/
git add tests/test_interventions_*.py
git add docs/ADVANCED_INTERVENTIONS.md
git commit -m "feat(interventions): add activation patching and scale ablation

- Systematic activation patching (causal analysis)
- Comprehensive ablation suite
- Scale-specific ablation (fractal integration)
- Virtual lesions (reversible interventions)
- Automated effect size computation
- 15+ tests covering all intervention types
- Best practices guide

Enables precise causal analysis of foundation model computations."
```

### Phase 5: Multi-Scale Integration (Commit 5)
```bash
git add src/neuros_neurofm/interpretability/multiscale/
git add src/neuros_neurofm/models/mamba_fractional.py
git add tests/test_multiscale_*.py
git add docs/MULTISCALE_INTEGRATION.md
git commit -m "feat(multiscale): add LFP generation, Virtual Brain integration, fractional Mamba

- LFP generation from spiking activity
- Current source density analysis
- Inter-area coupling metrics
- Virtual Brain (TVB) integration
- Fractional Mamba SSM with power-law kernels
- Scale bridging (micro-macro)
- 18+ tests validating cross-scale consistency
- Complete multi-scale modeling guide

First foundation model with full-brain context via TVB and fractal dynamics."
```

### Phase 6: Examples & Final Integration (Commit 6)
```bash
git add examples/06_fractal_analysis.py
git add examples/07_circuit_inference.py
git add examples/08_biophysical_constraints.py
git add examples/09_multiscale_modeling.py
git add configs/neurofm_fractals.yaml
git add docs/API_REFERENCE.md  # Updated
git add REVOLUTIONARY_EXPANSION_COMPLETE.md
git commit -m "feat(examples): add 4 revolutionary examples + final integration

- Example 06: Complete fractal analysis workflow
- Example 07: Latent circuit extraction end-to-end
- Example 08: Training with biophysical constraints
- Example 09: Multi-scale modeling (spikes→LFP→EEG)
- Updated API reference with all new modules
- Completion summary document

Total additions:
- 25 new modules (~12,500 lines)
- 120+ new test cases
- 5 comprehensive guides
- 4 production-ready examples

NeuroFMX is now TRULY REVOLUTIONARY with the world's most advanced mechanistic interpretability suite!"
```

---

## Success Metrics

### Quantitative Metrics

1. **Code Coverage:** >90% for all new modules
2. **Test Pass Rate:** 100% (all tests passing)
3. **Documentation Coverage:** 100% (all classes/functions documented)
4. **Performance:** Fractal metrics <1s for 128 sequences on GPU
5. **Accuracy:** Circuit inference R² >0.8 on synthetic data

### Qualitative Metrics

1. **Novelty:** First foundation model with:
   - Comprehensive fractal analysis
   - Latent circuit inference
   - Full biophysical constraints
   - Virtual Brain integration

2. **Usability:** Production-ready examples for all features

3. **Scientific Impact:** Methods publishable in top-tier venues (Nature Neuroscience, NeurIPS)

---

## Timeline Estimate

**Workstream 1 (Fractals):** 3-4 hours (parallel)
**Workstream 2 (Circuits):** 3-4 hours (parallel)
**Workstream 3 (Biophysical):** 3-4 hours (parallel)
**Workstream 4 (Interventions):** 2-3 hours (parallel)
**Workstream 5 (Multi-scale):** 3-4 hours (parallel)
**Workstream 6 (Integration):** 2-3 hours (depends on 1-5)

**Total Parallel Time:** ~4-5 hours (with efficient parallel execution)
**Total Sequential Time:** ~16-20 hours (if done sequentially)

**Speedup from Parallelization:** ~4x

---

## Risk Mitigation

### Technical Risks

1. **Fractal Metrics Numerical Stability**
   - Risk: Division by zero, NaN in power-law fits
   - Mitigation: Careful regularization, edge case handling

2. **Surrogate Gradient Bias**
   - Risk: Gradient mismatch in spiking networks
   - Mitigation: Validate on known benchmarks, multiple surrogate options

3. **Circuit Inference Identifiability**
   - Risk: Non-unique solutions
   - Mitigation: Regularization (L1, Dale's law), multiple initializations

### Integration Risks

1. **Performance Overhead**
   - Risk: Fractal priors slow down training
   - Mitigation: Make all losses optional, GPU optimization

2. **API Consistency**
   - Risk: New modules don't match existing patterns
   - Mitigation: Follow established conventions, comprehensive testing

---

## Conclusion

This revolutionary expansion will make NeuroFMX:

1. **The world's most comprehensive mechanistic interpretability suite** (15+ analysis families)
2. **The first foundation model with fractal geometry integration**
3. **The first foundation model with latent circuit inference**
4. **The first foundation model with full biophysical constraints**
5. **The first foundation model with Virtual Brain integration**

**Total Impact:**
- 25 new modules
- ~12,500 lines of new code
- 120+ new tests
- 5 comprehensive guides
- 4 production-ready examples

This goes **far beyond** existing foundation models (SAE, GPT-4, Gemini) by providing:
- **Deeper understanding** (circuits, fractals, biophysics)
- **Stronger constraints** (Dale's law, 1/f priors)
- **Multi-scale integration** (spikes → LFP → EEG → fMRI)
- **Causal analysis** (patching, ablation, interventions)

**NeuroFMX will be TRULY REVOLUTIONARY.**

---

**Document Status:** ✅ **COMPLETE**
**Ready for Execution:** YES
**Parallel Workstreams:** 6
**Estimated Completion:** 4-5 hours (parallel execution)

Let's build the future of neural foundation models! 🚀🧠
