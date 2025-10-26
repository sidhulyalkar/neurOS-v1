# NeuroFMX Mechanistic Interpretability Expansion Plan
## 10-Module Comprehensive Mech-Int Suite

**Version:** 2.0
**Date:** 2025-10-25
**Based on:** mechint_plan.xml (lines 134-352)

---

## Executive Summary

This plan implements a **world-class mechanistic interpretability suite** for NeuroFMX, going beyond standard interpretability to include:
- **Causal graphs** with temporal dynamics
- **Information theory** and energy landscapes
- **Topology** (persistent homology/TDA)
- **Control theory** (Koopman, controllability)
- **Hierarchical SAEs** with concept dictionaries
- **Meta-dynamics** tracking training evolution
- **Counterfactual analysis** with latent surgery
- **Advanced attribution** methods

### What's Already Complete ✅

From previous development:
- ✅ Basic SAE training and visualization (`sae_training.py`, `sae_visualization.py`, `feature_analysis.py`)
- ✅ Brain alignment (CCA/RSA/PLS) with noise ceilings (`alignment/`)
- ✅ Dynamics analysis (Koopman, Lyapunov, manifolds) (`dynamics.py`)
- ✅ Activation patching and circuit discovery (`circuit_discovery.py`)
- ✅ Neuron analysis and population geometry (`neuron_analysis.py`)

### What's New 🆕

**10 new modules** to build a **complete mechanistic interpretability ecosystem**:

1. **graph_builder.py** - Temporal causal graphs
2. **energy_flow.py** - Information/energy landscapes
3. **geometry_topology.py** - Manifold geometry + persistent homology
4. **control_dynamics.py** - Enhanced control theory (extends existing dynamics.py)
5. **concept_sae.py** - Hierarchical SAE with causal probes
6. **meta_dynamics.py** - Training-time representational trajectories
7. **counterfactuals.py** - Latent surgery and do-calculus
8. **attribution.py** - Integrated gradients and generative path attribution
9. **reporting.py** - Unified HTML/markdown report generator
10. **hooks.py** - Training/eval integration hooks

---

## Module Specifications

### Module 1: graph_builder.py

**Purpose:** Temporal causal graph estimation and perturbation tracing

**Key Components:**

```python
class CausalGraphBuilder:
    """Build temporal causal graphs from latent time-series"""

    def __init__(self, granularity='layer', regularization='lasso'):
        """
        Args:
            granularity: 'layer', 'channel', or 'neuron'
            regularization: 'lasso', 'elastic_net', or 'ridge'
        """

    def build_causal_graph(
        self,
        latents: torch.Tensor,  # (B, T, D)
        window_size: int = 256,
        lag: int = 10,
        alpha: float = 0.001
    ) -> CausalGraph:
        """
        Estimate causal graph using Granger causality

        Returns:
            CausalGraph with adjacency matrix, edge weights, confidence
        """

    def time_varying_graph(
        self,
        latents: torch.Tensor,
        window_size: int = 256,
        hop_size: int = 128
    ) -> TimeVaryingGraph:
        """Sliding window causal graph over time"""

    def perturbation_engine(
        self,
        model: nn.Module,
        latents: torch.Tensor,
        nodes: List[int],
        magnitude: float = 0.5,
        duration: int = 20
    ) -> PerturbationEffect:
        """
        Inject perturbations and measure causal effects

        Returns effects on outputs, alignment metrics, downstream tasks
        """

    def align_with_anatomy(
        self,
        graph: CausalGraph,
        anatomical_prior: Dict[str, List[str]]
    ) -> AlignmentScore:
        """Correlate discovered graph with anatomical connectivity"""
```

**Dependencies:**
- `tigramite` - Advanced Granger causality
- `networkx` - Graph algorithms
- `pywhy-graphlib` - Causal inference utilities

**Key Features:**
- Neural Granger causality with block/channel granularity
- Time-varying graphs with sliding windows
- Perturbation-based causal effect measurement
- Anatomical prior alignment
- Graph visualization with causal strength

**Deliverables:**
- Adjacency matrices over time
- Edge weight distributions
- Perturbation effect heatmaps
- Anatomical alignment scores

---

### Module 2: energy_flow.py

**Purpose:** Information/energy landscape and entropy production analysis

**Key Components:**

```python
class InformationFlowAnalyzer:
    """Analyze information flow through NeuroFMX layers"""

    def estimate_mutual_information(
        self,
        X: torch.Tensor,  # Input
        Z_layers: List[torch.Tensor],  # Layer activations
        Y: torch.Tensor,  # Output/target
        method: str = 'mine'  # 'mine', 'knn', 'histogram'
    ) -> Dict[str, float]:
        """
        Compute I(X;Z_l) and I(Z_l;Y) for each layer

        Returns mutual information curves across layers
        """

    def information_plane(
        self,
        activations: Dict[str, torch.Tensor]
    ) -> InformationPlane:
        """
        Tishby's information plane: I(X;T) vs I(T;Y)

        Visualizes compression vs prediction tradeoff
        """

class EnergyLandscape:
    """Estimate energy landscape of latent space"""

    def estimate_landscape(
        self,
        latents: torch.Tensor,
        method: str = 'score'  # 'score', 'quadratic', 'density'
    ) -> EnergyFunction:
        """
        Approximate energy U(z) such that p(z) ∝ exp(-U(z))
        """

    def find_basins(
        self,
        landscape: EnergyFunction,
        num_basins: int = 5
    ) -> List[Basin]:
        """Identify energy basins (stable states)"""

    def compute_barriers(
        self,
        landscape: EnergyFunction,
        basins: List[Basin]
    ) -> np.ndarray:
        """Compute energy barriers between basins"""

class EntropyProduction:
    """Measure entropy production along trajectories"""

    def estimate_entropy_production(
        self,
        trajectories: torch.Tensor,  # (B, T, D)
        dt: float = 0.01
    ) -> torch.Tensor:
        """
        Estimate dS/dt along trajectories

        High entropy production = far from equilibrium
        """
```

**Dependencies:**
- `hyppo` - High-dimensional independence testing
- `statsmodels` - Density estimation
- Custom MINE implementation (Mutual Information Neural Estimation)

**Key Features:**
- I(X;Z), I(Z;Y) estimation using MINE or k-NN
- Information plane visualization (Tishby et al.)
- Energy landscape via score estimation
- Basin detection and barrier heights
- Entropy production proxies

**Deliverables:**
- MI curves per layer (saved as JSON)
- Information plane plots
- Energy landscape heatmaps
- Entropy production time-series

---

### Module 3: geometry_topology.py

**Purpose:** Latent manifold geometry + persistent homology (TDA)

**Key Components:**

```python
class ManifoldGeometry:
    """Geometric analysis of latent manifolds"""

    def estimate_curvature(
        self,
        trajectories: torch.Tensor,
        k_neighbors: int = 10
    ) -> torch.Tensor:
        """
        Estimate Riemannian curvature locally

        Uses local PCA to estimate tangent spaces
        """

    def compute_divergence(
        self,
        trajectories: torch.Tensor
    ) -> torch.Tensor:
        """Compute divergence of latent flow"""

    def identify_slow_manifold(
        self,
        trajectories: torch.Tensor,
        n_components: int = 3
    ) -> SlowManifold:
        """
        Find slow manifold (low-dimensional attractor)

        Returns manifold basis, explained variance, geodesic distances
        """

class TopologicalAnalysis:
    """Persistent homology and topological features"""

    def __init__(self):
        import gudhi
        self.gudhi = gudhi

    def compute_persistence(
        self,
        point_cloud: torch.Tensor,
        max_dimension: int = 2
    ) -> PersistenceDiagram:
        """
        Compute persistent homology (Betti numbers)

        Detects loops, voids, and higher-dimensional holes
        """

    def betti_numbers(
        self,
        persistence: PersistenceDiagram,
        threshold: float = 0.1
    ) -> Dict[int, int]:
        """
        Extract Betti numbers (topological features)

        β0 = connected components
        β1 = loops/cycles
        β2 = voids
        """

    def compare_topologies(
        self,
        persistence1: PersistenceDiagram,
        persistence2: PersistenceDiagram
    ) -> float:
        """
        Wasserstein distance between persistence diagrams

        Compares topological structure across conditions
        """

class ManifoldVisualization:
    """Embedding and visualization"""

    def umap_embedding(
        self,
        latents: torch.Tensor,
        temporal_coloring: bool = True
    ) -> Tuple[np.ndarray, plt.Figure]:
        """UMAP embedding with temporal or task coloring"""

    def isomap_embedding(self, latents: torch.Tensor):
        """Isomap for geodesic distance preservation"""
```

**Dependencies:**
- `gudhi` - Computational topology
- `umap-learn` - UMAP embeddings
- `pyriemann` - Riemannian geometry on manifolds

**Key Features:**
- Manifold curvature estimation
- Persistent homology (Betti numbers)
- Loop/void detection across species/tasks
- UMAP/Isomap with temporal coloring
- Wasserstein distance for topology comparison

**Deliverables:**
- Persistence diagrams
- Betti number time-series
- UMAP plots with temporal coloring
- Curvature heatmaps

---

### Module 4: control_dynamics.py (Enhancement)

**Purpose:** Extended control theory analysis (builds on existing dynamics.py)

**New Components to Add:**

```python
class ExtendedControlAnalyzer(DynamicsAnalyzer):
    """Enhanced control and observability analysis"""

    def compute_controllability_gramian(
        self,
        A: torch.Tensor,
        B: torch.Tensor,
        time_horizon: float = 10.0
    ) -> torch.Tensor:
        """
        Solve Lyapunov equation: AW + WA^T + BB^T = 0

        Returns controllability Gramian W
        """

    def compute_observability_gramian(
        self,
        A: torch.Tensor,
        C: torch.Tensor,
        time_horizon: float = 10.0
    ) -> torch.Tensor:
        """
        Solve observability Lyapunov equation

        Returns observability Gramian
        """

    def controllability_indices(
        self,
        gramian: torch.Tensor
    ) -> torch.Tensor:
        """
        Eigenvalues of Gramian = controllability per mode

        Large eigenvalue = easy to control
        """

    def energy_to_control(
        self,
        initial_state: torch.Tensor,
        target_state: torch.Tensor,
        gramian: torch.Tensor
    ) -> float:
        """
        Minimum energy to steer from initial to target state

        E = (x1 - x0)^T W^-1 (x1 - x0)
        """

    def minimal_control_input(
        self,
        initial_state: torch.Tensor,
        target_state: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        time_horizon: float
    ) -> torch.Tensor:
        """
        Compute optimal control trajectory u(t)
        """
```

**Integration:** Extend existing `dynamics.py` with these methods

---

### Module 5: concept_sae.py

**Purpose:** Hierarchical sparse autoencoders with concept dictionaries

**Key Components:**

```python
class HierarchicalSAE:
    """Multi-layer SAE hierarchy for concept discovery"""

    def __init__(
        self,
        layer_sizes: List[int],  # e.g., [512, 4096, 16384]
        sparsity_coefficients: List[float]
    ):
        """
        Build hierarchical SAE

        Bottom layer: fine-grained features
        Top layer: abstract concepts
        """

    def train_hierarchy(
        self,
        activations: Dict[str, torch.Tensor],
        num_epochs: int = 100
    ):
        """Train all layers bottom-up"""

    def get_concept_tree(self) -> ConceptTree:
        """
        Build hierarchical concept tree

        Links low-level features to high-level concepts
        """

class ConceptDictionary:
    """Feature dictionary with semantic labels"""

    def build_dictionary(
        self,
        sae: HierarchicalSAE,
        probe_labels: Dict[str, torch.Tensor]
    ) -> Dict[int, ConceptLabel]:
        """
        Assign semantic labels to features

        Uses linear probes to map features to:
        - Brain regions
        - Behavioral states
        - Stimulus categories
        - Task epochs
        """

    def link_to_modalities(
        self,
        features: torch.Tensor,
        modality_data: Dict[str, torch.Tensor]
    ) -> Dict[int, List[str]]:
        """
        Which features respond to which modalities?

        Returns: {feature_id: ['eeg', 'video', ...]}
        """

class CausalSAEProbe:
    """Causal interventions using SAE features"""

    def reinsert_feature(
        self,
        model: nn.Module,
        feature_id: int,
        magnitude: float,
        layer: str
    ) -> torch.Tensor:
        """
        Reinsert specific feature at given magnitude

        Measure effect on outputs and brain alignment
        """

    def causal_importance_score(
        self,
        model: nn.Module,
        features: List[int],
        metric: Callable
    ) -> Dict[int, float]:
        """
        Rank features by causal importance

        Ablate each feature, measure impact on metric
        """
```

**Key Features:**
- Multi-layer SAE hierarchy (3-5 layers)
- Concept trees linking features
- Semantic labeling via probes
- Modality attribution per feature
- Causal importance via reinsertion

**Deliverables:**
- Hierarchical feature tree (JSON)
- Concept dictionary with labels
- Causal importance rankings
- Feature→modality attribution maps

---

### Module 6: meta_dynamics.py

**Purpose:** Training-time representational trajectories and gradient attribution

**Key Components:**

```python
class RepresentationalTrajectory:
    """Track how representations evolve during training"""

    def __init__(self, checkpoint_dir: str):
        """Load checkpoints at different training steps"""

    def compute_trajectory(
        self,
        dataset: DataLoader,
        layers: List[str]
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Extract representations at each checkpoint

        Returns: {layer_name: [repr_step0, repr_step1000, ...]}
        """

    def measure_drift(
        self,
        trajectory: List[torch.Tensor],
        metric: str = 'cca'  # 'cca', 'rsa', 'procrustes'
    ) -> np.ndarray:
        """
        Measure representational drift over training

        Returns drift score between consecutive checkpoints
        """

    def detect_emergence(
        self,
        trajectory: List[torch.Tensor],
        threshold: float = 0.1
    ) -> List[Tuple[int, str]]:
        """
        Detect when features "emerge" during training

        Returns: [(step, feature_description), ...]
        """

class GradientAttribution:
    """Attribute training dynamics to specific parameters"""

    def gradient_flow_over_time(
        self,
        checkpoint_dir: str,
        parameter_groups: Dict[str, List[str]]
    ) -> Dict[str, np.ndarray]:
        """
        Track gradient magnitudes over training

        Which parameters/circuits consolidate over time?
        """

    def feature_birth_analysis(
        self,
        saes: List[HierarchicalSAE],
        steps: List[int]
    ) -> Dict[int, int]:
        """
        When do interpretable features first appear?

        Returns: {feature_id: birth_step}
        """

    def plasticity_tracking(
        self,
        checkpoints: List[Dict],
        layer_names: List[str]
    ) -> pd.DataFrame:
        """
        Track per-layer plasticity (weight change rate)

        Columns: step, layer, weight_change, gradient_norm
        """

class TrainingPhaseDetection:
    """Detect phases in training dynamics"""

    def detect_phases(
        self,
        loss_curve: np.ndarray,
        drift_curve: np.ndarray
    ) -> List[TrainingPhase]:
        """
        Identify distinct training phases

        Phases: warmup, fitting, compression, saturation
        """
```

**Key Features:**
- Checkpoint-based representational trajectory
- CCA/RSA drift measurement over training
- Feature emergence detection
- Gradient flow analytics
- Training phase identification

**Deliverables:**
- Drift curves (CCA similarity over time)
- Feature birth timeline
- Gradient flow heatmaps per layer
- Training phase annotations

---

### Module 7: counterfactuals.py

**Purpose:** Counterfactual latent surgery and do-calculus interventions

**Key Components:**

```python
class LatentSurgery:
    """Targeted edits to hidden states"""

    def edit_latent(
        self,
        model: nn.Module,
        input_data: torch.Tensor,
        layer: str,
        edit_fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply edit_fn to latent at specified layer

        Regenerate outputs/decodings with edited latent
        """

    def swap_latent_dimension(
        self,
        source_latent: torch.Tensor,
        target_latent: torch.Tensor,
        dims: List[int]
    ) -> torch.Tensor:
        """
        Swap specific dimensions between two latents

        Useful for disentanglement analysis
        """

    def interpolate_latents(
        self,
        latent1: torch.Tensor,
        latent2: torch.Tensor,
        num_steps: int = 10
    ) -> List[torch.Tensor]:
        """
        Spherical interpolation between latents

        Returns smooth trajectory in latent space
        """

class DoCalculusInterventions:
    """Causal interventions using do-calculus"""

    def estimate_causal_effect(
        self,
        model: nn.Module,
        data: torch.Tensor,
        intervention: Dict[str, torch.Tensor],  # {layer: value}
        outcome_fn: Callable
    ) -> float:
        """
        Estimate P(Y | do(Z_k = z))

        Intervention: set latent Z_k to specific value z
        Outcome: measure effect on Y via outcome_fn
        """

    def causal_response_curve(
        self,
        model: nn.Module,
        data: torch.Tensor,
        layer: str,
        dim: int,
        values: np.ndarray
    ) -> np.ndarray:
        """
        Sweep do(Z_k[dim] = v) for v in values

        Returns response curve Y(v)
        """

class SyntheticLesions:
    """Knock-out heads/blocks/circuits"""

    def lesion_heads(
        self,
        model: nn.Module,
        layer: int,
        heads: List[int]
    ) -> nn.Module:
        """
        Zero-out attention heads or SSM blocks

        Returns modified model
        """

    def measure_compensation(
        self,
        original_model: nn.Module,
        lesioned_model: nn.Module,
        data: torch.Tensor,
        metric: Callable
    ) -> Dict[str, float]:
        """
        Measure how network compensates for lesion

        Returns: {
            'immediate_drop': ...,  # Performance right after lesion
            'recovered_after_finetune': ...,  # After 100 steps
            'compensation_score': ...
        }
        """
```

**Key Features:**
- Targeted latent editing with hooks
- Do-calculus interventions P(Y|do(Z=z))
- Causal response curves
- Synthetic lesions (head/block knockout)
- Compensation measurement after lesions

**Deliverables:**
- Causal effect estimates
- Response curve plots
- Lesion impact reports
- Compensation scores

---

### Module 8: attribution.py

**Purpose:** Input/channel/region attributions and generative path attribution

**Key Components:**

```python
class IntegratedGradients:
    """Integrated Gradients attribution"""

    def attribute(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor,
        baseline: Optional[torch.Tensor] = None,
        num_steps: int = 50
    ) -> torch.Tensor:
        """
        Compute IG attributions for input features

        Returns attribution scores per input channel/time
        """

    def attribute_channels(
        self,
        model: nn.Module,
        neural_data: torch.Tensor,  # (B, T, C)
        channel_names: List[str]
    ) -> Dict[str, float]:
        """
        Attribution per neural channel

        Returns: {'channel_1': score, ...}
        """

    def attribute_brain_regions(
        self,
        model: nn.Module,
        neural_data: torch.Tensor,
        region_map: Dict[int, str]  # {channel_idx: region_name}
    ) -> Dict[str, float]:
        """
        Attribution aggregated by brain region

        Returns: {'V1': score, 'M1': score, ...}
        """

class DeepLIFT:
    """DeepLIFT attribution (faster than IG)"""

    def attribute(
        self,
        model: nn.Module,
        input: torch.Tensor,
        baseline: torch.Tensor
    ) -> torch.Tensor:
        """DeepLIFT attribution scores"""

class GenerativePathAttribution:
    """Decompose reconstructions into contributing subcircuits"""

    def decompose_reconstruction(
        self,
        model: nn.Module,
        input: torch.Tensor,
        target_output: torch.Tensor
    ) -> Dict[str, float]:
        """
        Decompose reconstruction into:
        - Per-layer contribution
        - Per-head contribution (if attention)
        - Per-SAE-feature contribution

        Returns contribution scores summing to 1.0
        """

    def path_importance(
        self,
        model: nn.Module,
        input: torch.Tensor,
        paths: List[List[str]]  # List of layer sequences
    ) -> Dict[str, float]:
        """
        Measure importance of specific computational paths

        E.g., input → layer2 → layer5 → output
        """
```

**Key Features:**
- Integrated Gradients for inputs, channels, regions
- DeepLIFT for faster attribution
- Generative path decomposition
- Circuit contribution scoring
- Multi-modal attribution

**Deliverables:**
- Per-channel attribution scores
- Per-region heatmaps
- Path importance rankings
- Contribution decomposition pie charts

---

### Module 9: reporting.py

**Purpose:** Unified HTML/markdown report generator with plots and tables

**Key Components:**

```python
class MechIntReport:
    """Generate comprehensive mech-int report"""

    def __init__(self, output_dir: str):
        """Initialize report generator"""

    def add_section(self, title: str, content: str):
        """Add markdown section"""

    def add_figure(self, fig: plt.Figure, caption: str):
        """Add matplotlib figure"""

    def add_table(self, df: pd.DataFrame, caption: str):
        """Add pandas table"""

    def add_metric(self, name: str, value: float, unit: str = ""):
        """Add scalar metric"""

    def generate_html(self) -> str:
        """Generate HTML report"""

    def generate_markdown(self) -> str:
        """Generate markdown report"""

    def export_to_mlflow(self, mlflow_client):
        """Log report to MLflow"""

    def export_to_wandb(self, wandb_run):
        """Log report to W&B"""

class UnifiedMechIntReporter:
    """Run all analyses and generate unified report"""

    def __init__(
        self,
        model: nn.Module,
        data: DataLoader,
        config: Dict[str, Any]
    ):
        """Initialize with model, data, config"""

    def run_all_analyses(self) -> MechIntReport:
        """
        Run all mech-int modules:
        1. Causal graphs
        2. Energy flow
        3. Topology
        4. Control dynamics
        5. SAE features
        6. Alignment
        7. Counterfactuals
        8. Attribution

        Generate unified report with all results
        """

    def run_selected_analyses(
        self,
        analyses: List[str]
    ) -> MechIntReport:
        """Run only specified analyses"""
```

**Key Features:**
- Markdown + HTML generation
- Figure/table embedding
- MLflow/W&B integration
- Unified multi-analysis reports
- Templated sections for each module

**Deliverables:**
- HTML report with interactive plots
- Markdown for GitHub/docs
- MLflow artifacts
- W&B dashboard

---

### Module 10: hooks.py (Integration)

**Purpose:** Training/eval integration hooks for automatic mech-int

**Key Components:**

```python
class MechIntHooks:
    """Training loop hooks for mech-int"""

    def __init__(self, config: Dict[str, Any]):
        """
        Config includes:
        - sample_layers: [2, 6, 10]
        - save_hidden_every_n_steps: 200
        - analyses_to_run: ['causal_graph', 'sae', ...]
        """

    def register_hooks(
        self,
        model: nn.Module,
        trainer: pl.Trainer
    ):
        """
        Register forward hooks to sample hidden states

        Write to object storage as shards
        """

    def on_training_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Dict,
        batch: Any,
        batch_idx: int,
        global_step: int
    ):
        """
        Called after each training step

        Sample and save hidden states if global_step % save_every_n == 0
        """

    def on_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule
    ):
        """
        Run mech-int analyses at end of epoch

        Generate checkpoint-based reports
        """

class EvalMechIntRunner:
    """Add mech-int to evaluation pipeline"""

    def run_mechint_eval(
        self,
        model: nn.Module,
        eval_data: DataLoader,
        checkpoint_path: str,
        hidden_shards_path: str,
        config: Dict[str, Any]
    ) -> MechIntReport:
        """
        Run full mech-int suite on evaluation data

        Uses pre-saved hidden states from training
        """
```

**Integration Points:**
- Training loop: register forward hooks
- Save hidden states every N steps
- Run analyses at epoch end
- Evaluation: mech-int runner
- FastAPI: `/interpret` endpoint

---

## Hydra Configuration

**File:** `configs/mechint/default.yaml`

```yaml
mechint:
  # Which layers to sample
  sample_layers: [2, 6, 10, 14]

  # Sampling frequency
  save_hidden_every_n_steps: 200

  # Causal graph settings
  causal_graph:
    enabled: true
    window: 256
    lag: 10
    alpha: 0.001
    regularization: lasso
    granularity: layer

  # SAE settings
  sae:
    enabled: true
    k_features: 2048
    l1: 1e-3
    hierarchical: true
    layer_sizes: [512, 4096, 16384]

  # Alignment settings
  alignment:
    enabled: true
    method: CCA  # CCA, RSA, PLS
    n_components: 10
    bootstrap: 200
    noise_ceiling: true

  # Control dynamics
  control:
    enabled: true
    koopman_window: 128
    compute_gramians: true

  # Topology
  topology:
    enabled: true
    rips_maxdim: 2
    persistence_threshold: 0.1

  # Energy flow
  energy:
    enabled: true
    mi_method: mine  # mine, knn, histogram
    energy_method: score
    num_basins: 5

  # Counterfactuals
  counterfactuals:
    enabled: false  # Expensive, run on-demand
    magnitude: 0.5
    duration: 20

  # Attribution
  attribution:
    enabled: true
    method: integrated_gradients
    num_steps: 50

  # Meta-dynamics
  meta_dynamics:
    enabled: false  # Requires multiple checkpoints
    checkpoint_interval: 5000
    drift_metric: cca

  # Reporting
  reporting:
    format: html  # html, markdown, both
    export_mlflow: true
    export_wandb: true
    output_dir: reports/mechint
```

---

## Dependencies

### Required Packages

```bash
# Core
torch>=2.0
numpy>=1.24
scipy>=1.10

# ML utilities
scikit-learn>=1.3
umap-learn>=0.5

# Causality
networkx>=3.0
pywhy-graphlib>=0.2
tigramite>=5.0

# Statistics & information theory
hyppo>=0.3
statsmodels>=0.14
pingouin>=0.5

# Optimal transport
pot>=0.9

# Geometry
pyriemann>=0.4

# Topology
gudhi>=3.8
kmapper>=2.0

# Experiment tracking
mlflow>=2.8
wandb>=0.15
ray[tune]>=2.8

# Visualization
matplotlib>=3.7
seaborn>=0.12
plotly>=5.14
```

**Installation:**
```bash
pip install neuros-neurofm[mechint]
```

---

## Testing Strategy

### Unit Tests

**Files to create:**
1. `tests/unit/test_mechint_causal_graph.py` - Graph building, perturbations
2. `tests/unit/test_mechint_energy_flow.py` - MI estimation, landscapes
3. `tests/unit/test_mechint_topology.py` - Persistence, Betti numbers
4. `tests/unit/test_mechint_concept_sae.py` - Hierarchical SAE
5. `tests/unit/test_mechint_counterfactuals.py` - Latent surgery, do-calculus
6. `tests/unit/test_mechint_attribution.py` - IG, DeepLIFT
7. `tests/unit/test_mechint_meta_dynamics.py` - Trajectories, drift
8. `tests/unit/test_mechint_reporting.py` - Report generation

### Integration Tests

**File:** `tests/test_mechint_end_to_end.py`

```python
def test_full_mechint_pipeline():
    """Test complete mech-int pipeline"""

    # 1. Train model
    model = train_small_model()

    # 2. Run all analyses
    reporter = UnifiedMechIntReporter(model, data, config)
    report = reporter.run_all_analyses()

    # 3. Validate report
    assert report.has_section('causal_graph')
    assert report.has_section('sae_features')
    assert report.has_section('alignment')

    # 4. Export
    report.generate_html()
    report.export_to_mlflow(mlflow_client)
```

---

## Ray Jobs for Distributed Execution

**Job:** `mechint_eval_job`

```python
# Run via Ray
ray job submit \
    --working-dir . \
    --runtime-env requirements-mechint.txt \
    -- python -m neuros_neurofm.evaluation.run_mechint \
         +mechint=default \
         ckpt=s3://neurofmx/checkpoints/latest.pt \
         datablob=s3://neurofmx/runs/LATEST/hidden_shards/
```

**Resources:** 1 GPU, 8 CPU cores, 64GB RAM

---

## Metrics and Outputs

### Quantitative Metrics

1. **Graph causality strength:** Normalized edge weights, stability over time
2. **Mutual information curves:** I(X;Z_l) per layer
3. **Topological features:** Betti numbers (β0, β1, β2) across tasks/species
4. **Controllability indices:** Distribution of Gramian eigenvalues
5. **SAE feature scores:** Causal importance, cross-modal activation patterns
6. **Alignment scores:** CCA/RSA with noise-ceiling correction
7. **Counterfactual effect sizes:** Impact on decoding/behavior

### Qualitative Outputs

1. Causal graph visualizations (directed graphs with edge weights)
2. Energy landscape heatmaps (2D/3D)
3. Persistence diagrams (topological features)
4. Phase portraits and attractors
5. Feature hierarchies (concept trees)
6. Attribution heatmaps per brain region
7. Training trajectory plots (drift curves)

---

## Security and Ethics

**From XML specification:**
- Respect dataset licenses and privacy
- Gate `/interpret` endpoint for protected datasets
- Strip PHI (Protected Health Information) from reports
- Add data governance metadata to all outputs
- Implement consent checking before mech-int analysis

---

## Usage Examples

### Example 1: Run during training

```bash
python -m neuros_neurofm.training.train \
    +exp=multimodal_foundation \
    +mechint.save_hidden_every_n_steps=100 \
    mechint.sae.enabled=true \
    mechint.alignment.enabled=true
```

### Example 2: Run on existing checkpoint

```python
from neuros_neurofm.interpretability import UnifiedMechIntReporter

# Load model
model = load_checkpoint('path/to/checkpoint.pt')

# Run analyses
reporter = UnifiedMechIntReporter(
    model=model,
    data=eval_dataloader,
    config={'mechint': config}
)

report = reporter.run_all_analyses()
report.generate_html('mechint_report.html')
report.export_to_wandb(wandb.run)
```

### Example 3: API endpoint

```bash
curl -X POST http://localhost:8000/interpret \
     -F "task=causal_graph" \
     -F "hidden=@latent_activations.pt" \
     -F "config=@mechint_config.yaml"
```

---

## Development Timeline

### Week 1: Core Infrastructure
- ✅ Set up module structure
- ✅ Implement base classes
- ✅ Create Hydra configs
- ✅ Set up testing framework

### Week 2: Graph & Energy (Modules 1-2)
- Implement `graph_builder.py`
- Implement `energy_flow.py`
- Unit tests for both
- Integration with existing codebase

### Week 3: Geometry & Control (Modules 3-4)
- Implement `geometry_topology.py`
- Enhance `control_dynamics.py`
- Persistent homology integration
- Controllability Gramians

### Week 4: Advanced SAE & Meta (Modules 5-6)
- Implement `concept_sae.py`
- Implement `meta_dynamics.py`
- Hierarchical SAE training
- Training trajectory analysis

### Week 5: Counterfactuals & Attribution (Modules 7-8)
- Implement `counterfactuals.py`
- Implement `attribution.py`
- Latent surgery framework
- Integrated gradients

### Week 6: Reporting & Integration (Modules 9-10)
- Implement `reporting.py`
- Implement `hooks.py`
- End-to-end integration tests
- Documentation and examples

---

## Success Criteria

**Quantitative:**
- All 10 modules pass unit tests
- End-to-end integration test succeeds
- Report generation completes in <10 minutes for standard model
- MLflow/W&B logging works correctly

**Qualitative:**
- Reports are human-readable and informative
- Visualizations are publication-quality
- Code is well-documented
- Examples run without errors

**Scientific:**
- Causal graphs align with known anatomy (when available)
- SAE features are interpretable
- Alignment scores achieve >80% of noise ceiling
- Counterfactual effects are consistent with expectations

---

## Next Steps

1. **Set up dependencies:** Install all required packages
2. **Create module stubs:** Empty files with class definitions
3. **Implement in order:** graph_builder → energy_flow → ... → reporting
4. **Test continuously:** Write tests alongside implementation
5. **Integrate incrementally:** Add hooks to training/eval as modules complete
6. **Document thoroughly:** Update docs with each module
7. **Validate scientifically:** Test on known systems (Lorenz, etc.)

---

**Let's build the most comprehensive mechanistic interpretability suite for neural foundation models!** 🧠🔬🚀
