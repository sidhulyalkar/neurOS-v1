"""
Mechanistic Interpretability for NeuroFMX

World's most comprehensive mechanistic interpretability suite for neural foundation models.

Core Components:
- Neuron activation analysis
- Circuit discovery via interventions
- Sparse autoencoders for feature decomposition
- Hierarchical SAE with concept dictionaries
- SAE training suite for multi-layer analysis
- SAE visualization tools
- Feature analysis and attribution

Fractal Geometry Suite:
- Temporal fractal metrics (Higuchi FD, DFA, Hurst, spectral slope)
- Graph fractal dimension (box-covering algorithm)
- Multifractal spectrum analysis
- Fractal regularizers and priors for training
- Fractal stimulus generation (fBm, cascades, colored noise)
- Biophysical fractal simulators (fractional OU, dendrite growth)
- Real-time fractal probes (latent FD tracking, attention-fractal coupling)

Circuit Inference:
- Latent RNN extraction (minimal computational circuits)
- DUNL sparse coding for mixed selectivity decomposition
- Feature visualization via activation maximization
- Recurrent dynamics analysis (fixed points, stability)

Biophysical Modeling:
- Differentiable spiking networks (LIF, Izhikevich, Hodgkin-Huxley)
- Surrogate gradients for backpropagation through spikes
- Dale's law enforcement (E/I neuron separation)
- Biologically-constrained learning

Causal Interventions:
- Activation patching for causal tracing
- Systematic ablation studies (neurons, layers, components)
- Path analysis and information flow
- Causal graph construction

Brain Alignment:
- CCA (Canonical Correlation Analysis)
- RSA (Representational Similarity Analysis)
- Procrustes alignment
- Noise ceiling estimation

Dynamical Systems:
- Koopman operator analysis
- Lyapunov exponents
- Manifold geometry
- Phase portraits

Causal Analysis:
- Temporal causal graphs (Granger causality)
- Perturbation-based causal inference
- Counterfactual interventions
- Do-calculus

Meta-Dynamics:
- Training trajectory tracking
- Feature emergence detection
- Representational drift
- Phase transition detection

Topology & Geometry:
- Persistent homology
- Betti numbers
- Manifold curvature
- Intrinsic dimensionality

Information Theory:
- Information flow analysis (mutual information)
- Energy landscape estimation
- Basin detection
- Entropy production

Reporting & Integration:
- Automated HTML report generation
- MLflow/W&B integration
- Training/evaluation hooks
- PyTorch Lightning callbacks
- FastAPI endpoints for real-time interpretation
"""

from neuros_neurofm.interpretability.neuron_analysis import NeuronActivationAnalyzer
from neuros_neurofm.interpretability.circuit_discovery import CircuitDiscovery
from neuros_neurofm.interpretability.sparse_autoencoder import SparseAutoencoder

# Hierarchical SAE and Concept Discovery
from neuros_neurofm.interpretability.concept_sae import (
    HierarchicalSAE,
    ConceptDictionary,
    ConceptLabel,
    CausalSAEProbe,
)

# SAE Training Suite
from neuros_neurofm.interpretability.sae_training import (
    ActivationCache,
    MultiLayerSAETrainer,
    SAETrainingPipeline
)

# SAE Visualization Suite
from neuros_neurofm.interpretability.sae_visualization import (
    SAEVisualizer,
    MultiLayerSAEVisualizer
)

# Feature Analysis Suite
from neuros_neurofm.interpretability.feature_analysis import (
    FeatureAttributionAnalyzer,
    TemporalDynamicsAnalyzer,
    CausalImportanceAnalyzer,
    FeatureClusteringAnalyzer,
    FeatureSteeringAnalyzer
)

# Causal Graph Builder Suite
from neuros_neurofm.interpretability.graph_builder import (
    CausalGraph,
    TimeVaryingGraph,
    PerturbationEffect,
    AlignmentScore,
    CausalGraphBuilder,
    CausalGraphVisualizer,
    build_and_visualize_graph
)

# Energy Flow and Information Landscape Suite
from neuros_neurofm.interpretability.energy_flow import (
    MutualInformationEstimate,
    InformationPlane,
    EnergyFunction,
    Basin,
    EntropyProductionEstimate,
    MINENetwork,
    InformationFlowAnalyzer,
    EnergyLandscape,
    EntropyProduction,
    compute_information_plane_trajectory
)

# Training/Evaluation Hooks
from neuros_neurofm.interpretability.hooks import (
    MechIntConfig,
    ActivationSampler,
    MechIntHooks,
    EvalMechIntRunner,
    MechIntCallback,
    FastAPIIntegrationMixin
)

# Advanced Attribution Methods
from neuros_neurofm.interpretability.attribution import (
    IntegratedGradients,
    DeepLIFT,
    GradientSHAP,
    GenerativePathAttribution,
    visualize_attributions
)

# Brain Alignment Suite
from neuros_neurofm.interpretability.alignment import (
    CCAAlignment,
    RSAAlignment,
    ProcrustesAlignment,
)

# Dynamics Analysis
from neuros_neurofm.interpretability.dynamics import (
    DynamicsAnalyzer,
    KoopmanOperator,
    LyapunovAnalyzer,
)

# Counterfactual Interventions
from neuros_neurofm.interpretability.counterfactuals import (
    LatentSurgery,
    DoCalculusEngine,
    SyntheticLesion,
)

# Meta-Dynamics (Training Trajectory Analysis)
from neuros_neurofm.interpretability.meta_dynamics import (
    MetaDynamicsTracker,
    CheckpointComparison,
    TrainingPhase,
)

# Geometry and Topology
from neuros_neurofm.interpretability.geometry_topology import (
    ManifoldAnalyzer,
    TopologyAnalyzer,
    CurvatureEstimator,
)

# Comprehensive Reporting
from neuros_neurofm.interpretability.reporting import (
    MechIntReporter,
    ReportSection,
    Figure,
)

# Fractal Geometry Suite
from neuros_neurofm.interpretability.fractals import (
    HiguchiFractalDimension,
    DetrendedFluctuationAnalysis,
    HurstExponent,
    SpectralSlope,
    GraphFractalDimension,
    MultifractalSpectrum,
    FractalMetricsBundle,
    SpectralPrior,
    MultifractalSmoothness,
    GraphFractalityPrior,
    FractalRegularizationLoss,
    FractionalBrownianMotion,
    ColoredNoise,
    MultiplicativeCascade,
    FractalPatterns,
    FractionalOU,
    DendriteGrowthSimulator,
    FractalNetworkModel,
    LatentFDTracker,
    AttentionFractalCoupling,
    CausalScaleAblation,
)

# Circuit Inference Suite
from neuros_neurofm.interpretability.circuits import (
    LatentCircuitModel,
    CircuitFitter,
    RecurrentDynamicsAnalyzer,
    DUNLModel,
    MixedSelectivityAnalyzer,
    FactorDecomposition,
    FeatureVisualizer,
    OptimalStimulus,
    ActivationMaximization,
)

# Biophysical Modeling Suite
from neuros_neurofm.interpretability.biophysical import (
    SurrogateGradient,
    LeakyIntegrateFireNeuron,
    IzhikevichNeuron,
    HodgkinHuxleyNeuron,
    SpikingNeuralNetwork,
    DalesLawConstraint,
    DalesLinear,
    EINetworkClassifier,
    RecurrentDalesNetwork,
    DalesLossRegularizer,
)

# Causal Interventions Suite
from neuros_neurofm.interpretability.interventions import (
    ActivationPatcher,
    ResidualStreamPatcher,
    AttentionPatcher,
    MLPPatcher,
    NeuronAblation,
    LayerAblation,
    ComponentAblation,
    AblationStudy,
    PathAnalyzer,
    InformationFlow,
    CausalGraph as InterventionCausalGraph,  # Alias to avoid conflict
)

try:
    from neuros_neurofm.interpretability.latent_viz import LatentSpaceVisualizer
    _has_latent_viz = True
except ImportError:
    _has_latent_viz = False

__all__ = [
    # Core components
    'NeuronActivationAnalyzer',
    'CircuitDiscovery',
    'SparseAutoencoder',

    # Hierarchical SAE and Concept Discovery
    'HierarchicalSAE',
    'ConceptDictionary',
    'ConceptLabel',
    'CausalSAEProbe',

    # SAE Training
    'ActivationCache',
    'MultiLayerSAETrainer',
    'SAETrainingPipeline',

    # SAE Visualization
    'SAEVisualizer',
    'MultiLayerSAEVisualizer',

    # Feature Analysis
    'FeatureAttributionAnalyzer',
    'TemporalDynamicsAnalyzer',
    'CausalImportanceAnalyzer',
    'FeatureClusteringAnalyzer',
    'FeatureSteeringAnalyzer',

    # Causal Graph Builder
    'CausalGraph',
    'TimeVaryingGraph',
    'PerturbationEffect',
    'AlignmentScore',
    'CausalGraphBuilder',
    'CausalGraphVisualizer',
    'build_and_visualize_graph',

    # Energy Flow and Information Landscape
    'MutualInformationEstimate',
    'InformationPlane',
    'EnergyFunction',
    'Basin',
    'EntropyProductionEstimate',
    'MINENetwork',
    'InformationFlowAnalyzer',
    'EnergyLandscape',
    'EntropyProduction',
    'compute_information_plane_trajectory',

    # Training/Evaluation Hooks
    'MechIntConfig',
    'ActivationSampler',
    'MechIntHooks',
    'EvalMechIntRunner',
    'MechIntCallback',
    'FastAPIIntegrationMixin',

    # Advanced Attribution Methods
    'IntegratedGradients',
    'DeepLIFT',
    'GradientSHAP',
    'GenerativePathAttribution',
    'visualize_attributions',

    # Brain Alignment
    'CCAAlignment',
    'RSAAlignment',
    'ProcrustesAlignment',

    # Dynamics Analysis
    'DynamicsAnalyzer',
    'KoopmanOperator',
    'LyapunovAnalyzer',

    # Counterfactual Interventions
    'LatentSurgery',
    'DoCalculusEngine',
    'SyntheticLesion',

    # Meta-Dynamics
    'MetaDynamicsTracker',
    'CheckpointComparison',
    'TrainingPhase',

    # Geometry and Topology
    'ManifoldAnalyzer',
    'TopologyAnalyzer',
    'CurvatureEstimator',

    # Reporting
    'MechIntReporter',
    'ReportSection',
    'Figure',

    # Fractal Geometry Suite
    'HiguchiFractalDimension',
    'DetrendedFluctuationAnalysis',
    'HurstExponent',
    'SpectralSlope',
    'GraphFractalDimension',
    'MultifractalSpectrum',
    'FractalMetricsBundle',
    'SpectralPrior',
    'MultifractalSmoothness',
    'GraphFractalityPrior',
    'FractalRegularizationLoss',
    'FractionalBrownianMotion',
    'ColoredNoise',
    'MultiplicativeCascade',
    'FractalPatterns',
    'FractionalOU',
    'DendriteGrowthSimulator',
    'FractalNetworkModel',
    'LatentFDTracker',
    'AttentionFractalCoupling',
    'CausalScaleAblation',

    # Circuit Inference Suite
    'LatentCircuitModel',
    'CircuitFitter',
    'RecurrentDynamicsAnalyzer',
    'DUNLModel',
    'MixedSelectivityAnalyzer',
    'FactorDecomposition',
    'FeatureVisualizer',
    'OptimalStimulus',
    'ActivationMaximization',

    # Biophysical Modeling Suite
    'SurrogateGradient',
    'LeakyIntegrateFireNeuron',
    'IzhikevichNeuron',
    'HodgkinHuxleyNeuron',
    'SpikingNeuralNetwork',
    'DalesLawConstraint',
    'DalesLinear',
    'EINetworkClassifier',
    'RecurrentDalesNetwork',
    'DalesLossRegularizer',

    # Causal Interventions Suite
    'ActivationPatcher',
    'ResidualStreamPatcher',
    'AttentionPatcher',
    'MLPPatcher',
    'NeuronAblation',
    'LayerAblation',
    'ComponentAblation',
    'AblationStudy',
    'PathAnalyzer',
    'InformationFlow',
    'InterventionCausalGraph',
]

if _has_latent_viz:
    __all__.append('LatentSpaceVisualizer')
