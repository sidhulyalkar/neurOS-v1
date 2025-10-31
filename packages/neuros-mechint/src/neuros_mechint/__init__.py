"""
neuros-mechint: Mechanistic Interpretability Toolbox

World's most comprehensive mechanistic interpretability suite for neural networks.

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

__version__ = "0.1.0"

from neuros_mechint.neuron_analysis import NeuronActivationAnalyzer
from neuros_mechint.circuit_discovery import CircuitDiscovery
from neuros_mechint.sparse_autoencoder import SparseAutoencoder

# Hierarchical SAE and Concept Discovery
from neuros_mechint.concept_sae import (
    HierarchicalSAE,
    ConceptDictionary,
    ConceptLabel,
    CausalSAEProbe,
)

# SAE Training Suite
from neuros_mechint.sae_training import (
    ActivationCache,
    MultiLayerSAETrainer,
    SAETrainingPipeline
)

# SAE Visualization Suite (optional - requires matplotlib)
try:
    from neuros_mechint.sae_visualization import (
        SAEVisualizer,
        MultiLayerSAEVisualizer
    )
    _HAS_VISUALIZATION = True
except ImportError:
    SAEVisualizer = None
    MultiLayerSAEVisualizer = None
    _HAS_VISUALIZATION = False

# Feature Analysis Suite
try:
    from neuros_mechint.feature_analysis import (
        FeatureAttributionAnalyzer,
        TemporalDynamicsAnalyzer,
        CausalImportanceAnalyzer,
        FeatureClusteringAnalyzer,
        FeatureSteeringAnalyzer
    )
except ImportError:
    FeatureAttributionAnalyzer = None
    TemporalDynamicsAnalyzer = None
    CausalImportanceAnalyzer = None
    FeatureClusteringAnalyzer = None
    FeatureSteeringAnalyzer = None

# Causal Graph Builder Suite
from neuros_mechint.graph_builder import (
    CausalGraph,
    TimeVaryingGraph,
    PerturbationEffect,
    AlignmentScore,
    CausalGraphBuilder,
    CausalGraphVisualizer,
    build_and_visualize_graph
)

# Energy Flow and Information Landscape Suite
try:
    from neuros_mechint.energy_flow import (
        MutualInformationEstimate,
        InformationPlane,
        EnergyFunction,
        Basin,
        EntropyProductionEstimate,
        MINENetwork,
        InformationFlowAnalyzer,
        EnergyLandscape,
        EntropyProduction,
        compute_information_plane_trajectory,
        LANDAUER_LIMIT,
        LandauerAnalysis,
        LandauerAnalyzer,
        SteadyStateMetrics,
        NESSAnalysis,
        NESSAnalyzer,
        FluctuationTheoremResult,
        FluctuationTheoremAnalyzer,
    )
except ImportError:
    MutualInformationEstimate = None
    InformationPlane = None
    EnergyFunction = None
    Basin = None
    EntropyProductionEstimate = None
    MINENetwork = None
    InformationFlowAnalyzer = None
    EnergyLandscape = None
    EntropyProduction = None
    compute_information_plane_trajectory = None
    LANDAUER_LIMIT = None
    LandauerAnalysis = None
    LandauerAnalyzer = None
    SteadyStateMetrics = None
    NESSAnalysis = None
    NESSAnalyzer = None
    FluctuationTheoremResult = None
    FluctuationTheoremAnalyzer = None

# Training/Evaluation Hooks
from neuros_mechint.hooks import (
    MechIntConfig,
    ActivationSampler,
    MechIntHooks,
    EvalMechIntRunner
)
try:
    from neuros_mechint.hooks import MechIntCallback, FastAPIIntegrationMixin
except ImportError:
    MechIntCallback = None
    FastAPIIntegrationMixin = None



# Advanced Attribution Methods (optional - requires matplotlib)
try:
    from neuros_mechint.attribution import (
        IntegratedGradients,
        DeepLIFT,
        GradientSHAP,
        GenerativePathAttribution,
        visualize_attributions
    )
except ImportError:
    IntegratedGradients = None
    DeepLIFT = None
    GradientSHAP = None
    GenerativePathAttribution = None
    visualize_attributions = None

# Brain Alignment Suite (optional - requires sklearn)
try:
    from neuros_mechint.alignment import (
        CCA,
        RegularizedCCA,
        KernelCCA,
        TimeVaryingCCA,
        RSA,
        RepresentationalDissimilarityMatrix,
        HierarchicalRSA,
        PLS,
        CrossValidatedPLS,
        NoiseCeiling,
        BootstrapCI,
        PermutationTest,
    )
except ImportError:
    CCA = None
    RegularizedCCA = None
    KernelCCA = None
    TimeVaryingCCA = None
    RSA = None
    RepresentationalDissimilarityMatrix = None
    HierarchicalRSA = None
    PLS = None
    CrossValidatedPLS = None
    NoiseCeiling = None
    BootstrapCI = None
    PermutationTest = None

# Dynamics Analysis (optional - requires matplotlib)
try:
    from neuros_mechint.dynamics import (
        DynamicsAnalyzer,
    )
except ImportError:
    DynamicsAnalyzer = None

# Counterfactual Interventions
try:
    from neuros_mechint.counterfactuals import (
        LatentSurgery,
        DoCalculusInterventions,
        SyntheticLesions,
        CounterfactualResult,
    )
except ImportError:
    LatentSurgery = None
    DoCalculusInterventions = None
    SyntheticLesions = None
    CounterfactualResult = None

# Meta-Dynamics (Training Trajectory Analysis - optional)
try:
    from neuros_mechint.meta_dynamics import (
        TrainingPhase,
        RepresentationalTrajectory,
        TrainingPhaseDetection,
        GradientAttribution,
    )
except ImportError:
    TrainingPhase = None
    RepresentationalTrajectory = None
    TrainingPhaseDetection = None
    GradientAttribution = None

# Geometry and Topology (optional - requires scipy)
try:
    from neuros_mechint.geometry_topology import (
        ManifoldGeometry,
        TopologicalAnalysis,
        ManifoldVisualization,
        PersistenceResults,
        ManifoldMetrics,
    )
except ImportError:
    ManifoldGeometry = None
    TopologicalAnalysis = None
    ManifoldVisualization = None
    PersistenceResults = None
    ManifoldMetrics = None

# Comprehensive Reporting (optional - requires matplotlib)
try:
    from neuros_mechint.reporting import (
        MechIntReporter,
        ReportSection,
        Figure,
    )
except ImportError:
    MechIntReporter = None
    ReportSection = None
    Figure = None

# Fractal Geometry Suite
from neuros_mechint.fractals import (
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

# Circuit Inference Suite (optional)
try:
    from neuros_mechint.circuits import (
        LatentCircuitModel,
        CircuitFitter,
        RecurrentDynamicsAnalyzer,
        DUNLModel,
        MixedSelectivityAnalyzer,
        FactorDecomposition,
        FeatureVisualizer,
        OptimalStimulus,
        ActivationMaximization,
        Edge,
        Circuit,
        AutomatedCircuitDiscovery,
        PatchEffect,
        PathPatchingResult,
        PathPatcher,
    )
except ImportError:
    LatentCircuitModel = None
    CircuitFitter = None
    RecurrentDynamicsAnalyzer = None
    DUNLModel = None
    MixedSelectivityAnalyzer = None
    FactorDecomposition = None
    FeatureVisualizer = None
    OptimalStimulus = None
    ActivationMaximization = None
    Edge = None
    Circuit = None
    AutomatedCircuitDiscovery = None
    PatchEffect = None
    PathPatchingResult = None
    PathPatcher = None

# Biophysical Modeling Suite (optional)
try:
    from neuros_mechint.biophysical import (
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
except ImportError:
    SurrogateGradient = None
    LeakyIntegrateFireNeuron = None
    IzhikevichNeuron = None
    HodgkinHuxleyNeuron = None
    SpikingNeuralNetwork = None
    DalesLawConstraint = None
    DalesLinear = None
    EINetworkClassifier = None
    RecurrentDalesNetwork = None
    DalesLossRegularizer = None

# Causal Interventions Suite
try:
    from neuros_mechint.interventions import (
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
except ImportError:
    ActivationPatcher = None
    ResidualStreamPatcher = None
    AttentionPatcher = None
    MLPPatcher = None
    NeuronAblation = None
    LayerAblation = None
    ComponentAblation = None
    AblationStudy = None
    PathAnalyzer = None
    InformationFlow = None
    InterventionCausalGraph = None

# Unified Result Data Structures (always available - no dependencies)
from neuros_mechint.results import (
    ResultProtocol,
    MechIntResult,
    CircuitResult,
    DynamicsResult,
    InformationResult,
    AlignmentResult,
    FractalResult,
    ResultCollection,
)

# Database for Result Storage and Caching
from neuros_mechint.database import (
    MechIntDatabase,
)

# Standardized Pipeline for Workflow Automation
from neuros_mechint.pipeline import (
    PipelineConfig,
    AnalysisStage,
    MechIntPipeline,
)

__all__ = [
    # Version
    '__version__',

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
    'LANDAUER_LIMIT',
    'LandauerAnalysis',
    'LandauerAnalyzer',
    'SteadyStateMetrics',
    'NESSAnalysis',
    'NESSAnalyzer',
    'FluctuationTheoremResult',
    'FluctuationTheoremAnalyzer',

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
    'CCA',
    'RegularizedCCA',
    'KernelCCA',
    'TimeVaryingCCA',
    'RSA',
    'RepresentationalDissimilarityMatrix',
    'HierarchicalRSA',
    'PLS',
    'CrossValidatedPLS',
    'NoiseCeiling',
    'BootstrapCI',
    'PermutationTest',

    # Dynamics Analysis
    'DynamicsAnalyzer',

    # Counterfactual Interventions
    'LatentSurgery',
    'DoCalculusInterventions',
    'SyntheticLesions',
    'CounterfactualResult',

    # Meta-Dynamics
    'TrainingPhase',
    'RepresentationalTrajectory',
    'TrainingPhaseDetection',
    'GradientAttribution',

    # Geometry and Topology
    'ManifoldGeometry',
    'TopologicalAnalysis',
    'ManifoldVisualization',
    'PersistenceResults',
    'ManifoldMetrics',

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
    'Edge',
    'Circuit',
    'AutomatedCircuitDiscovery',
    'PatchEffect',
    'PathPatchingResult',
    'PathPatcher',

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

    # Unified Result Data Structures
    'ResultProtocol',
    'MechIntResult',
    'CircuitResult',
    'DynamicsResult',
    'InformationResult',
    'AlignmentResult',
    'FractalResult',
    'ResultCollection',

    # Database
    'MechIntDatabase',

    # Pipeline
    'PipelineConfig',
    'AnalysisStage',
    'MechIntPipeline',
]
