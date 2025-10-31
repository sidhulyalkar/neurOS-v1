"""
Latent Circuit Inference for NeuroFMX

Extract interpretable low-dimensional circuits from high-dimensional neural representations.
Includes latent RNN models, DUNL disentanglement, feature visualization, and circuit extraction.

References:
    - Langdon & Engel (2025): Latent circuit models
    - Sussillo & Barak (2013): Opening the black box
    - Olah et al. (2018): Feature visualization
"""

from .latent_rnn import (
    LatentCircuitModel,
    CircuitFitter,
    RecurrentDynamicsAnalyzer,
)

from .dunl import (
    DUNLModel,
    MixedSelectivityAnalyzer,
    FactorDecomposition,
)

from .feature_viz import (
    FeatureVisualizer,
    OptimalStimulus,
    ActivationMaximization,
)

# Automated Circuit Discovery (ACDC)
from .acdc import (
    Edge,
    Circuit,
    AutomatedCircuitDiscovery,
)

# Path Patching for Causal Circuit Discovery
from .path_patching import (
    PatchEffect,
    PathPatchingResult,
    PathPatcher,
)

# Circuit Comparison for Cross-Model Analysis
from .circuit_comparator import (
    CircuitComparison,
    MultiCircuitAnalysis,
    CircuitComparator,
)

# Motif Detection for Recurring Patterns
from .motif_detection import (
    MotifInstance,
    MotifAnalysis,
    MotifDetector,
)

# Circuit extraction module - additional components to be implemented
# from .circuit_extraction import (
#     CircuitExtractor,
#     EICircuitDiagram,
# )

__all__ = [
    # Latent RNN
    'LatentCircuitModel',
    'CircuitFitter',
    'RecurrentDynamicsAnalyzer',
    # DUNL
    'DUNLModel',
    'MixedSelectivityAnalyzer',
    'FactorDecomposition',
    # Feature Visualization
    'FeatureVisualizer',
    'OptimalStimulus',
    'ActivationMaximization',
    # ACDC (Automated Circuit Discovery)
    'Edge',
    'Circuit',
    'AutomatedCircuitDiscovery',
    # Path Patching
    'PatchEffect',
    'PathPatchingResult',
    'PathPatcher',
    # Circuit Comparison
    'CircuitComparison',
    'MultiCircuitAnalysis',
    'CircuitComparator',
    # Motif Detection
    'MotifInstance',
    'MotifAnalysis',
    'MotifDetector',
    # Circuit Extraction (not yet implemented)
    # 'CircuitExtractor',
    # 'EICircuitDiagram',
]
