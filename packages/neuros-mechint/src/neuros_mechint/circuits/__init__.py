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

from .circuit_extraction import (
    CircuitExtractor,
    EICircuitDiagram,
    MotifFinder,
)

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
    # Circuit Extraction
    'CircuitExtractor',
    'EICircuitDiagram',
    'MotifFinder',
]
