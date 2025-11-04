"""
Fractal Geometry Suite for NeuroFMX

Comprehensive fractal analysis tools for neural foundation models, including:

## Temporal Fractal Analysis:
- Higuchi fractal dimension
- Detrended fluctuation analysis (DFA)
- Hurst exponent estimation
- Spectral slope analysis
- Wavelet-based multifractal analysis
- MF-DFA (Multifractal DFA)

## Criticality & Avalanches:
- Neuronal avalanche detection
- Branching process analysis
- Distance from criticality
- Self-organized criticality testing
- Power law fitting

## Graph Fractal Analysis:
- Box-covering dimension
- Power-law degree distribution
- Fractal connectivity patterns

## Training Tools:
- Fractal regularizers and priors
- Spectral smoothness constraints
- Multifractal loss functions

## Stimulus Generation:
- Fractional Brownian motion
- Colored noise (1/f^β)
- Multiplicative cascades
- Fractal patterns

## Biophysical Simulators:
- Fractional Ornstein-Uhlenbeck
- Dendritic growth models
- Fractal network generation

All implementations are GPU-accelerated and support batched computation.
"""

from .metrics import (
    HiguchiFractalDimension,
    DetrendedFluctuationAnalysis,
    HurstExponent,
    SpectralSlope,
    GraphFractalDimension,
    MultifractalSpectrum,
    FractalMetricsBundle,
)

from .regularizers import (
    SpectralPrior,
    MultifractalSmoothness,
    GraphFractalityPrior,
    FractalRegularizationLoss,
)

from .stimuli import (
    FractionalBrownianMotion,
    ColoredNoise,
    MultiplicativeCascade,
    FractalPatterns,
)

from .simulators import (
    FractionalOU,
    DendriteGrowthSimulator,
    FractalNetworkModel,
)

from .probes import (
    LatentFDTracker,
    AttentionFractalCoupling,
    CausalScaleAblation,
)

# New advanced modules
from .criticality import (
    NeuronalAvalanche,
    BranchingProcess,
    CriticalityDetector,
    SelfOrganizedCriticality,
    AvalancheStatistics,
    CriticalityMetrics,
)

from .wavelet_multifractal import (
    WaveletMultifractal,
    MultifractalDetrendedFluctuationAnalysis,
    MultifractalTemporalCorrelation,
    MultifractalSpectrum as WaveletMultifractalSpectrum,
)

__all__ = [
    # Basic Metrics
    'HiguchiFractalDimension',
    'DetrendedFluctuationAnalysis',
    'HurstExponent',
    'SpectralSlope',
    'GraphFractalDimension',
    'MultifractalSpectrum',
    'FractalMetricsBundle',

    # Regularizers
    'SpectralPrior',
    'MultifractalSmoothness',
    'GraphFractalityPrior',
    'FractalRegularizationLoss',

    # Stimuli
    'FractionalBrownianMotion',
    'ColoredNoise',
    'MultiplicativeCascade',
    'FractalPatterns',

    # Simulators
    'FractionalOU',
    'DendriteGrowthSimulator',
    'FractalNetworkModel',

    # Probes
    'LatentFDTracker',
    'AttentionFractalCoupling',
    'CausalScaleAblation',

    # Criticality Analysis
    'NeuronalAvalanche',
    'BranchingProcess',
    'CriticalityDetector',
    'SelfOrganizedCriticality',
    'AvalancheStatistics',
    'CriticalityMetrics',

    # Wavelet Multifractal
    'WaveletMultifractal',
    'MultifractalDetrendedFluctuationAnalysis',
    'MultifractalTemporalCorrelation',
    'WaveletMultifractalSpectrum',
]

__version__ = '2.0.0'
__author__ = 'NeuroS Team'
