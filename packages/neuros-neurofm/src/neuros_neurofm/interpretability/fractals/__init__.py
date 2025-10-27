"""
Fractal Geometry Suite for NeuroFMX

Comprehensive fractal analysis tools for neural foundation models, including:
- Temporal fractal metrics (Higuchi FD, DFA, Hurst exponent, spectral slope)
- Graph fractal metrics (box-covering dimension, power-law fitting)
- Multifractal spectrum analysis
- Fractal regularizers and priors for training
- Fractal stimulus generation (fBm, cascades, colored noise)
- Biophysical fractal simulators (fractional OU, dendrite growth)

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

__all__ = [
    # Metrics
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
]
