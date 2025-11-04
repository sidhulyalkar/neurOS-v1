"""
Dynamical Systems Analysis Module for NeuroS-MechInt

This module provides comprehensive tools for analyzing dynamical systems,
including spectral, geometric, topological, and information-theoretic methods.

## Quick Start:

### Unified Interface (Recommended):
```python
from neuros_mechint.dynamics import DynamicsAnalyzer

analyzer = DynamicsAnalyzer(dt=0.01)

# Run all analyses
results = analyzer.run_all_analyses(trajectories)

# Or individual analyses
koopman = analyzer.estimate_koopman_operator(trajectories)
lyapunov = analyzer.compute_lyapunov_exponents(trajectories)
manifold = analyzer.analyze_manifold(trajectories)
```

### Individual Operators:
```python
from neuros_mechint.dynamics import (
    KoopmanOperator,
    LyapunovAnalyzer,
    FixedPointFinder,
    ManifoldAnalyzer,
    PhaseSpaceAnalyzer
)

# Create and use individual operators
koopman = KoopmanOperator(dt=0.01)
result = koopman.fit(trajectories, method="standard")

lyapunov = LyapunovAnalyzer(dt=0.01)
result = lyapunov.compute_exponents(trajectories)
```

## Module Organization:

### Core Operators:
- `KoopmanOperator`: Koopman operator theory and DMD variants
- `LyapunovAnalyzer`: Lyapunov exponents and stability analysis
- `FixedPointFinder`: Fixed point detection and classification
- `ManifoldAnalyzer`: Manifold geometry and topology
- `PhaseSpaceAnalyzer`: Phase space structure and attractors

### Advanced Operators:
- `GrangerCausality`: Temporal causal relationships
- `BifurcationDetector`: Bifurcation and critical transition detection
- `PerturbationAnalyzer`: Perturbation response and sensitivity
- `NeuralODEIntegrator`: Neural ODE integration
- `SlowFeatureAnalyzer`: Slow feature extraction

### Novel Methods:
- `RecurrenceAnalyzer`: Recurrence plots and RQA
- `TransferOperator`: Transfer operator and Perron-Frobenius methods
- `SynchronizationAnalyzer`: Synchronization in coupled systems
- `InformationAnalyzer`: Information-theoretic measures

### Additional Tools:
- `OptimalTransport`: Wasserstein distances and optimal transport
- `SpectralAnalyzer`: Spectral analysis and wavelets
- `ReservoirComputing`: Echo state networks and reservoir methods

Author: NeuroS Team
Date: 2025-01-03
"""

# ==================== Unified Interface ====================
from .analyzer import DynamicsAnalyzer

# ==================== Core Operators ====================
from .koopman import KoopmanOperator, KoopmanResult
from .lyapunov import (
    LyapunovAnalyzer,
    LyapunovResult,
    LyapunovFunctionResult
)
from .fixed_points import (
    FixedPointFinder,
    FixedPoint,
    PeriodicOrbit,
    FixedPointResult
)
from .manifold import ManifoldAnalyzer, ManifoldResult
from .phase_space import (
    PhaseSpaceAnalyzer,
    PhaseSpaceResult,
    PoincareSection,
    AttractorResult
)

# ==================== Advanced Operators ====================
from .granger import (
    GrangerCausality,
    GrangerResult,
    CausalGraph
)
from .bifurcation import (
    BifurcationDetector,
    BifurcationResult,
    BifurcationPoint,
    EarlyWarningSignals
)
from .perturbation import (
    PerturbationAnalyzer,
    PerturbationResponse,
    SensitivityResult,
    RobustnessResult
)
from .neural_ode import (
    NeuralODEIntegrator,
    FlowFieldAnalysis,
    ODETrajectory
)
from .slow_features import (
    SlowFeatureAnalyzer,
    SlowFeatureResult
)

# ==================== Novel Methods ====================
from .recurrence import RecurrenceAnalyzer, RecurrenceResult
from .transfer_operator import (
    TransferOperator,
    TransferOperatorResult,
    TransitionPath
)
from .synchronization import (
    SynchronizationAnalyzer,
    SynchronizationResult,
    PhaseResult
)
from .information import (
    InformationAnalyzer,
    InformationResult,
    InformationDecomposition
)

# ==================== Additional Tools ====================
from .optimal_transport import OptimalTransport, OptimalTransportResult
from .spectral import SpectralAnalyzer, SpectralResult
from .reservoir import ReservoirComputing, ReservoirResult

# ==================== Public API ====================
__all__ = [
    # Unified Interface
    'DynamicsAnalyzer',

    # Core Operators
    'KoopmanOperator',
    'KoopmanResult',
    'LyapunovAnalyzer',
    'LyapunovResult',
    'LyapunovFunctionResult',
    'FixedPointFinder',
    'FixedPoint',
    'PeriodicOrbit',
    'FixedPointResult',
    'ManifoldAnalyzer',
    'ManifoldResult',
    'PhaseSpaceAnalyzer',
    'PhaseSpaceResult',
    'PoincareSection',
    'AttractorResult',

    # Advanced Operators
    'GrangerCausality',
    'GrangerResult',
    'CausalGraph',
    'BifurcationDetector',
    'BifurcationResult',
    'BifurcationPoint',
    'EarlyWarningSignals',
    'PerturbationAnalyzer',
    'PerturbationResponse',
    'SensitivityResult',
    'RobustnessResult',
    'NeuralODEIntegrator',
    'FlowFieldAnalysis',
    'ODETrajectory',
    'SlowFeatureAnalyzer',
    'SlowFeatureResult',

    # Novel Methods
    'RecurrenceAnalyzer',
    'RecurrenceResult',
    'TransferOperator',
    'TransferOperatorResult',
    'TransitionPath',
    'SynchronizationAnalyzer',
    'SynchronizationResult',
    'PhaseResult',
    'InformationAnalyzer',
    'InformationResult',
    'InformationDecomposition',

    # Additional Tools
    'OptimalTransport',
    'OptimalTransportResult',
    'SpectralAnalyzer',
    'SpectralResult',
    'ReservoirComputing',
    'ReservoirResult',
]

# Version info
__version__ = '2.0.0'
__author__ = 'NeuroS Team'
