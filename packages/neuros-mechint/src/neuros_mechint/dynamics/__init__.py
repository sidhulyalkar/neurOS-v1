"""
Dynamical Systems Analysis Module for NeuroS-MechInt (FUTURE STRUCTURE).

⚠️ CURRENTLY A PLACEHOLDER ⚠️

This directory will contain modular dynamical systems components.
For now, use: from neuros_mechint.dynamics import DynamicsAnalyzer

## Planned Organization (Post-Refactor):

### Core Operators:
- koopman.py: KoopmanOperator, DMD (Dynamic Mode Decomposition)
- lyapunov.py: LyapunovAnalyzer (stability/chaos quantification)
- fixed_points.py: FixedPointFinder (attractor detection)
- manifold.py: ManifoldAnalyzer (geometry and curvature)
- phase_space.py: PhaseSpaceAnalyzer (phase portraits)

### Advanced Operators:
- neural_ode.py: NeuralODEIntegrator (continuous-time dynamics)
- slow_features.py: SlowFeatureAnalyzer (extract slow modes)
- granger.py: GrangerCausality (temporal causal graphs)
- bifurcation.py: BifurcationDetector (critical transitions)
- perturbation.py: PerturbationAnalyzer (response analysis)

### Unified Interface:
- analyzer.py: DynamicsAnalyzer (combines all operators)

## Usage (Current):
```python
from neuros_mechint.dynamics import DynamicsAnalyzer

analyzer = DynamicsAnalyzer(dt=0.01)
results = analyzer.estimate_koopman_operator(trajectories)
```

## Usage (Future):
```python
from neuros_mechint.dynamics import (
    KoopmanOperator,
    LyapunovAnalyzer,
    FixedPointFinder,
    DynamicsAnalyzer  # Unified interface
)

# Individual operators
koopman = KoopmanOperator()
koopman.fit(trajectories)

# Or unified interface
analyzer = DynamicsAnalyzer()
results = analyzer.run_all_analyses(trajectories)
```

Author: NeuroS Team
Date: 2025-10-30
Status: PLACEHOLDER - Using parent-level dynamics.py for now
"""

# This file intentionally left mostly empty
# Import is handled at package level in neuros_mechint/__init__.py
# which imports directly from neuros_mechint.dynamics (the .py file)

# Partial activation: Import completed modules
from .neural_ode import FlowFieldAnalysis, ODETrajectory, NeuralODEIntegrator
from .slow_features import SlowFeatureResult, SlowFeatureAnalyzer

__all__ = [
    # Neural ODE components
    'FlowFieldAnalysis',
    'ODETrajectory',
    'NeuralODEIntegrator',
    # Slow feature components
    'SlowFeatureResult',
    'SlowFeatureAnalyzer',
]

# When full refactoring is complete, add these imports:
# from .koopman import KoopmanOperator, DMD
# from .lyapunov import LyapunovAnalyzer
# from .fixed_points import FixedPointFinder
# from .manifold import ManifoldAnalyzer
# from .phase_space import PhaseSpaceAnalyzer
# from .granger import GrangerCausality
# from .bifurcation import BifurcationDetector
# from .perturbation import PerturbationAnalyzer
# from .analyzer import DynamicsAnalyzer
