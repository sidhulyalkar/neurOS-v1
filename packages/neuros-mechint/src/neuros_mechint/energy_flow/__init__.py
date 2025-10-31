"""
Energy Flow and Thermodynamics of Computation Module.

This module combines information-theoretic analysis with thermodynamic principles
to understand energy dissipation and information processing in neural networks.

## Components:

### Information Theory:
- InformationFlowAnalyzer: Mutual information estimation (MINE, k-NN, histogram)
- Information Plane: Tishby's I(X;Z) vs I(Z;Y) analysis
- Information Bottleneck: Optimal compression-prediction tradeoff

### Energy Landscape:
- EnergyLandscape: Estimate latent space energy function U(z)
- Basin Detection: Find stable states and energy barriers
- Landscape Visualization: 2D/3D energy surface plots

### Entropy Production:
- EntropyProduction: Measure dS/dt along trajectories
- Dissipation Rate: Total energy dissipation
- Nonequilibrium Score: Distance from equilibrium

### Thermodynamics of Computation (NEW):
- Landauer's Principle: Minimum energy per bit erased (E_min = kT ln(2))
- Information Erasure: Measure bits erased during forward pass
- Reversibility Analysis: Score operations by thermodynamic reversibility
- Per-Layer Cost: Thermodynamic breakdown by layer

## References:
- Tishby & Zaslavsky (2015): Information bottleneck
- Landauer (1961): Irreversibility and heat generation
- Bennett (1973): Logical reversibility of computation
- Seifert (2012): Stochastic thermodynamics

Author: NeuroS Team
Date: 2025-10-30
"""

# Import all components from parent energy_flow.py
# We need to import from the .py file at the parent level, not from this directory
# Using importlib to load the module file directly
import importlib.util
import sys
from pathlib import Path

# Get path to parent energy_flow.py module file
parent_module_path = Path(__file__).parent.parent / "energy_flow.py"

# Load the module
spec = importlib.util.spec_from_file_location("_energy_flow_parent", parent_module_path)
_energy_flow_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(_energy_flow_module)

# Import components from the loaded module
MutualInformationEstimate = _energy_flow_module.MutualInformationEstimate
InformationPlane = _energy_flow_module.InformationPlane
EnergyFunction = _energy_flow_module.EnergyFunction
Basin = _energy_flow_module.Basin
EntropyProductionEstimate = _energy_flow_module.EntropyProductionEstimate
MINENetwork = _energy_flow_module.MINENetwork
InformationFlowAnalyzer = _energy_flow_module.InformationFlowAnalyzer
EnergyLandscape = _energy_flow_module.EnergyLandscape
EntropyProduction = _energy_flow_module.EntropyProduction
compute_information_plane_trajectory = _energy_flow_module.compute_information_plane_trajectory

# Import Landauer thermodynamics components
from .landauer import (
    LANDAUER_LIMIT,
    LandauerAnalysis,
    LandauerAnalyzer,
)

__all__ = [
    # Data structures (from energy_flow.py)
    'MutualInformationEstimate',
    'InformationPlane',
    'EnergyFunction',
    'Basin',
    'EntropyProductionEstimate',

    # Main analyzers (from energy_flow.py)
    'MINENetwork',
    'InformationFlowAnalyzer',
    'EnergyLandscape',
    'EntropyProduction',

    # Utility functions (from energy_flow.py)
    'compute_information_plane_trajectory',

    # Landauer thermodynamics (from landauer.py)
    'LANDAUER_LIMIT',
    'LandauerAnalysis',
    'LandauerAnalyzer',
]
