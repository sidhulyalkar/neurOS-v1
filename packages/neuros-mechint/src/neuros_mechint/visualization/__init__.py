"""
Enhanced Visualization Module for NeuroS-MechInt.

Provides interactive visualizations for:
- 3D Energy Landscapes (Bokeh)
- Animated Information Planes
- Multi-panel comparisons
- Dynamic circuit visualizations
- Interactive 3D Brain Activity (Plotly)
- Connectivity graphs and network layouts
- Criticality avalanche visualization
- Cross-species comparison views
- Temporal dynamics animation
- Multifractal analysis visualization
- Intervention effects (optogenetics, pharmacology, stimulation)
- Phase space trajectories

Author: NeuroS Team
Date: 2025-10-31
"""

from .enhanced_viz import EnhancedVisualizer
from .interactive_brain import (
    BrainRegion,
    BrainAtlas,
    Interactive3DBrain,
    ForceDirectedGraph,
    CriticalityVisualizer,
)
from .advanced_viz import (
    MultifractalVisualizer,
    CrossSpeciesVisualizer,
    InterventionVisualizer,
    TemporalDynamicsVisualizer,
)

__all__ = [
    'EnhancedVisualizer',
    'BrainRegion',
    'BrainAtlas',
    'Interactive3DBrain',
    'ForceDirectedGraph',
    'CriticalityVisualizer',
    'MultifractalVisualizer',
    'CrossSpeciesVisualizer',
    'InterventionVisualizer',
    'TemporalDynamicsVisualizer',
]
