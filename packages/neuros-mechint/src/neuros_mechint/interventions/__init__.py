"""
Causal Interventions for Mechanistic Interpretability

Tools for performing causal interventions on neural networks to understand
information flow and computational structure. Includes activation patching,
ablation studies, and path analysis.
"""

from .patching import (
    ActivationPatcher,
    ResidualStreamPatcher,
    AttentionPatcher,
    MLPPatcher,
)

from .ablation import (
    NeuronAblation,
    LayerAblation,
    ComponentAblation,
    AblationStudy,
)

from .paths import (
    PathAnalyzer,
    InformationFlow,
    CausalGraph,
)

__all__ = [
    # Patching
    'ActivationPatcher',
    'ResidualStreamPatcher',
    'AttentionPatcher',
    'MLPPatcher',
    # Ablation
    'NeuronAblation',
    'LayerAblation',
    'ComponentAblation',
    'AblationStudy',
    # Path analysis
    'PathAnalyzer',
    'InformationFlow',
    'CausalGraph',
]
