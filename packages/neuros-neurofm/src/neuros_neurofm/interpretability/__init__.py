"""
Mechanistic Interpretability for NeuroFMx

Tools for understanding and analyzing the learned representations:
- Neuron activation analysis
- Circuit discovery via interventions
- Sparse autoencoders for feature decomposition
- Gradient-based attribution
- Latent space visualization
"""

from neuros_neurofm.interpretability.neuron_analysis import NeuronActivationAnalyzer
from neuros_neurofm.interpretability.circuit_discovery import CircuitDiscovery
from neuros_neurofm.interpretability.sparse_autoencoder import SparseAutoencoder
from neuros_neurofm.interpretability.attribution import IntegratedGradients, TemporalGradCAM
from neuros_neurofm.interpretability.latent_viz import LatentSpaceVisualizer

__all__ = [
    'NeuronActivationAnalyzer',
    'CircuitDiscovery',
    'SparseAutoencoder',
    'IntegratedGradients',
    'TemporalGradCAM',
    'LatentSpaceVisualizer'
]
