"""
NeuroFM-X: Foundation Model for Neural Population Dynamics.

This package provides a state-of-the-art foundation model combining:
- Selective State-Space Models (Mamba/SSM) for linear-complexity sequence modeling
- Perceiver-IO for multi-modal fusion
- Population Transformers (PopT) for neural population aggregation
- Latent Diffusion for generative modeling
- Transfer learning adapters (Unit-ID, LoRA)
"""

__version__ = "0.1.0"

from neuros_neurofm.models.neurofmx import NeuroFMX
from neuros_neurofm.training.trainer import NeuroFMXTrainer

__all__ = [
    "NeuroFMX",
    "NeuroFMXTrainer",
    "__version__",
]
