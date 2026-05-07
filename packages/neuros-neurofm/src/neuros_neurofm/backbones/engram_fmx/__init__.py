"""
ENGRAM-FMx: Energy-guided Neural Generative Recurrent Attractor Model.

An experimental backbone for NeuroFMx combining:
- Selective State-Space Models (SSM) for efficient temporal propagation
- Perceiver-style latent workspace compression
- Hopfield-style energy-guided attractor memory
- Neural-operator (spectral/FFT) latent dynamics
- Sparse anchor attention for exact grounding
- Gated fusion with controller routing
"""

from neuros_neurofm.backbones.engram_fmx.config import ENGRAMFMxConfig
from neuros_neurofm.backbones.engram_fmx.block import ENGRAMBlock, ENGRAMBlockOutput
from neuros_neurofm.backbones.engram_fmx.backbone import ENGRAMBackbone, ENGRAMBackboneOutput

__all__ = [
    "ENGRAMFMxConfig",
    "ENGRAMBlock",
    "ENGRAMBlockOutput",
    "ENGRAMBackbone",
    "ENGRAMBackboneOutput",
]
