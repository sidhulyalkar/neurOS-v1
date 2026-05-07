"""
ENGRAM-FMx Modules.

Individual components that compose the ENGRAM-FMx architecture.
"""

from neuros_neurofm.backbones.engram_fmx.modules.local_processing import LocalProcessingBlock
from neuros_neurofm.backbones.engram_fmx.modules.selective_ssm import SelectiveSSMBlock
from neuros_neurofm.backbones.engram_fmx.modules.latent_workspace import LatentWorkspace
from neuros_neurofm.backbones.engram_fmx.modules.attractor_memory import AttractorMemory
from neuros_neurofm.backbones.engram_fmx.modules.operator_dynamics import SpectralOperatorDynamics
from neuros_neurofm.backbones.engram_fmx.modules.sparse_anchor_attention import SparseAnchorAttention
from neuros_neurofm.backbones.engram_fmx.modules.gated_fusion import GatedFusion

__all__ = [
    "LocalProcessingBlock",
    "SelectiveSSMBlock",
    "LatentWorkspace",
    "AttractorMemory",
    "SpectralOperatorDynamics",
    "SparseAnchorAttention",
    "GatedFusion",
]
